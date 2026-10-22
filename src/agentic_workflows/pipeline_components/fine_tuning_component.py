from kfp.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel",
    packages_to_install=[
        "transformers==4.46.3", "peft==0.13.2", "trl==0.12.2",
        "accelerate>=1.0,<2", "bitsandbytes>=0.45.5,<0.49",
        "datasets>=4.1.1,<6", "pandas>=2.3.3", "tensorboard>=2.18,<3",
    ],
)
def fine_tuning_component(
    train_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    base_model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    learning_rate: float = 0.0003,
    num_train_epochs: float = 3.0,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    max_seq_length: int = 128,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    validation_size: float = 0.1,
    seed: int = 42,
    max_steps: int = -1,
) -> None:
    import json
    import logging
    import math
    import tempfile
    from pathlib import Path

    import pandas as pd
    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    logging.basicConfig(level=logging.INFO)
    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning requires a CUDA GPU. Run this step on Vertex AI or a CUDA machine.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    set_seed(seed)
    parameters = dict(
        base_model_id=base_model_id, learning_rate=learning_rate,
        num_train_epochs=num_train_epochs, lora_rank=lora_rank, lora_alpha=lora_alpha,
        max_seq_length=max_seq_length, batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        validation_size=validation_size, seed=seed, max_steps=max_steps,
    )
    frame = pd.read_csv(train_dataset.path)
    if len(frame) < 2:
        raise ValueError("At least two training rows are required for a validation split.")
    # CSV stores conversations as JSON; never pass these serialized strings to the trainer.
    conversations = [json.loads(value) for value in frame["messages"]]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # Keep EOS distinct from padding so the collator learns when to stop.
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    dataset = HFDataset.from_dict({"text": [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in conversations
    ]})
    split = dataset.train_test_split(test_size=validation_size, seed=seed)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    pretrained = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=quantization, device_map={"": 0},
        torch_dtype=torch.float16, attn_implementation="eager",
    )
    pretrained.config.use_cache = False
    pretrained = prepare_model_for_kbit_training(
        pretrained, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
    )
    adapted = get_peft_model(pretrained, lora)
    adapted.print_trainable_parameters()
    logging.info("LoRA modules: %s", [name for name, _ in adapted.named_modules() if "lora_A" in name])
    trainable, total = adapted.get_nb_trainable_parameters()
    Path(model.path).mkdir(parents=True, exist_ok=True)
    Path(metrics.path).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yoda-training-") as checkpoint_dir:
        config = SFTConfig(
            output_dir=checkpoint_dir, logging_dir=metrics.path, report_to=["tensorboard"],
            logging_steps=1, eval_strategy="epoch", save_strategy="no",
            num_train_epochs=num_train_epochs, max_steps=max_steps,
            learning_rate=learning_rate, per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            max_seq_length=max_seq_length, dataset_text_field="text", packing=False,
            dataset_kwargs={"add_special_tokens": False},
            optim="adamw_torch", fp16=True, bf16=False, seed=seed,
        )
        trainer = SFTTrainer(
            model=adapted, processing_class=tokenizer, args=config,
            train_dataset=split["train"], eval_dataset=split["test"],
        )
        result = trainer.train()
        scores = {**result.metrics, **trainer.evaluate()}
        trainer.save_model(model.path)
        tokenizer.save_pretrained(model.path)
        Path(model.path, "training_history.json").write_text(
            json.dumps(trainer.state.log_history, indent=2), encoding="utf-8",
        )
    scores.update(trainable_parameters=trainable, total_parameters=total,
                  trainable_fraction=trainable / total)
    for name, value in scores.items():
        if isinstance(value, (int, float)) and math.isfinite(value):
            metrics.log_metric(name, float(value))
    model.metadata.update(base_model_id=base_model_id, format="peft-lora")
    metrics.metadata["hyperparameters"] = parameters
    Path(model.path, "hyperparameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8",
    )
