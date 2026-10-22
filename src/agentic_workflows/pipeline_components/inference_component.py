from kfp.dsl import Dataset, Input, Model, Output, component


@component(
    base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel",
    packages_to_install=[
        "transformers==4.46.3", "peft==0.13.2", "accelerate>=1.0,<2", "pandas>=2.3.3",
    ],
)
def inference_component(
    model: Input[Model],
    test_dataset: Input[Dataset],
    predictions: Output[Dataset],
    max_new_tokens: int = 64,
    max_samples: int = 0,
) -> None:
    import json
    from pathlib import Path

    import pandas as pd
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("The inference component requires a CUDA GPU.")
    if max_samples < 0 or max_new_tokens < 1:
        raise ValueError("max_samples must be nonnegative and max_new_tokens positive.")
    # Training saves adapters, not a merged copy of the multi-GB base model.
    adapter_config = PeftConfig.from_pretrained(model.path)
    base = AutoModelForCausalLM.from_pretrained(
        adapter_config.base_model_name_or_path, torch_dtype=torch.float16,
        device_map={"": 0}, attn_implementation="eager",
    )
    adapted = PeftModel.from_pretrained(base, model.path)
    adapted.config.use_cache = True
    adapted.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.path)
    frame = pd.read_csv(test_dataset.path)
    if max_samples:
        frame = frame.head(max_samples)
    rows = []
    for value in frame["messages"]:
        conversation = json.loads(value)
        sentence = next(message["content"] for message in conversation if message["role"] == "user")
        reference = next(message["content"] for message in conversation if message["role"] == "assistant")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": sentence}], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(adapted.device)
        with torch.inference_mode():
            output = adapted.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        rows.append(dict(user_input=sentence, reference=reference, extracted_response=response))
    Path(predictions.path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["user_input", "reference", "extracted_response"]).to_csv(predictions.path, index=False)
    predictions.metadata["num_samples"] = len(rows)
