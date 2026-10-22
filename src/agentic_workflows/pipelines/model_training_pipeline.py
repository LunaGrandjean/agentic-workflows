from kfp import dsl

from agentic_workflows.constants import RAW_DATASET_URI
from agentic_workflows.pipeline_components.data_transformation_component import (
    transform_yoda_data,
)
from agentic_workflows.pipeline_components.fine_tuning_component import fine_tuning_component
from agentic_workflows.pipeline_components.inference_component import inference_component
from agentic_workflows.pipeline_components.evaluation_component import evaluation_component


@dsl.pipeline(
    name="lulik-yoda-data-preparation",
    description="Prepare the Yoda sentences dataset for Phi-3 fine-tuning.",
)
def yoda_data_preparation_pipeline(
    raw_dataset_uri: str = RAW_DATASET_URI,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    transform_yoda_data(
        raw_dataset_uri=raw_dataset_uri,
        test_size=test_size,
        seed=seed,
    )


@dsl.pipeline(
    name="lulik-yoda-model-training",
    description="Prepare Yoda data, fine-tune Phi-3 with LoRA, predict, and evaluate with Ragas.",
)
def yoda_model_training_pipeline(
    raw_dataset_uri: str = RAW_DATASET_URI,
    test_size: float = 0.2,
    seed: int = 42,
    base_model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    learning_rate: float = 0.0003,
    num_train_epochs: float = 3.0,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    max_seq_length: int = 128,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    validation_size: float = 0.1,
    max_steps: int = -1,
    max_new_tokens: int = 64,
    max_samples: int = 0,
) -> None:
    preparation = transform_yoda_data(raw_dataset_uri=raw_dataset_uri, test_size=test_size, seed=seed)
    training = fine_tuning_component(
        train_dataset=preparation.outputs["train_dataset"],
        base_model_id=base_model_id, learning_rate=learning_rate,
        num_train_epochs=num_train_epochs, lora_rank=lora_rank, lora_alpha=lora_alpha,
        max_seq_length=max_seq_length, batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        validation_size=validation_size, seed=seed, max_steps=max_steps,
    )
    training.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
    training.set_cpu_limit("16").set_memory_limit("50G")
    inference = inference_component(
        model=training.outputs["model"], test_dataset=preparation.outputs["test_dataset"],
        max_new_tokens=max_new_tokens, max_samples=max_samples,
    )
    inference.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
    inference.set_cpu_limit("16").set_memory_limit("50G")
    evaluation_component(predictions=inference.outputs["predictions"])
