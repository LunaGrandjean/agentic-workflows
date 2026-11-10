"""Final-project workflow; official splits and a paired base/LoRA comparison."""

from kfp import dsl

from agentic_workflows.pipeline_components.fine_tuning_component import fine_tuning_component
from agentic_workflows.pipeline_components.inference_component import inference_component
from agentic_workflows.pipeline_components.restaurant_data_component import prepare_restaurant_data
from agentic_workflows.pipeline_components.restaurant_evaluation_component import evaluate_restaurant_predictions


@dsl.pipeline(name="lulik-restaurant-description-training", description="GEM/E2E official splits -> Phi-3 QLoRA -> base/adapted predictions -> multi-reference ROUGE.")
def restaurant_training_pipeline(
    dataset_manifest_uri: str,
    train_limit: int = 4000, validation_limit: int = 200, test_limit: int = 200,
    evaluation_split: str = "validation", seed: int = 42,
    base_model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    learning_rate: float = 0.0001, num_train_epochs: float = 1.0,
    lora_rank: int = 8, lora_alpha: int = 16,
    max_seq_length: int = 256, batch_size: int = 2, gradient_accumulation_steps: int = 8,
    max_steps: int = -1, max_new_tokens: int = 128,
) -> None:
    preparation = prepare_restaurant_data(
        dataset_manifest_uri=dataset_manifest_uri, train_limit=train_limit,
        validation_limit=validation_limit, test_limit=test_limit, seed=seed,
    )
    training = fine_tuning_component(
        train_dataset=preparation.outputs["train_dataset"],
        validation_dataset=preparation.outputs["validation_dataset"],
        base_model_id=base_model_id, learning_rate=learning_rate, num_train_epochs=num_train_epochs,
        lora_rank=lora_rank, lora_alpha=lora_alpha, max_seq_length=max_seq_length,
        batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed, max_steps=max_steps,
    )
    training.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
    training.set_cpu_limit("16").set_memory_limit("50G")
    with dsl.If(evaluation_split == "validation", name="validation-evaluation"):
        validation_predictions = inference_component(
            model=training.outputs["model"], test_dataset=preparation.outputs["validation_dataset"],
            max_new_tokens=max_new_tokens, compare_base_model=True,
        )
        validation_predictions.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
        validation_predictions.set_cpu_limit("16").set_memory_limit("50G")
        evaluate_restaurant_predictions(predictions=validation_predictions.outputs["predictions"])
    with dsl.Elif(evaluation_split == "test", name="test-evaluation"):
        test_predictions = inference_component(
            model=training.outputs["model"], test_dataset=preparation.outputs["test_dataset"],
            max_new_tokens=max_new_tokens, compare_base_model=True,
        )
        test_predictions.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
        test_predictions.set_cpu_limit("16").set_memory_limit("50G")
        evaluate_restaurant_predictions(predictions=test_predictions.outputs["predictions"])


@dsl.pipeline(name="lulik-restaurant-heldout-evaluation", description="Evaluate frozen adapters on the official test split without retraining.")
def restaurant_test_pipeline(
    dataset_manifest_uri: str, model_uri: str, test_limit: int = 0, max_new_tokens: int = 128,
) -> None:
    preparation = prepare_restaurant_data(dataset_manifest_uri=dataset_manifest_uri,
        train_limit=1, validation_limit=1, test_limit=test_limit)
    imported = dsl.importer(artifact_uri=model_uri, artifact_class=dsl.Model, reimport=False)
    predictions = inference_component(model=imported.output, test_dataset=preparation.outputs["test_dataset"],
        max_new_tokens=max_new_tokens, compare_base_model=True)
    predictions.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit(1)
    predictions.set_cpu_limit("16").set_memory_limit("50G")
    evaluate_restaurant_predictions(predictions=predictions.outputs["predictions"])
