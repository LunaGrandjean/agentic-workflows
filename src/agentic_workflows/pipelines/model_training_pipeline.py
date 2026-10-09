from kfp import dsl

from agentic_workflows.constants import RAW_DATASET_URI
from agentic_workflows.pipeline_components.data_transformation_component import (
    transform_yoda_data,
)


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
