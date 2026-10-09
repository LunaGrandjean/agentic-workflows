import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from google.cloud import aiplatform
from kfp import compiler

from agentic_workflows.constants import (
    GCP_PROJECT_ID,
    GCP_REGION,
    PIPELINE_ROOT,
    RAW_DATASET_URI,
)
from agentic_workflows.pipelines.model_training_pipeline import (
    yoda_data_preparation_pipeline,
)


PACKAGE_PATH = PROJECT_ROOT / "compiled_pipelines/lulik_yoda_data_preparation.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the pipeline JSON without submitting it to Vertex AI.",
    )
    args = parser.parse_args()

    PACKAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Compiling pipeline to {PACKAGE_PATH}", flush=True)
    compiler.Compiler().compile(
        pipeline_func=yoda_data_preparation_pipeline,
        package_path=str(PACKAGE_PATH),
    )
    print("Pipeline compiled.", flush=True)

    if args.compile_only:
        return

    print(
        f"Initializing Vertex AI project={GCP_PROJECT_ID} region={GCP_REGION}",
        flush=True,
    )
    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=GCP_REGION,
        staging_bucket=PIPELINE_ROOT,
    )

    print(f"Submitting pipeline with root {PIPELINE_ROOT}", flush=True)
    job = aiplatform.PipelineJob(
        display_name="lulik-yoda-data-preparation",
        template_path=str(PACKAGE_PATH),
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "raw_dataset_uri": RAW_DATASET_URI,
            "test_size": 0.2,
            "seed": 42,
        },
        enable_caching=False,
    )
    job.submit()
    print(f"Submitted pipeline job: {job.resource_name}")
    print(f"Pipeline root: {PIPELINE_ROOT}")


if __name__ == "__main__":
    main()
