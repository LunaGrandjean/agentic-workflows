import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kfp import compiler

from agentic_workflows.constants import (
    GCP_PROJECT_ID,
    GCP_REGION,
    PIPELINE_ROOT,
    RAW_DATASET_URI,
)
from agentic_workflows.pipelines.model_training_pipeline import (
    yoda_data_preparation_pipeline,
    yoda_model_training_pipeline,
)


PACKAGE_PATH = PROJECT_ROOT / "compiled_pipelines/lulik_yoda_model_training.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the pipeline JSON without submitting it to Vertex AI.",
    )
    parser.add_argument("--data-only", action="store_true", help="Run the original Session 2 pipeline.")
    parser.add_argument("--smoke-test", action="store_true", help="Train for two steps and predict three test rows.")
    parser.add_argument("--region", default=GCP_REGION, help="Region with available T4 GPU quota.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--service-account", help="Service account used by the Vertex pipeline.")
    parser.add_argument("--wait", action="store_true", help="Wait for completion after submission.")
    args = parser.parse_args()

    if args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("--epochs and --learning-rate must be positive.")
    pipeline = yoda_data_preparation_pipeline if args.data_only else yoda_model_training_pipeline
    package_path = PACKAGE_PATH.with_name("lulik_yoda_data_preparation.json") if args.data_only else PACKAGE_PATH

    package_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Compiling pipeline to {package_path}", flush=True)
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=str(package_path),
    )
    print("Pipeline compiled.", flush=True)

    if args.compile_only:
        return

    from google.cloud import aiplatform

    parameters = {"raw_dataset_uri": RAW_DATASET_URI, "test_size": 0.2, "seed": 42}
    if not args.data_only:
        parameters.update(
            num_train_epochs=args.epochs, learning_rate=args.learning_rate,
            max_steps=2 if args.smoke_test else -1,
            max_samples=3 if args.smoke_test else 0,
        )

    print(
        f"Initializing Vertex AI project={GCP_PROJECT_ID} region={args.region}",
        flush=True,
    )
    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=args.region,
        staging_bucket=PIPELINE_ROOT,
    )

    print(f"Submitting pipeline with root {PIPELINE_ROOT}", flush=True)
    job = aiplatform.PipelineJob(
        display_name="lulik-yoda-data-preparation" if args.data_only else "lulik-yoda-model-training",
        template_path=str(package_path),
        pipeline_root=PIPELINE_ROOT,
        parameter_values=parameters,
        enable_caching=False,
    )
    job.submit(service_account=args.service_account)
    print(f"Submitted pipeline job: {job.resource_name}")
    print(f"Pipeline root: {PIPELINE_ROOT}")
    if args.wait:
        job.wait()


if __name__ == "__main__":
    main()
