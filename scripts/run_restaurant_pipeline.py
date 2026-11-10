"""Compile or submit the final-project pipeline with explicit experiment limits."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from kfp import compiler
    from agentic_workflows.constants import GCP_BUCKET_NAME, GCP_PROJECT_ID, GCP_REGION
    from agentic_workflows.pipelines.restaurant_pipeline import restaurant_test_pipeline, restaurant_training_pipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--evaluate-model", help="gs:// adapters from a chosen run: evaluate on test without retraining.")
    parser.add_argument("--dataset-manifest-uri")
    parser.add_argument("--train-limit", type=int, default=4000)
    parser.add_argument("--validation-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--region", default=GCP_REGION)
    parser.add_argument("--service-account")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if any(value < 0 for value in (args.train_limit, args.validation_limit, args.test_limit)) or args.epochs <= 0 or args.learning_rate <= 0 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("Limits must be nonnegative; epochs/rate positive; max-steps positive or -1.")
    if args.evaluate_model and (args.smoke_test or not args.evaluate_model.startswith("gs://")):
        parser.error("--evaluate-model requires a gs:// model URI and cannot be a smoke test.")
    path = ROOT / ("compiled_pipelines/restaurant_test.json" if args.evaluate_model else "compiled_pipelines/restaurant_training.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(restaurant_test_pipeline if args.evaluate_model else restaurant_training_pipeline, str(path))
    print(f"Compiled {path}", flush=True)
    if args.compile_only:
        return
    uri_file = ROOT / "data/restaurant/dataset_uri.txt"
    uri = args.dataset_manifest_uri or (uri_file.read_text(encoding="utf-8").strip() if uri_file.exists() else "")
    if not uri.startswith("gs://"):
        parser.error("Run prepare_restaurant_dataset.py --upload or provide --dataset-manifest-uri gs://.../manifest.json.")
    parameters = dict(dataset_manifest_uri=uri, train_limit=args.train_limit,
        validation_limit=args.validation_limit, test_limit=args.test_limit, evaluation_split=args.evaluation_split,
        num_train_epochs=args.epochs, learning_rate=args.learning_rate, max_steps=args.max_steps)
    if args.smoke_test:
        parameters.update(train_limit=32, validation_limit=8, test_limit=8, max_steps=2)
    if args.evaluate_model:
        parameters = dict(dataset_manifest_uri=uri, model_uri=args.evaluate_model, test_limit=args.test_limit)
    from google.cloud import aiplatform
    import google.auth

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"], quota_project_id=GCP_PROJECT_ID,
    )
    aiplatform.init(project=GCP_PROJECT_ID, location=args.region, credentials=credentials,
                     staging_bucket=f"gs://{GCP_BUCKET_NAME}/restaurant/pipelines")
    display_name = "lulik-restaurant-test" if args.evaluate_model else "lulik-restaurant-smoke" if args.smoke_test else "lulik-restaurant-training"
    job = aiplatform.PipelineJob(display_name=display_name,
        template_path=str(path), pipeline_root=f"gs://{GCP_BUCKET_NAME}/restaurant/pipelines",
        parameter_values=parameters, enable_caching=False,
        labels={"project-task": "restaurant", "team": "lulik", "experiment-stage": "smoke" if args.smoke_test else "test" if args.evaluate_model else "pilot"})
    job.submit(service_account=args.service_account)
    record = {"job": job.resource_name, "submitted_at": datetime.now(timezone.utc).isoformat(), "parameters": parameters,
              "smoke_test": args.smoke_test, "state": job.state.name}
    output = ROOT / "data/restaurant/runs"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{job.name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    if args.wait:
        job.wait()


if __name__ == "__main__":
    main()
