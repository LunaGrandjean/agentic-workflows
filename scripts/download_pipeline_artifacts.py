"""Download Session 3 outputs, adapters, TensorBoard events and metric metadata."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentic_workflows.constants import GCP_PROJECT_ID, GCP_REGION
from agentic_workflows.inference import download_model


def main() -> None:
    from google.cloud import aiplatform, storage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", help="Full Vertex pipeline resource name.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/session3")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models/phi3-yoda-adapter")
    parser.add_argument("--logs-dir", type=Path, default=PROJECT_ROOT / "logs/session3")
    parser.add_argument("--metadata-only", action="store_true", help="Save run_summary.json without downloading artifact files.")
    args = parser.parse_args()
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    job = aiplatform.PipelineJob.get(args.job)
    if job.state.name != "PIPELINE_STATE_SUCCEEDED":
        raise RuntimeError(f"Wait for successful completion before downloading: {job.state.name}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=GCP_PROJECT_ID)
    filenames = {
        ("transform-yoda-data", "train_dataset"): "train.csv",
        ("transform-yoda-data", "test_dataset"): "test.csv",
        ("inference-component", "predictions"): "predictions.csv",
        ("evaluation-component", "evaluation_results"): "evaluation_results.csv",
    }
    summary = {"job": args.job, "state": job.state.name, "artifacts": {}}
    for task in job.gca_resource.job_detail.task_details:
        for name, output in task.outputs.items():
            for artifact in output.artifacts:
                key = f"{task.task_name}/{name}"
                summary["artifacts"][key] = {
                    "uri": artifact.uri, "metadata": type(artifact).to_dict(artifact).get("metadata", {}),
                }
                if args.metadata_only:
                    continue
                filename = filenames.get((task.task_name, name))
                if filename:
                    bucket, object_name = artifact.uri[5:].split("/", 1)
                    client.bucket(bucket).blob(object_name).download_to_filename(str(args.output_dir / filename))
                    print(f"Downloaded {filename}", flush=True)
                elif task.task_name == "fine-tuning-component" and name == "model":
                    download_model(artifact.uri, str(args.model_dir))
                    print(f"Downloaded adapters to {args.model_dir}", flush=True)
                elif task.task_name == "fine-tuning-component" and name == "metrics":
                    download_model(artifact.uri, str(args.logs_dir))
                    print(f"Downloaded TensorBoard logs to {args.logs_dir}", flush=True)
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved run_summary.json to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
