"""Create, inspect or remove the endpoint for the selected workflow task."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentic_workflows.constants import (
    ENDPOINT_ENV_KEY, MODEL_ENV_KEY, GCP_MODEL_RESOURCE, GCP_PROJECT_ID, GCP_REGION,
    TASK_ENDPOINT_NAME, TASK_LABELS, TASK_MODEL_NAME, TASK_OUTPUT_DIR, WORKFLOW_TASK,
)

MANIFEST_PATH = TASK_OUTPUT_DIR / "endpoint.json"


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    from dotenv import set_key
    from google.cloud import aiplatform

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Read models and endpoints for the selected task.")
    deploy = commands.add_parser("deploy", help="Create a billable one-T4 endpoint for the registered model.")
    deploy.add_argument("--model", default=GCP_MODEL_RESOURCE)
    deploy.add_argument("--name", default=TASK_ENDPOINT_NAME)
    deploy.add_argument("--machine-type", default="n1-standard-8")
    commands.add_parser("status", help="Read the endpoint recorded in the local manifest.")
    commands.add_parser("cleanup", help="Undeploy and delete only the endpoint created by this script.")
    args = parser.parse_args()
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    if args.command == "list":
        print("Models:", [(m.resource_name, m.display_name) for m in aiplatform.Model.list() if m.display_name == TASK_MODEL_NAME])
        print("Endpoints:", [(e.resource_name, e.display_name) for e in aiplatform.Endpoint.list() if e.display_name == TASK_ENDPOINT_NAME])
        return
    if args.command == "deploy":
        if not args.model:
            parser.error("Register a model first or pass --model projects/.../models/...@VERSION.")
        if MANIFEST_PATH.exists():
            previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if previous.get("state") != "deleted":
                parser.error("An endpoint manifest already exists. Inspect or clean it up before creating another endpoint.")
        endpoint = aiplatform.Endpoint.create(
            display_name=args.name, description=f"Temporary {WORKFLOW_TASK} model endpoint; remove after use.",
            labels=TASK_LABELS,
        )
        manifest = {
            "endpoint_resource": endpoint.resource_name, "endpoint_id": endpoint.name,
            "project_number": endpoint.resource_name.split("/")[1], "region": GCP_REGION,
            "model_resource": args.model, "state": "deploying", "machine_type": args.machine_type,
            "accelerator_type": "NVIDIA_TESLA_T4", "accelerator_count": 1,
            "min_replicas": 1, "max_replicas": 1,
            "workflow_task": WORKFLOW_TASK, "ownership_labels": TASK_LABELS,
        }
        save_manifest(manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        try:
            aiplatform.Model(args.model).deploy(
                endpoint=endpoint, deployed_model_display_name=TASK_MODEL_NAME,
                machine_type=args.machine_type, accelerator_type="NVIDIA_TESLA_T4", accelerator_count=1,
                min_replica_count=1, max_replica_count=1, traffic_percentage=100,
                deploy_request_timeout=3600,
            )
        except Exception as exc:
            manifest.update(state="deployment_failed", error=str(exc))
            save_manifest(manifest)
            raise
        manifest["state"] = "deployed"
        save_manifest(manifest)
        for name, value in {
            "GCP_PROJECT_NUMBER": manifest["project_number"], ENDPOINT_ENV_KEY: str(manifest["endpoint_id"]),
            MODEL_ENV_KEY: args.model,
        }.items():
            set_key(str(PROJECT_ROOT / ".env"), name, value)
        print("Deployment ready. .env updated.", flush=True)
        return
    if not MANIFEST_PATH.exists():
        parser.error("No endpoint manifest found. Deploy first.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("state") == "deleted":
        print(json.dumps(manifest, indent=2))
        return
    endpoint = aiplatform.Endpoint(manifest["endpoint_resource"])
    models = endpoint.list_models()
    if args.command == "status":
        print(json.dumps({**manifest, "deployed_models": [type(m).to_dict(m) for m in models]}, indent=2))
        return
    if any(endpoint.labels.get(key) != value for key, value in manifest.get("ownership_labels", TASK_LABELS).items()):
        raise RuntimeError("Refusing cleanup: the endpoint does not have this practice's ownership labels.")
    for deployed in models:
        # The SDK removes the last model's traffic automatically. Passing an
        # explicit empty mapping fails its traffic-total validation.
        endpoint.undeploy(deployed.id)
    endpoint.delete()
    manifest["state"] = "deleted"
    save_manifest(manifest)
    set_key(str(PROJECT_ROOT / ".env"), ENDPOINT_ENV_KEY, "")
    print("Model undeployed and endpoint deleted. Registered models are preserved.", flush=True)


if __name__ == "__main__":
    main()
