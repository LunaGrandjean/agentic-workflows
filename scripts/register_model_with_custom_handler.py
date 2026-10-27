"""Upload the standalone handler and register a Session 3 adapter as a Vertex model/version."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentic_workflows.constants import GCP_PROJECT_ID, GCP_REGION, SERVING_CONTAINER_IMAGE_URI


def resolve_source(model_uri: str | None, run_summary: Path) -> tuple[str, str]:
    source_run = ""
    if run_summary.is_file():
        summary = json.loads(run_summary.read_text(encoding="utf-8"))
        if not model_uri:
            if summary.get("state") != "PIPELINE_STATE_SUCCEEDED":
                raise ValueError("The source pipeline must have succeeded.")
            model_uri = summary["artifacts"]["fine-tuning-component/model"]["uri"]
        if model_uri == summary.get("artifacts", {}).get("fine-tuning-component/model", {}).get("uri"):
            source_run = summary.get("job", "")
    if not model_uri or not model_uri.startswith("gs://") or not model_uri[5:].partition("/")[2]:
        raise ValueError("Provide --model-uri gs://bucket/model-directory or a downloaded Session 3 run summary.")
    return model_uri.rstrip("/"), source_run


def register_model(
    model_uri: str, model_name: str, *, source_run: str = "", parent_model: str | None = None,
    version_alias: str = "session4", image_uri: str = SERVING_CONTAINER_IMAGE_URI,
) -> dict:
    from google.cloud import aiplatform, storage

    client = storage.Client(project=GCP_PROJECT_ID)
    bucket_name, _, prefix = model_uri[5:].partition("/")
    bucket = client.bucket(bucket_name)
    if not bucket.blob(f"{prefix}/adapter_config.json").exists(timeout=30):
        raise FileNotFoundError(f"No LoRA adapter_config.json at {model_uri}")
    files = {
        "handler.py": PROJECT_ROOT / "src/agentic_workflows/handler.py",
        "requirements.txt": PROJECT_ROOT / "serving/requirements.txt",
    }
    checksums = {}
    for name, path in files.items():
        checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        bucket.blob(f"{prefix}/{name}").upload_from_filename(str(path), timeout=60)
        print(f"Uploaded {name} to {model_uri}", flush=True)
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    description = f"Session 4 Yoda Phi-3 LoRA serving. Source artifacts: {model_uri}. Source pipeline: {source_run or 'specified by artifact URI'}."
    model = aiplatform.Model.upload(
        display_name=model_name, artifact_uri=model_uri, parent_model=parent_model,
        description=description, version_description=description,
        version_aliases=[version_alias], serving_container_image_uri=image_uri,
        serving_container_ports=[8080],
        serving_container_environment_variables={"HF_TASK": "text-generation", "HF_TRUST_REMOTE_CODE": "0"},
        serving_container_deployment_timeout=3600,
        labels={"course-session": "4", "team": "lulik"},
    )
    result = {
        "model_resource": model.resource_name, "version_id": model.version_id,
        "versioned_model_resource": f"{model.resource_name}@{model.version_id}",
        "display_name": model_name, "source_artifact_uri": model_uri,
        "source_pipeline": source_run, "serving_image": image_uri, "file_sha256": checksums,
    }
    if source_run:
        from google.api_core.exceptions import GoogleAPICallError

        try:
            result["lineage"] = link_model_lineage(model, source_run, model_uri)
        except GoogleAPICallError as exc:
            result["lineage"] = {"status": "not_linked", "error": str(exc)}
            print(f"Model registered, but native metadata lineage could not be linked: {exc}", flush=True)
    return result


def link_model_lineage(model, source_run: str, model_uri: str) -> dict:
    """Link the pipeline's model artifact to the registry model in Vertex ML Metadata."""
    from google.cloud import aiplatform
    from google.cloud.aiplatform.metadata.artifact import Artifact
    from google.cloud.aiplatform.metadata.execution import Execution
    from google.cloud.aiplatform_v1.types import Execution as ExecutionProto

    pipeline = aiplatform.PipelineJob.get(source_run)
    task = next(task for task in pipeline.gca_resource.job_detail.task_details if task.task_name == "fine-tuning-component")
    runtime_artifact = next(artifact for artifact in task.outputs["model"].artifacts if artifact.uri.rstrip("/") == model_uri)
    source = Artifact(runtime_artifact.name)
    execution = Execution.create(
        schema_title="system.ContainerExecution", schema_version="0.0.1",
        display_name=f"Register Yoda model version {model.version_id}",
        metadata={"source_pipeline": source_run, "registered_model": f"{model.resource_name}@{model.version_id}"},
        credentials=aiplatform.initializer.global_config.credentials,
    )
    execution.assign_input_artifacts([source])
    execution.assign_output_artifacts([model])
    execution.update(state=ExecutionProto.State.COMPLETE)
    return {"status": "linked", "execution": execution.resource_name, "source_artifact": source.resource_name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri")
    parser.add_argument("--model-name", default="lulik-yoda-phi3")
    parser.add_argument("--run-summary", type=Path, default=PROJECT_ROOT / "data/session3/run_summary.json")
    parser.add_argument("--parent-model", help="Register another version of this model resource.")
    parser.add_argument("--version-alias", default="session4")
    parser.add_argument("--image-uri", default=SERVING_CONTAINER_IMAGE_URI)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--link-existing", help="Link an existing registered model version without uploading another version.")
    args = parser.parse_args()
    try:
        model_uri, source_run = resolve_source(args.model_uri, args.run_summary)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({"model_uri": model_uri, "source_run": source_run, "model_name": args.model_name,
                          "parent_model": args.parent_model, "image_uri": args.image_uri}, indent=2))
        return
    output = PROJECT_ROOT / "data/session4/registration.json"
    if args.link_existing:
        from google.cloud import aiplatform

        if not source_run:
            parser.error("Native lineage requires the matching source pipeline in --run-summary.")
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        model = aiplatform.Model(args.link_existing)
        lineage = link_model_lineage(model, source_run, model_uri)
        manifest = json.loads(output.read_text(encoding="utf-8"))
        for registration in manifest["registrations"]:
            if registration["versioned_model_resource"] == args.link_existing:
                registration["lineage"] = lineage
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(lineage, indent=2))
        return
    result = register_model(
        model_uri, args.model_name, source_run=source_run, parent_model=args.parent_model,
        version_alias=args.version_alias, image_uri=args.image_uri,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"registrations": []}
    manifest["registrations"].append(result)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    from dotenv import set_key

    set_key(str(PROJECT_ROOT / ".env"), "GCP_MODEL_RESOURCE", result["versioned_model_resource"])
    set_key(str(PROJECT_ROOT / ".env"), "GCP_PROJECT_NUMBER", result["model_resource"].split("/")[1])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
