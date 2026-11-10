"""Validate immutable raw inputs and prepare the three official E2E splits."""

from kfp.dsl import Artifact, Dataset, Output, component

from agentic_workflows.restaurant_data import (
    canonical_restaurant_mr, parse_restaurant_mr, prepare_restaurant_frames, restaurant_prompt,
)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.3.3", "pyarrow>=19,<26", "gcsfs>=2025.9.0,<2026.8.0"],
    additional_funcs=[parse_restaurant_mr, canonical_restaurant_mr, restaurant_prompt, prepare_restaurant_frames],
)
def prepare_restaurant_data(
    dataset_manifest_uri: str,
    train_dataset: Output[Dataset], validation_dataset: Output[Dataset], test_dataset: Output[Dataset],
    dataset_audit: Output[Artifact],
    train_limit: int = 0, validation_limit: int = 0, test_limit: int = 0, seed: int = 42,
) -> None:
    import hashlib
    import io
    import json
    from pathlib import Path

    import fsspec
    import pandas as pd

    with fsspec.open(dataset_manifest_uri, "rt", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest["dataset"] != "GEM/e2e_nlg":
        raise ValueError("This pipeline expects the frozen GEM/e2e_nlg manifest.")
    prefix = dataset_manifest_uri.replace("\\", "/").rsplit("/", 1)[0]
    raw = {}
    for split in ("train", "validation", "test"):
        with fsspec.open(f"{prefix}/{split}.parquet", "rb") as stream:
            content = stream.read()
        if hashlib.sha256(content).hexdigest() != manifest["files"][split]["sha256"]:
            raise ValueError(f"Raw dataset checksum mismatch: {split}.")
        raw[split] = pd.read_parquet(io.BytesIO(content))
    frames, audit = prepare_restaurant_frames(raw,
        {"train": train_limit, "validation": validation_limit, "test": test_limit}, seed)
    for split, output in (("train", train_dataset), ("validation", validation_dataset), ("test", test_dataset)):
        Path(output.path).parent.mkdir(parents=True, exist_ok=True)
        frames[split].to_csv(output.path, index=False)
        output.metadata.update(audit["splits"][split], source_revision=manifest["source_revision"],
                               parquet_revision=manifest["parquet_revision"], prompt_version=audit["prompt_version"])
    audit["source_manifest"] = manifest
    Path(dataset_audit.path).parent.mkdir(parents=True, exist_ok=True)
    Path(dataset_audit.path).write_text(json.dumps(audit, indent=2), encoding="utf-8")
