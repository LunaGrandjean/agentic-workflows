"""Local inference helpers for the Session 3 notebook."""

from pathlib import Path
import re
from typing import Any


def download_model(model_uri: str, local_dir: str) -> str:
    """Download every file under a GCS model artifact, preserving subdirectories."""
    from google.cloud import storage
    from agentic_workflows.constants import GCP_PROJECT_ID

    if not model_uri.startswith("gs://"):
        raise ValueError("Expected a gs://bucket/model-directory URI.")
    bucket, _, prefix = model_uri[5:].partition("/")
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    destination = Path(local_dir).resolve()
    count = 0
    for blob in storage.Client(project=GCP_PROJECT_ID).list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        target = (destination / blob.name[len(prefix):]).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe object path: {blob.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        count += 1
    if not count:
        raise FileNotFoundError(f"No model files found under {model_uri}")
    return str(destination)


def build_prompt(tokenizer: Any, sentence: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": sentence}], tokenize=False, add_generation_prompt=True,
    )


def generate_response(model: Any, tokenizer: Any, prompt: str, **kwargs: Any) -> str:
    """Decode only newly generated tokens, excluding the input prompt."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    options = {"max_new_tokens": 64, "do_sample": False, "pad_token_id": tokenizer.pad_token_id}
    options.update(kwargs)
    model.eval()
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extract_response(model_output: str) -> str:
    response = model_output.split("<|assistant|>", 1)[-1]
    return re.split(r"<\|(?:end|endoftext|user|system)\|>", response, maxsplit=1)[0].strip()
