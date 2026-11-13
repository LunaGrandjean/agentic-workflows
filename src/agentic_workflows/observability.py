"""Optional Langfuse tracing for chat predictions."""

from contextlib import nullcontext
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Langfuse Python SDK v4 renamed LANGFUSE_HOST to LANGFUSE_BASE_URL.
if not os.getenv("LANGFUSE_BASE_URL") and os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_BASE_URL"] = os.environ["LANGFUSE_HOST"]

_client = None
if all(os.getenv(key) for key in ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_BASE_URL")):
    from langfuse import Langfuse

    _client = Langfuse(
        should_export_span=lambda span: span.instrumentation_scope.name != "chainlit"
    )


def start_chat_trace(user_input: str, *, task: str) -> Any:
    """Start a chat-turn span when Langfuse is configured; otherwise do nothing."""
    if _client is None:
        return nullcontext(None)
    return _client.start_as_current_observation(
        name=f"{task}-chat-turn",
        as_type="span",
        input=user_input,
    )


def start_model_generation(
    *,
    prompt: str,
    parameters: dict[str, Any],
    model_name: str,
    task: str,
    model_resource: str,
    endpoint_id: str,
    region: str,
) -> Any:
    """Start a nested Vertex generation observation when tracing is enabled."""
    if _client is None:
        return nullcontext(None)
    version = model_resource.rsplit("@", 1)[1] if "@" in model_resource else None
    return _client.start_as_current_observation(
        name=f"{task}-vertex-prediction",
        as_type="generation",
        input={"prompt": prompt, "parameters": parameters},
        model=model_name,
        version=version,
        model_parameters=parameters,
        metadata={
            "model_resource": model_resource or None,
            "endpoint_id": endpoint_id,
            "region": region,
        },
    )
