"""Authenticated HTTP predictions for the Session 4 command-line client and chat app."""

from functools import lru_cache
import subprocess
from typing import Any

from agentic_workflows.inference import build_prompt, extract_response
from agentic_workflows.observability import start_chat_trace, start_model_generation


class PredictionError(RuntimeError):
    """An actionable failure to obtain a prediction from the configured endpoint."""


def endpoint_url(project: str, region: str, endpoint_id: str) -> str:
    if not project:
        raise PredictionError("The prediction service has not been configured yet.")
    if not endpoint_id:
        raise PredictionError("The model is offline. Deploy its Vertex endpoint to generate descriptions.")
    if not endpoint_id.isdecimal():
        raise PredictionError("GCP_ENDPOINT_ID must be the numeric endpoint ID.")
    return f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{endpoint_id}:predict"


def prediction_raw_text(payload: dict[str, Any]) -> str:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise PredictionError("The endpoint returned no predictions.")
    first = predictions[0]
    text = first.get("generated_text") if isinstance(first, dict) else first
    if not isinstance(text, str) or not text.strip():
        raise PredictionError("The endpoint returned an empty or unsupported response.")
    return text


def prediction_text(payload: dict[str, Any]) -> str:
    return extract_response(prediction_raw_text(payload))


def post_prediction(
    url: str, payload: dict[str, Any], *, auth_mode: str = "adc", session: Any = None,
) -> dict[str, Any]:
    import requests

    if session is None:
        if auth_mode == "adc":
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            session = AuthorizedSession(credentials)
        elif auth_mode == "gcloud":
            try:
                token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise PredictionError("gcloud authentication failed. Sign in with gcloud or use AUTH_MODE=adc.") from exc
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {token}"})
        else:
            raise PredictionError("AUTH_MODE must be adc or gcloud.")
    try:
        response = session.post(url, json=payload, timeout=(10, 120))
        response.raise_for_status()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        guidance = {
            401: "Authentication expired; refresh your Google Cloud credentials.",
            403: "The active account cannot predict on this endpoint; check its project access.",
            404: "The configured endpoint no longer exists; deploy the registered model and update GCP_ENDPOINT_ID.",
            400: "The prediction request was rejected; check the endpoint deployment and request format.",
            503: "The model is not ready; wait for deployment to finish.",
        }.get(status, "The prediction service could not be reached; check the endpoint and try again.")
        raise PredictionError(guidance) from exc
    try:
        result = response.json()
    except ValueError as exc:
        raise PredictionError("The endpoint returned an invalid JSON response.") from exc
    if not isinstance(result, dict):
        raise PredictionError("The endpoint returned an invalid prediction response.")
    return result


@lru_cache(maxsize=1)
def load_tokenizer():
    from transformers import AutoTokenizer
    from agentic_workflows.constants import TOKENIZER_DIR

    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


def _traced_prediction(
    *,
    user_input: str,
    prompt: str,
    parameters: dict[str, Any],
    url: str,
    auth_mode: str,
    model_name: str,
    task: str,
    model_resource: str,
    endpoint_id: str,
    region: str,
    answer_guard: Any = None,
) -> str:
    """Call Vertex and capture both the chat turn and raw model generation."""
    with start_chat_trace(user_input, task=task) as trace:
        with start_model_generation(
            prompt=prompt,
            parameters=parameters,
            model_name=model_name,
            task=task,
            model_resource=model_resource,
            endpoint_id=endpoint_id,
            region=region,
        ) as generation:
            prediction = post_prediction(
                url,
                {"instances": [{"prompt": prompt}], "parameters": parameters},
                auth_mode=auth_mode,
            )
            raw_output = prediction_raw_text(prediction)
            answer = extract_response(raw_output)
            if answer_guard is not None:
                answer = answer_guard(answer)
            if generation is not None:
                # Store the response that passed the application guard, not an unsafe raw draft.
                generation.update(output=answer)
        if trace is not None:
            trace.update(output=answer)
        return answer


def translate_sentence(sentence: str, *, parameters: dict[str, Any] | None = None) -> str:
    from agentic_workflows.constants import (
        AUTH_MODE,
        GCP_ENDPOINT_ID,
        GCP_MODEL_RESOURCE,
        GCP_PROJECT_ID,
        GCP_PROJECT_NUMBER,
        GCP_REGION,
        TASK_MODEL_NAME,
        WORKFLOW_TASK,
    )

    if not isinstance(sentence, str) or not sentence.strip():
        raise PredictionError("Enter an English sentence to translate.")
    url = endpoint_url(GCP_PROJECT_NUMBER or GCP_PROJECT_ID, GCP_REGION, GCP_ENDPOINT_ID)
    prompt = build_prompt(load_tokenizer(), sentence.strip())
    model_parameters = parameters or {"max_new_tokens": 64, "do_sample": False}
    return _traced_prediction(
        user_input=sentence,
        prompt=prompt,
        parameters=model_parameters,
        url=url,
        auth_mode=AUTH_MODE,
        model_name=TASK_MODEL_NAME,
        task=WORKFLOW_TASK,
        model_resource=GCP_MODEL_RESOURCE,
        endpoint_id=GCP_ENDPOINT_ID,
        region=GCP_REGION,
    )


def describe_restaurant(attributes: str, *, parameters: dict[str, Any] | None = None) -> str:
    from agentic_workflows.constants import (
        AUTH_MODE,
        GCP_ENDPOINT_ID,
        GCP_MODEL_RESOURCE,
        GCP_PROJECT_ID,
        GCP_PROJECT_NUMBER,
        GCP_REGION,
        TASK_MODEL_NAME,
        WORKFLOW_TASK,
    )
    from agentic_workflows.restaurant_data import (
        parse_restaurant_mr,
        restaurant_prompt,
        restaurant_response_needs_fallback,
        safe_restaurant_fallback,
    )

    try:
        pairs = parse_restaurant_mr(attributes)
        if len(dict(pairs)) != len(pairs):
            raise ValueError("Provide each restaurant attribute once to avoid conflicting instructions.")
        instruction = restaurant_prompt(attributes)
    except ValueError as exc:
        raise PredictionError(str(exc)) from exc
    url = endpoint_url(GCP_PROJECT_NUMBER or GCP_PROJECT_ID, GCP_REGION, GCP_ENDPOINT_ID)
    prompt = build_prompt(load_tokenizer(), instruction)
    model_parameters = parameters or {"max_new_tokens": 128, "do_sample": False}

    def guard_response(answer: str) -> str:
        if restaurant_response_needs_fallback(pairs, answer):
            return safe_restaurant_fallback(pairs)
        return answer

    return _traced_prediction(
        user_input=attributes,
        prompt=prompt,
        parameters=model_parameters,
        url=url,
        auth_mode=AUTH_MODE,
        model_name=TASK_MODEL_NAME,
        task=WORKFLOW_TASK,
        model_resource=GCP_MODEL_RESOURCE,
        endpoint_id=GCP_ENDPOINT_ID,
        region=GCP_REGION,
        answer_guard=guard_response,
    )
