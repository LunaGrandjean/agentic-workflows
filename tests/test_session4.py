from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from agentic_workflows.handler import EndpointHandler
from agentic_workflows.online_prediction import PredictionError, endpoint_url, post_prediction, prediction_text


def test_handler_loads_base_and_attaches_saved_adapter(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text("{}")
    calls = []
    model = SimpleNamespace(config=SimpleNamespace(use_cache=False), eval=lambda: calls.append("eval"))
    tokenizer = SimpleNamespace(pad_token_id=0)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), float16="fp16", float32="fp32",
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda name, **kwargs: calls.append((name, kwargs)) or model),
    ))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftConfig=SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(base_model_name_or_path="base-phi3")),
        PeftModel=SimpleNamespace(from_pretrained=lambda base, path: calls.append(("adapter", path)) or base),
    ))
    handler = EndpointHandler(str(tmp_path))
    assert calls[0][0] == "base-phi3"
    assert calls[0][1]["device_map"] == {"": "cpu"}
    assert calls[0][1]["torch_dtype"] == "fp32"
    assert calls[1] == ("adapter", str(tmp_path)) and calls[2] == "eval"
    assert handler.model.config.use_cache


def test_handler_returns_only_new_tokens_for_each_instance():
    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, prompt, **kwargs):
            assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
            return Inputs(input_ids=np.array([[1, 2, int(prompt)]]))

        def decode(self, output, *, skip_special_tokens):
            assert skip_special_tokens and output.tolist() == [90, 91]
            return " Strong in you, the Force is. "

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 32 and kwargs["do_sample"] is False
            return np.concatenate([kwargs["input_ids"], np.array([[90, 91]])], axis=1)

    handler = EndpointHandler.__new__(EndpointHandler)
    handler.tokenizer, handler.model = Tokenizer(), Model()
    handler.torch = SimpleNamespace(inference_mode=nullcontext)
    payload = {"instances": [{"prompt": "3"}, {"prompt": "4"}], "parameters": {"max_new_tokens": 32}}
    original = copy.deepcopy(payload)
    assert handler(payload) == {"predictions": [{"generated_text": "Strong in you, the Force is."}] * 2}
    assert payload == original


@pytest.mark.parametrize("payload", [
    {}, {"instances": []}, {"instances": [{"prompt": ""}]},
    {"instances": [{"prompt": "test"}], "parameters": {"max_new_tokens": 10000}},
    {"instances": [{"prompt": "test"}], "parameters": {"max_new_tokens": True}},
    {"instances": [{"prompt": "test"}], "parameters": {"device_map": "cpu"}},
    {"instances": [{"prompt": "test"}], "parameters": {"do_sample": "false"}},
])
def test_invalid_vertex_requests_are_rejected(payload):
    with pytest.raises(ValueError):
        EndpointHandler.validate_request(payload)


def test_authenticated_http_client_passes_vertex_json_and_parses_response():
    class Session:
        def post(self, url, *, json, timeout):
            assert url == endpoint_url("123", "europe-west2", "456")
            assert json["instances"][0]["prompt"] == "<|user|>Hello<|end|><|assistant|>"
            assert timeout == (10, 120)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                "predictions": [{"generated_text": "Hello, there is.<|end|>"}],
            })

    result = post_prediction(endpoint_url("123", "europe-west2", "456"), {
        "instances": [{"prompt": "<|user|>Hello<|end|><|assistant|>"}],
    }, session=Session())
    assert prediction_text(result) == "Hello, there is."


def test_http_errors_provide_actionable_messages_without_returning_server_details():
    class Session:
        def post(self, *args, **kwargs):
            response = requests.Response()
            response.status_code = 403
            response._content = b"private server diagnostic"
            return response

    with pytest.raises(PredictionError, match="active account") as error:
        post_prediction("https://example.invalid", {}, session=Session())
    assert "private server" not in str(error.value)


def test_missing_endpoint_and_empty_prediction_fail_clearly():
    with pytest.raises(PredictionError, match="offline"):
        endpoint_url("123", "europe-west2", "")
    with pytest.raises(PredictionError, match="no predictions"):
        prediction_text({"predictions": []})


def test_registration_source_preserves_pipeline_provenance(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/register_model_with_custom_handler.py"
    spec = importlib.util.spec_from_file_location("session4_registration", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "state": "PIPELINE_STATE_SUCCEEDED", "job": "projects/123/locations/europe-west2/pipelineJobs/training",
        "artifacts": {"fine-tuning-component/model": {"uri": "gs://bucket/run/model"}},
    }))
    assert module.resolve_source(None, summary) == (
        "gs://bucket/run/model", "projects/123/locations/europe-west2/pipelineJobs/training",
    )
    assert module.resolve_source("gs://bucket/another/model", summary) == ("gs://bucket/another/model", "")
    with pytest.raises(ValueError, match="model-uri"):
        module.resolve_source("https://example.com/model", summary)


def test_lineage_uses_real_credentials_and_connects_input_and_output(monkeypatch):
    from google.cloud import aiplatform
    from google.cloud.aiplatform.metadata.artifact import Artifact
    from google.cloud.aiplatform.metadata.execution import Execution

    script = Path(__file__).resolve().parents[1] / "scripts/register_model_with_custom_handler.py"
    spec = importlib.util.spec_from_file_location("session4_lineage", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    credentials = object()
    monkeypatch.setattr(aiplatform.initializer.global_config, "_credentials", credentials)
    artifact = SimpleNamespace(uri="gs://bucket/run/model", name="source-artifact")
    task = SimpleNamespace(task_name="fine-tuning-component", outputs={"model": SimpleNamespace(artifacts=[artifact])})
    pipeline = SimpleNamespace(gca_resource=SimpleNamespace(job_detail=SimpleNamespace(task_details=[task])))
    monkeypatch.setattr(aiplatform.PipelineJob, "get", lambda resource: pipeline)
    monkeypatch.setattr(Artifact, "__init__", lambda self, name: None)
    monkeypatch.setattr(Artifact, "resource_name", property(lambda self: "source-artifact"))
    events = []
    execution = SimpleNamespace(
        resource_name="registration-execution",
        assign_input_artifacts=lambda artifacts: events.append(("input", artifacts[0].resource_name)),
        assign_output_artifacts=lambda artifacts: events.append(("output", artifacts[0].resource_name)),
        update=lambda **kwargs: events.append(("state", kwargs["state"].name)),
    )

    def create(**kwargs):
        assert kwargs["credentials"] is credentials
        return execution

    monkeypatch.setattr(Execution, "create", create)
    model = SimpleNamespace(resource_name="registry-model", version_id="2")
    result = module.link_model_lineage(model, "source-pipeline", "gs://bucket/run/model")
    assert result["status"] == "linked"
    assert events == [("input", "source-artifact"), ("output", "registry-model"), ("state", "COMPLETE")]
