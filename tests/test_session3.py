import json

import pandas as pd
import pytest
from kfp import compiler
from kfp.dsl import Dataset, Metrics

from agentic_workflows.data_processing import format_yoda_dataset
from agentic_workflows.evaluation import aggregate_metrics, compute_metrics
from agentic_workflows.inference import build_prompt, download_model, extract_response
from agentic_workflows.pipeline_components.data_transformation_component import transform_yoda_data
from agentic_workflows.pipeline_components.evaluation_component import evaluation_component
from agentic_workflows.pipelines.model_training_pipeline import yoda_model_training_pipeline


@pytest.mark.parametrize("pipeline_component", [False, True])
def test_csv_conversations_round_trip(tmp_path, pipeline_component):
    source = tmp_path / "raw.csv"
    # Quotes, apostrophes, Unicode and embedded newlines must survive the CSV/JSON round trip.
    raw = pd.DataFrame({
        "sentence": [f'She said "it\'s café time"\n{i}' for i in range(10)],
        "translation": [f'"Café time", it is.\n{i}' for i in range(10)],
    })
    raw.to_csv(source, index=False)
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    if pipeline_component:
        transform_yoda_data.python_func(str(source), 0.2, 42, str(train), str(test))
    else:
        format_yoda_dataset(str(source), str(train), str(test))
    frames = [pd.read_csv(path) for path in (train, test)]
    assert [len(frame) for frame in frames] == [8, 2]
    conversations = [json.loads(value) for frame in frames for value in frame["messages"]]
    assert {row[0]["content"] for row in conversations} == set(raw["sentence"])
    assert {row[1]["content"] for row in conversations} == set(raw["translation"])


def test_prompt_and_response_extraction():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert not tokenize and add_generation_prompt
            assert messages == [{"role": "user", "content": "The Force is strong in you!"}]
            return "<|user|>\nThe Force is strong in you!<|end|>\n<|assistant|>\n"

    prompt = build_prompt(Tokenizer(), "The Force is strong in you!")
    answer = "Strong in you, the Force is. Yes, hrrmmm."
    assert extract_response(prompt + answer + "<|end|>\n<|endoftext|>") == answer
    assert extract_response(answer) == answer


def test_evaluation_component_and_local_helpers(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    from ragas.metrics import RougeScore

    definitions = [RougeScore(rouge_type=kind, name=kind) for kind in ("rouge1", "rouge2", "rougeL")]
    exact = compute_metrics("Hello", "Strong in you, the Force is.", "Strong in you, the Force is.", definitions)
    empty = compute_metrics("Hello", "", "Strong in you, the Force is.", definitions)
    assert all(value == pytest.approx(1.0) for value in exact.values())
    assert all(value == 0 for value in empty.values())
    expected = aggregate_metrics(pd.DataFrame([exact, empty]))
    assert all(value == pytest.approx(0.5) for value in expected.values())
    predictions = Dataset(uri=(tmp_path / "predictions.csv").as_uri())
    results = Dataset(uri=(tmp_path / "results.csv").as_uri())
    # Local artifacts normally expose URI-derived paths; explicitly set paths for the local run.
    predictions.path = str(tmp_path / "predictions.csv")
    results.path = str(tmp_path / "results.csv")
    pd.DataFrame([
        dict(user_input="Hello", reference="Strong in you, the Force is.", extracted_response="Strong in you, the Force is."),
        dict(user_input="Hello", reference="Strong in you, the Force is.", extracted_response=""),
    ]).to_csv(predictions.path, index=False)
    metrics = Metrics()
    evaluation_component.python_func(predictions, results, metrics)
    assert len(pd.read_csv(results.path)) == 2
    for name, value in expected.items():
        assert metrics.metadata[name] == pytest.approx(value)


def test_pipeline_artifact_connections_and_gpu_requests(tmp_path):
    package = tmp_path / "pipeline.json"
    compiler.Compiler().compile(yoda_model_training_pipeline, str(package))
    spec = json.loads(package.read_text(encoding="utf-8"))
    tasks = spec["root"]["dag"]["tasks"]
    assert len(tasks) == 4
    assert tasks["fine-tuning-component"]["inputs"]["artifacts"]["train_dataset"]["taskOutputArtifact"]["producerTask"] == "transform-yoda-data"
    assert tasks["inference-component"]["inputs"]["artifacts"]["model"]["taskOutputArtifact"]["producerTask"] == "fine-tuning-component"
    assert tasks["evaluation-component"]["inputs"]["artifacts"]["predictions"]["taskOutputArtifact"]["producerTask"] == "inference-component"
    executors = spec["deploymentSpec"]["executors"]
    for name in ("exec-fine-tuning-component", "exec-inference-component"):
        accelerator = executors[name]["container"]["resources"]["accelerator"]
        assert accelerator["resourceType"] == "NVIDIA_TESLA_T4"
        assert accelerator["resourceCount"] == "1"


def test_empty_metrics_rejected():
    with pytest.raises(ValueError, match="empty"):
        aggregate_metrics(pd.DataFrame())


def test_download_preserves_subdirectories_and_rejects_traversal(tmp_path, monkeypatch):
    from pathlib import Path
    from google.cloud import storage

    class Blob:
        def __init__(self, name):
            self.name = name

        def download_to_filename(self, filename):
            Path(filename).write_text("model file", encoding="utf-8")

    objects = [Blob("run/model/adapter_config.json"), Blob("run/model/subdir/tokenizer.json")]

    class Client:
        def __init__(self, *, project):
            assert project

        def list_blobs(self, bucket, *, prefix):
            assert bucket == "test-bucket" and prefix == "run/model/"
            return objects

    monkeypatch.setattr(storage, "Client", Client)
    download_model("gs://test-bucket/run/model", str(tmp_path / "model"))
    assert (tmp_path / "model/subdir/tokenizer.json").read_text() == "model file"
    objects[:] = [Blob("run/model/../../escape.json")]
    with pytest.raises(ValueError, match="Unsafe"):
        download_model("gs://test-bucket/run/model", str(tmp_path / "model"))
