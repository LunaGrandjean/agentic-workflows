# Agentic workflows: Yoda fine-tuning

Session 4 adds Model Registry versions, a custom Vertex serving handler and a Chainlit
chat app. Both model versions are registered. Version 2 was deployed on a T4 and
verified through authenticated HTTP requests and the Chainlit browser interface.
The temporary endpoint was then deleted to stop serving charges. See the
[Session 4 results and screenshot](reports/session4_results.md).

Recorded execution checks and cloud runs: [Session 3 results](reports/session3_results.md).

The full Session 3 run succeeded through all four stages. Its 144 held-out predictions
achieved mean ROUGE-1/2/L F1 scores of 0.9741 / 0.8551 / 0.8856. Adapters, predictions,
scores and TensorBoard logs have been downloaded and checked; the report includes examples
and the validation-loss trend.

Session 3 extends the Session 2 Kubeflow pipeline:

```text
GCS CSV -> train/test conversations -> Phi-3 LoRA adapters -> predictions -> Ragas ROUGE scores
```

The training split alone is split again for validation. The held-out test set is used
only for inference and evaluation. Training and inference each request one NVIDIA T4,
16 CPUs and 50 GB RAM. Training saves PEFT adapters and the tokenizer; inference loads
the original Phi-3 base model and attaches those adapters.

Use Python 3.11, `uv`, and the existing `.env` configuration. `transformers==4.46.3`
and `peft==0.13.2` follow the course serving-image requirements. `trl==0.12.2` uses
`max_seq_length` in `SFTConfig` and `processing_class` in `SFTTrainer`.

## Set up and verify locally

```powershell
uv sync --extra evaluation --group dev
uv run --extra evaluation pytest -q
uv run python scripts/pipeline_runner.py --compile-only
```

If Windows blocks an update to the old `.venv`, use the separate Session 3 environment:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-session3"
uv sync --extra evaluation --group dev --link-mode copy
```

The `training` extra adds PyTorch, Transformers, PEFT, TRL, bitsandbytes and TensorBoard
for local GPU experiments. It is optional for compilation and cloud submission because
the pipeline installs these packages inside its GPU containers.

```powershell
uv sync --extra training --extra evaluation --group dev
```

Open `notebooks/session3_practice.ipynb` for configuration inspection, dataset checks,
local GPU training, downloading adapters, three-row inference and evaluation. Local
fine-tuning needs an NVIDIA CUDA GPU; the notebook checks this before loading Phi-3.
The ROUGE evaluation uses no LLM provider or API key.

## Submit to Vertex AI

Application Default Credentials must be valid, the raw CSV must exist at
`RAW_DATASET_URI`, and the selected region must have T4 quota. The pipeline execution
service account needs Vertex AI and read/write access to the GCS bucket.

```powershell
# A short end-to-end run: two training steps, three test predictions.
uv run python scripts/pipeline_runner.py --smoke-test

# Full training (three epochs), predictions on the entire test split, and evaluation.
uv run python scripts/pipeline_runner.py --epochs 3 --learning-rate 0.0003

# Optionally select the region/service account and wait for completion.
uv run python scripts/pipeline_runner.py --region europe-west2 --service-account YOUR_SERVICE_ACCOUNT --wait

# Keep access to the Session 2 pipeline.
uv run python scripts/pipeline_runner.py --data-only --compile-only
```

Submissions use the configured GCP project and bucket and disable caching. Fresh data
preparation is required: conversations are now JSON strings in the `messages` CSV column;
the older NumPy/Python-style CSV representation is not suitable for the trainer.
Regenerate local files with `uv run python scripts/run_data_processing_locally.py`.

With the current account, the Vertex SDK logs a bucket ownership-check error before
successfully creating the job. `pipeline_root` already explicitly selects the existing
course bucket. Check the returned job state with `inspect_pipeline.py`; the Session 3
smoke run completed successfully despite that SDK message.

## Inspect artifacts

Follow the job link printed by Vertex AI. The four tasks produce training/test CSVs,
LoRA adapters, predictions, per-sample scores and aggregated metrics. Predictions have
`user_input,reference,extracted_response` columns; evaluation adds `rouge1`, `rouge2`
and `rougeL` F1 scores. Scores range from zero to one, with one indicating exact overlap.

Print task states, errors and artifact URIs from the command line:

```powershell
uv run python scripts/inspect_pipeline.py projects/PROJECT_NUMBER/locations/europe-west2/pipelineJobs/JOB_ID
uv run python scripts/download_pipeline_artifacts.py projects/PROJECT_NUMBER/locations/europe-west2/pipelineJobs/JOB_ID
```

Training stores TensorBoard events at the training `metrics` artifact URI and logs
training/evaluation losses and trainable parameter counts. The model artifact includes
`hyperparameters.json` and `training_history.json`. Download the TensorBoard events to
`logs/`, then run `uv run --extra training tensorboard --logdir logs/`.

To view the logs without installing the full training extra:

```powershell
uvx --from "tensorboard>=2.18,<3" python -m tensorboard.main --logdir logs/
```

The download script writes CSVs and a `run_summary.json` containing metric metadata to
`data/session3/`, adapters to `models/phi3-yoda-adapter/`, and events to `logs/session3/`.
In the notebook, use `data/session3/test.csv` and `data/session3/predictions.csv` when
working with these downloaded outputs.

References: [course article](https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face),
[pinned TRL trainer](https://github.com/huggingface/trl/blob/v0.12.2/trl/trainer/sft_trainer.py),
[Ragas metrics](https://github.com/explodinggradients/ragas/tree/v0.2.15/src/ragas/metrics).

## Session 4: register, deploy, and chat

The Session 4 flow is:

```text
Session 3 adapters + handler.py -> Model Registry versions -> T4 endpoint <- Chainlit chat
```

Install the app dependencies in the separate Windows environment:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-session4"
uv sync --extra app --extra evaluation --group dev --link-mode copy
.\.venv-session4\Scripts\python.exe -m pytest -q
```

`notebooks/session4_practice.ipynb` walks through the handler, registration, versioning,
deployment, HTTP prediction, chat app and cleanup. Use the `.venv-session4` interpreter.

The standalone handler is `src/agentic_workflows/handler.py`. The registration script
copies it to `handler.py` in the GCS model artifact directory, together with
`serving/requirements.txt`. It records native lineage to the source pipeline and saves
registration resource names and file hashes in `data/session4/registration.json`.

For a new registration, use the downloaded Session 3 run summary or pass `--model-uri`:

```powershell
.\.venv-session4\Scripts\python.exe scripts/register_model_with_custom_handler.py --dry-run
.\.venv-session4\Scripts\python.exe scripts/register_model_with_custom_handler.py --model-name lulik-yoda-phi3
# To register another version, add --parent-model projects/PROJECT/locations/REGION/models/MODEL_ID
```

Versions 1 and 2 are already registered, and live serving has been verified for
version 2. The test endpoint has been deleted. To run the app again, deploy version 2
when one T4 is available in the project's serving quota:

```powershell
.\.venv-session4\Scripts\python.exe scripts/manage_endpoint.py deploy --model projects/123905721265/locations/europe-west2/models/8215968697160826880@2
.\.venv-session4\Scripts\python.exe scripts/test_online_endpoint.py
.\.venv-session4\Scripts\python.exe scripts/test_online_endpoint.py --sentence "The Force is strong in you."
.\.venv-session4\Scripts\python.exe scripts/run_chat.py
```

Open `http://localhost:8000`. The app applies the saved tokenizer's chat template,
authenticates its HTTP request, and displays the generated response. `run_chat.py`
sets the child process's `DEBUG` option to a valid boolean so IDE settings such as
`DEBUG=release` do not prevent Chainlit from starting. Restart the app after changes
to `.env` or deployment settings.

`manage_endpoint.py` uses `n1-standard-8`, one T4 GPU and one replica. It saves the
endpoint resource to `data/session4/endpoint.json` before starting deployment and updates
the ignored `.env` on success. `status` inspects that endpoint. `cleanup` checks the
endpoint's ownership labels, undeploys its models, deletes it and clears `GCP_ENDPOINT_ID`:

```powershell
.\.venv-session4\Scripts\python.exe scripts/manage_endpoint.py status
.\.venv-session4\Scripts\python.exe scripts/manage_endpoint.py cleanup
```

Cleanup is required at the end of the practice to stop provisioned serving resources.
Model Registry versions and GCS adapters remain available. Deploy again when needed.

The `.env.example` includes `GCP_PROJECT_NUMBER`, `GCP_ENDPOINT_ID`, `GCP_MODEL_RESOURCE`,
`INPUT_DATA_FILE`, `TOKENIZER_DIR` and `AUTH_MODE`. The default `adc` mode uses existing
Application Default Credentials and automatic token refresh; `gcloud` mode uses
`gcloud auth print-access-token` as in the course exercise. No access token is stored.
For a fresh checkout, download the Session 3 artifacts before starting the app so its
tokenizer is available, or set `TOKENIZER_DIR` to a local tokenizer directory.

To verify the local browser flow, run the app and then:

```powershell
uvx --from playwright python scripts/verify_chainlit_browser.py
# While no endpoint is deployed, verify only the offline error flow:
uvx --from playwright python scripts/verify_chainlit_browser.py --expect-offline
```

The browser script uses installed Microsoft Edge on Windows. On other systems, install
Chromium with `uvx --from playwright python -m playwright install chromium` first.
Browser screenshots and result manifests are saved in the ignored `data/session4/`.
The successful live-chat screenshot is also retained in the Session 4 report.
