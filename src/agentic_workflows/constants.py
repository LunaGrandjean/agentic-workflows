import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME", "lulik")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "agentics-workflows-510809")
GCP_REGION = os.getenv("GCP_REGION", "europe-west2")
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", f"{TEAM_NAME}-llmops")

RAW_DATASET_URI = os.getenv(
    "RAW_DATASET_URI",
    f"gs://{GCP_BUCKET_NAME}/yoda_sentences.csv",
)
PIPELINE_ROOT = os.getenv(
    "PIPELINE_ROOT",
    f"gs://{GCP_BUCKET_NAME}/pipelines",
)
PROCESSED_DATASET_PREFIX = os.getenv(
    "PROCESSED_DATASET_PREFIX",
    f"gs://{GCP_BUCKET_NAME}/processed/yoda",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_TASK = os.getenv("WORKFLOW_TASK", "restaurant")
if WORKFLOW_TASK not in {"restaurant", "yoda"}:
    raise ValueError("WORKFLOW_TASK must be restaurant or yoda.")
TASK_OUTPUT_DIR = PROJECT_ROOT / ("data/restaurant" if WORKFLOW_TASK == "restaurant" else "data/session4")
TASK_MODEL_NAME = f"{TEAM_NAME}-restaurant-phi3" if WORKFLOW_TASK == "restaurant" else f"{TEAM_NAME}-yoda-phi3"
TASK_ENDPOINT_NAME = f"{TEAM_NAME}-restaurant-final" if WORKFLOW_TASK == "restaurant" else f"{TEAM_NAME}-yoda-session4"
TASK_LABELS = {"project-task": WORKFLOW_TASK, "team": TEAM_NAME} if WORKFLOW_TASK == "restaurant" else {"course-session": "4", "team": TEAM_NAME}
ENDPOINT_ENV_KEY = "RESTAURANT_ENDPOINT_ID" if WORKFLOW_TASK == "restaurant" else "GCP_ENDPOINT_ID"
MODEL_ENV_KEY = "RESTAURANT_MODEL_RESOURCE" if WORKFLOW_TASK == "restaurant" else "GCP_MODEL_RESOURCE"
GCP_PROJECT_NUMBER = os.getenv("GCP_PROJECT_NUMBER", "")
GCP_ENDPOINT_ID = os.getenv(ENDPOINT_ENV_KEY, "")
GCP_MODEL_RESOURCE = os.getenv(MODEL_ENV_KEY, "")
INPUT_DATA_FILE = os.getenv("RESTAURANT_INPUT_DATA_FILE", str(PROJECT_ROOT / "examples/restaurant_prediction_request.json")) if WORKFLOW_TASK == "restaurant" else os.getenv("INPUT_DATA_FILE", str(PROJECT_ROOT / "examples/prediction_request.json"))
TOKENIZER_DIR = os.getenv("RESTAURANT_TOKENIZER_DIR", str(PROJECT_ROOT / "models/phi3-restaurant-adapter")) if WORKFLOW_TASK == "restaurant" else os.getenv("TOKENIZER_DIR", str(PROJECT_ROOT / "models/phi3-yoda-adapter"))
AUTH_MODE = os.getenv("AUTH_MODE", "adc")
SERVING_CONTAINER_IMAGE_URI = os.getenv(
    "SERVING_CONTAINER_IMAGE_URI",
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
    "huggingface-pytorch-inference-cu121.2-3.transformers.4-46.ubuntu2204.py311",
)
