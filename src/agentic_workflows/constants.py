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
GCP_PROJECT_NUMBER = os.getenv("GCP_PROJECT_NUMBER", "")
GCP_ENDPOINT_ID = os.getenv("GCP_ENDPOINT_ID", "")
GCP_MODEL_RESOURCE = os.getenv("GCP_MODEL_RESOURCE", "")
INPUT_DATA_FILE = os.getenv("INPUT_DATA_FILE", str(PROJECT_ROOT / "examples/prediction_request.json"))
TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", str(PROJECT_ROOT / "models/phi3-yoda-adapter"))
AUTH_MODE = os.getenv("AUTH_MODE", "adc")
SERVING_CONTAINER_IMAGE_URI = os.getenv(
    "SERVING_CONTAINER_IMAGE_URI",
    "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
    "huggingface-pytorch-inference-cu121.2-3.transformers.4-46.ubuntu2204.py311",
)
