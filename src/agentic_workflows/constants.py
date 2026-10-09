import os

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
