from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentic_workflows.constants import RAW_DATASET_URI
from agentic_workflows.data_processing import format_yoda_dataset


def main() -> None:
    train_path, test_path = format_yoda_dataset(
        raw_dataset_uri=RAW_DATASET_URI,
        train_output_path=str(PROJECT_ROOT / "data/processed/train.csv"),
        test_output_path=str(PROJECT_ROOT / "data/processed/test.csv"),
    )
    print(f"Train dataset written to {train_path}")
    print(f"Test dataset written to {test_path}")


if __name__ == "__main__":
    main()
