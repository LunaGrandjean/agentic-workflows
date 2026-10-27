"""Send an authenticated HTTP prediction request and save its JSON response."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    from agentic_workflows.constants import AUTH_MODE, GCP_ENDPOINT_ID, GCP_PROJECT_ID, GCP_PROJECT_NUMBER, GCP_REGION, INPUT_DATA_FILE
    from agentic_workflows.online_prediction import endpoint_url, post_prediction, prediction_text, translate_sentence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=Path(INPUT_DATA_FILE))
    parser.add_argument("--sentence", help="Apply the tokenizer's chat template and translate this sentence.")
    args = parser.parse_args()
    if args.sentence:
        result = {"sentence": args.sentence, "response": translate_sentence(args.sentence)}
    else:
        payload = json.loads(args.input_file.read_text(encoding="utf-8"))
        result = post_prediction(endpoint_url(GCP_PROJECT_NUMBER or GCP_PROJECT_ID, GCP_REGION, GCP_ENDPOINT_ID), payload, auth_mode=AUTH_MODE)
        print(prediction_text(result), flush=True)
    output = PROJECT_ROOT / "data/session4/online_prediction.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
