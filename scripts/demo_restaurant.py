"""Run live restaurant examples, or explicitly replay recorded pipeline predictions."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributes")
    parser.add_argument("--replay", type=Path, help="Recorded predictions/evaluation CSV. This does not call Vertex.")
    args = parser.parse_args()
    if args.replay:
        import pandas as pd

        print("RECORDED PIPELINE OUTPUTS — not live inference", flush=True)
        frame = pd.read_csv(args.replay, keep_default_na=False)
        for row in frame.head(5).to_dict("records"):
            print("\nAttributes:", row["meaning_representation"])
            print("Base Phi-3:", row.get("base_response", "not recorded"))
            print("Fine-tuned:", row["extracted_response"])
        return
    from agentic_workflows.online_prediction import describe_restaurant

    examples = [{"title": "Custom example", "attributes": args.attributes}] if args.attributes else json.loads((ROOT / "examples/restaurants.json").read_text(encoding="utf-8"))
    print("LIVE VERTEX PREDICTIONS", flush=True)
    responses = []
    for example in examples:
        response = describe_restaurant(example["attributes"])
        responses.append(response)
        print(f"\n{example['title']}\n{example['attributes']}\n=> {response}", flush=True)
    if not args.attributes:
        print("\nSame description after slot-order permutation:", responses[0] == responses[-1], flush=True)
    output = ROOT / "data/restaurant/live_demo.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"mode": "live", "examples": examples, "responses": responses}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
