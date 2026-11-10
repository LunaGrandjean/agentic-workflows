"""Audit conflicting slots and reference-text overlap in the frozen raw splits."""

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import pandas as pd
    from agentic_workflows.restaurant_data import parse_restaurant_mr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/restaurant/raw")
    args = parser.parse_args()
    manifest = json.loads((args.raw_dir / "manifest.json").read_text(encoding="utf-8"))
    quality = {"snapshot": manifest["parquet_revision"], "conflicting_slot_rows": {}, "reference_overlap": {}}
    texts = {}
    for split in ("train", "validation", "test"):
        frame = pd.read_parquet(args.raw_dir / f"{split}.parquet")
        conflicts = 0
        references = set()
        for row in frame.to_dict("records"):
            values = {}
            for key, value in parse_restaurant_mr(row["meaning_representation"]):
                values.setdefault(key, set()).add(value)
            conflicts += any(len(options) > 1 for options in values.values())
            for reference in [row["target"], *list(row["references"])]:
                if reference.strip():
                    references.add(re.sub(r"\s+", " ", reference).strip().casefold())
        texts[split] = references
        quality["conflicting_slot_rows"][split] = conflicts
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        quality["reference_overlap"][f"{first}_{second}"] = len(texts[first] & texts[second])
    (ROOT / "reports/restaurant_data_quality.json").write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(quality, indent=2), flush=True)


if __name__ == "__main__":
    main()
