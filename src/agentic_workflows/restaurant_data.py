"""GEM/E2E preparation shared by local execution and the Kubeflow component."""


def parse_restaurant_mr(text: str) -> list[tuple[str, str]]:
    """Parse the E2E slot syntax without silently losing repeated slots."""
    import re

    if not isinstance(text, str) or not text.strip() or len(text) > 2048:
        raise ValueError("Enter restaurant attributes as name[The Eagle], food[Italian].")
    allowed = {"name", "eatType", "food", "priceRange", "customer rating", "area", "familyFriendly", "near"}
    pairs = []
    position = 0
    pattern = re.compile(r"\s*([^\[\],]+)\[([^\[\]]+)\]\s*(?:,\s*|$)")
    while position < len(text):
        match = pattern.match(text, position)
        if match is None:
            raise ValueError("Use comma-separated attributes such as name[The Eagle], food[Italian].")
        key, value = (part.strip() for part in match.groups())
        if key not in allowed or not value:
            raise ValueError(f"Unknown or empty restaurant attribute: {key}.")
        pairs.append((key, value))
        position = match.end()
    return pairs


def canonical_restaurant_mr(text: str) -> str:
    pairs = parse_restaurant_mr(text.strip())
    order = {key: i for i, key in enumerate(("name", "eatType", "food", "priceRange", "customer rating", "area", "familyFriendly", "near"))}
    return ", ".join(f"{key}[{value}]" for key, value in sorted(pairs, key=lambda pair: (order[pair[0]], pair[1])))


def restaurant_prompt(mr: str) -> str:
    return (
        "Write a short restaurant description in English using all and only the facts below. "
        "Do not invent missing details.\n\nRestaurant attributes:\n" + canonical_restaurant_mr(mr)
    )


def prepare_restaurant_frames(raw_frames: dict, limits: dict | None = None, seed: int = 42) -> tuple[dict, dict]:
    """Keep official splits, group evaluation references and audit canonical overlap."""
    import hashlib
    import json

    import pandas as pd

    limits = limits or {}
    audit = {"dataset": "GEM/e2e_nlg", "seed": seed, "splits": {}}
    keyed = {}
    for split in ("train", "validation", "test"):
        frame = raw_frames[split].copy()
        required = {"gem_id", "meaning_representation", "target", "references"}
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError(f"Invalid or empty {split} split; expected {sorted(required)}.")
        if not frame["target"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Empty reference in {split}.")
        frame["mr_key"] = frame["meaning_representation"].map(canonical_restaurant_mr)
        keyed[split] = frame
        audit["splits"][split] = {"raw_rows": len(frame), "unique_mrs": frame["mr_key"].nunique()}
    keys = {split: set(frame["mr_key"]) for split, frame in keyed.items()}
    overlap = {
        "train_validation": len(keys["train"] & keys["validation"]),
        "train_test": len(keys["train"] & keys["test"]),
        "validation_test": len(keys["validation"] & keys["test"]),
    }
    audit["canonical_overlap_before_filtering"] = overlap
    if overlap["validation_test"]:
        raise ValueError("Validation/test inputs overlap after canonicalization; resolve this before evaluation.")
    # Never change held-out data to repair a leak from training.
    held_out = keys["validation"] | keys["test"]
    keep = ~keyed["train"]["mr_key"].isin(held_out)
    audit["training_rows_removed_for_overlap"] = int((~keep).sum())
    keyed["train"] = keyed["train"].loc[keep]
    prepared = {}
    for split, frame in keyed.items():
        records = []
        groups = frame.groupby("mr_key", sort=False) if split != "train" else ((row["mr_key"], pd.DataFrame([row])) for row in frame.to_dict("records"))
        for mr, group in groups:
            references = []
            for row in group.to_dict("records"):
                for reference in [row["target"], *list(row["references"])]:
                    if isinstance(reference, str) and reference.strip() and reference not in references:
                        references.append(reference)
            prompt = restaurant_prompt(mr)
            records.append({
                "gem_id": group.iloc[0]["gem_id"], "source_split": split,
                "meaning_representation": mr,
                "mr_id": hashlib.sha256(mr.encode("utf-8")).hexdigest(),
                "prompt": prompt, "target": references[0],
                "references": json.dumps(references, ensure_ascii=False),
                "messages": json.dumps([{"role": "user", "content": prompt}, {"role": "assistant", "content": references[0]}], ensure_ascii=False),
            })
        result = pd.DataFrame(records)
        limit = limits.get(split, 0)
        if type(limit) is not int or limit < 0:
            raise ValueError("Split limits must be nonnegative integers; zero uses all rows.")
        if limit:
            result = result.sample(n=min(limit, len(result)), random_state=seed).sort_index()
        if result.empty:
            raise ValueError(f"No prepared rows in {split}.")
        prepared[split] = result.reset_index(drop=True)
        audit["splits"][split].update(prepared_rows=len(result), limit=limit,
            grouped_references=(split != "train"),
            repeated_slot_rows=sum(len(parse_restaurant_mr(mr)) != len(dict(parse_restaurant_mr(mr))) for mr in result["meaning_representation"]))
    audit["prompt_version"] = "restaurant-v1"
    audit["evaluation_protocol"] = "Macro mean over unique MRs; maximum ROUGE F1 over the union of references for each MR."
    return prepared, audit


def write_restaurant_splits(raw_paths: dict, output_dir: str, limits: dict | None = None, seed: int = 42) -> dict:
    import json
    from pathlib import Path

    import pandas as pd

    frames, audit = prepare_restaurant_frames({key: pd.read_parquet(value) for key, value in raw_paths.items()}, limits, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame.to_csv(output / f"{split}.csv", index=False)
    (output / "preparation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
