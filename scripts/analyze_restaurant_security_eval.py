"""Score raw and guarded outputs from a held-out restaurant evaluation run."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agentic_workflows.restaurant_data import (
    parse_restaurant_mr,
    restaurant_response_needs_fallback,
    safe_restaurant_fallback,
)
from agentic_workflows.restaurant_evaluation import score_restaurant_predictions


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path, help="Directory containing evaluation_results.csv and preparation.json")
    args = parser.parse_args()
    results_path = args.evaluation_dir / "evaluation_results.csv"
    preparation_path = args.evaluation_dir / "preparation.json"
    if not results_path.is_file() or not preparation_path.is_file():
        parser.error("The directory must contain evaluation_results.csv and preparation.json from a completed run.")

    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    prompt_version = preparation.get("prompt_version")
    if prompt_version != "restaurant-v2-security":
        raise ValueError(f"Expected restaurant-v2-security predictions, found {prompt_version!r}.")

    frame = pd.read_csv(results_path, keep_default_na=False)
    required = {"meaning_representation", "extracted_response", "mr_id", "references"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Predictions must be nonempty and include {sorted(required)}.")

    accepted_positions = []
    rejections = []
    for position, row in enumerate(frame.to_dict("records")):
        try:
            pairs = parse_restaurant_mr(row["meaning_representation"])
            if len(dict(pairs)) != len(pairs):
                raise ValueError("Provide each restaurant attribute once to avoid conflicting instructions.")
        except ValueError as exc:
            rejections.append({"mr_id": row["mr_id"], "meaning_representation": row["meaning_representation"], "reason": str(exc)})
        else:
            accepted_positions.append(position)

    if not accepted_positions:
        raise ValueError("The serving parser rejected every held-out input.")
    frame = frame.iloc[accepted_positions].reset_index(drop=True)
    guarded = frame.copy()
    guarded_outputs = []
    fallback_flags = []
    for row in frame.to_dict("records"):
        pairs = parse_restaurant_mr(row["meaning_representation"])
        raw_output = row["extracted_response"]
        use_fallback = restaurant_response_needs_fallback(pairs, raw_output)
        fallback_flags.append(use_fallback)
        guarded_outputs.append(safe_restaurant_fallback(pairs) if use_fallback else raw_output)

    score_columns = [
        column for column in frame.columns
        if column.startswith(("finetuned_", "base_", "delta_")) and column != "base_response"
    ]
    scoring_frame = frame.drop(columns=score_columns)
    raw_results, raw_summary = score_restaurant_predictions(scoring_frame)
    guarded = scoring_frame.copy()
    guarded["extracted_response"] = guarded_outputs
    guarded_results, guarded_summary = score_restaurant_predictions(guarded)

    results = frame.copy()
    results["restaurant_v2_security_raw_response"] = frame["extracted_response"]
    results["restaurant_v2_security_guarded_response"] = guarded_outputs
    results["restaurant_v2_security_fallback_applied"] = fallback_flags
    for column in raw_results.columns:
        if column.startswith("finetuned_"):
            results[f"raw_{column}"] = raw_results[column]
    for column in guarded_results.columns:
        if column.startswith("finetuned_"):
            results[f"guarded_{column}"] = guarded_results[column]
    results.to_csv(args.evaluation_dir / "security_v2_evaluation.csv", index=False)
    pd.DataFrame(rejections, columns=["mr_id", "meaning_representation", "reason"]).to_csv(
        args.evaluation_dir / "security_v2_rejections.csv", index=False,
    )

    summary = {
        "prompt_version": prompt_version,
        "test_rows_total": len(accepted_positions) + len(rejections),
        "accepted_rows_scored": len(frame),
        "input_rows_rejected_before_serving": len(rejections),
        "fallback_count": sum(fallback_flags),
        "fallback_rate": sum(fallback_flags) / len(fallback_flags),
        "raw_model_metrics": raw_summary,
        "guarded_serving_metrics": guarded_summary,
        "note": "The raw metrics score model outputs; guarded metrics score the outputs actually returned after the application fallback guard. ROUGE is lexical, not a complete security or factuality measure.",
    }
    prior_v1_path = args.evaluation_dir.parent / "final_test" / "evaluation_results.csv"
    if prior_v1_path.is_file():
        prior_v1 = pd.read_csv(prior_v1_path, keep_default_na=False)
        accepted_ids = set(frame["mr_id"])
        prior_v1 = prior_v1.loc[prior_v1["mr_id"].isin(accepted_ids)].copy()
        prior_metric_columns = [
            column for column in prior_v1.columns
            if column.startswith(("finetuned_", "base_", "delta_")) and column != "base_response"
        ]
        prior_v1 = prior_v1.drop(columns=prior_metric_columns)
        _, prior_v1_summary = score_restaurant_predictions(prior_v1)
        summary["restaurant_v1_previous_predictions_same_accepted_inputs"] = prior_v1_summary
    (args.evaluation_dir / "security_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
