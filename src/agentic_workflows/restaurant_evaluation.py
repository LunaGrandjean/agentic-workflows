"""Reference-aware ROUGE comparison without an LLM judge or API key."""

from agentic_workflows.restaurant_data import parse_restaurant_mr


def score_restaurant_predictions(frame):
    import json

    import pandas as pd
    from rouge_score.rouge_scorer import RougeScorer

    required = {"mr_id", "meaning_representation", "references", "extracted_response"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Expected nonempty restaurant predictions with {sorted(required)}.")
    if frame["mr_id"].duplicated().any():
        raise ValueError("Evaluate each meaning representation once, with all its references.")
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for row in frame.to_dict("records"):
        references = json.loads(row["references"])
        if not isinstance(references, list) or not references or not all(isinstance(ref, str) and ref.strip() for ref in references):
            raise ValueError("Each evaluation input must have a nonempty list of reference strings.")
        scores = {}
        entities = [value for key, value in parse_restaurant_mr(row["meaning_representation"]) if key in ("name", "near")]
        for label, column in (("finetuned", "extracted_response"), ("base", "base_response")):
            if column not in row:
                continue
            response = row[column]
            if not isinstance(response, str):
                raise ValueError(f"Invalid response in {column}.")
            per_reference = [scorer.score(reference, response) for reference in references]
            for kind in ("rouge1", "rouge2", "rougeL"):
                scores[f"{label}_{kind}"] = max(result[kind].fmeasure for result in per_reference)
            scores[f"{label}_empty_response"] = float(not response.strip())
            scores[f"{label}_response_words"] = len(response.split())
            # Diagnostic only: literal names/landmarks, not full semantic correctness.
            scores[f"{label}_entity_coverage"] = sum(value.casefold() in response.casefold() for value in entities) / len(entities) if entities else float("nan")
        rows.append(scores)
    scores = pd.DataFrame(rows)
    summary = {name: float(value) for name, value in scores.mean().items() if pd.notna(value)}
    if "base_response" in frame:
        for kind in ("rouge1", "rouge2", "rougeL"):
            summary[f"delta_{kind}"] = summary[f"finetuned_{kind}"] - summary[f"base_{kind}"]
    summary["num_samples"] = len(frame)
    return pd.concat([frame.reset_index(drop=True), scores], axis=1), summary
