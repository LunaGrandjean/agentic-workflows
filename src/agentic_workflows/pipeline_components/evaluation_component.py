from kfp.dsl import Dataset, Input, Metrics, Output, component


@component(
    base_image="cicirello/pyaction:3.11",
    packages_to_install=[
        "ragas==0.2.15", "rouge-score==0.1.2", "pandas>=2.3.3",
        "langchain>=0.3,<0.4", "langchain-core>=0.3,<0.4",
        "langchain-community>=0.3,<0.4", "langchain-openai>=0.3,<0.4",
    ],
)
def evaluation_component(
    predictions: Input[Dataset],
    evaluation_results: Output[Dataset],
    metrics: Output[Metrics],
) -> None:
    import os
    from pathlib import Path

    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    import pandas as pd
    from ragas import SingleTurnSample
    from ragas.metrics import RougeScore

    frame = pd.read_csv(predictions.path, keep_default_na=False)
    required = {"user_input", "reference", "extracted_response"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError("Cannot evaluate an empty predictions file.")
    definitions = [RougeScore(rouge_type=kind, name=kind) for kind in ("rouge1", "rouge2", "rougeL")]
    rows = []
    for row in frame.to_dict(orient="records"):
        sample = SingleTurnSample(
            user_input=row["user_input"], response=row["extracted_response"], reference=row["reference"],
        )
        rows.append({metric.name: float(metric.single_turn_score(sample)) for metric in definitions})
    scores = pd.DataFrame(rows)
    Path(evaluation_results.path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat([frame.reset_index(drop=True), scores], axis=1).to_csv(evaluation_results.path, index=False)
    for name, value in scores.mean().items():
        metrics.log_metric(name, float(value))
    metrics.log_metric("num_samples", len(frame))
