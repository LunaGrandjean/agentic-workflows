from kfp.dsl import Dataset, Input, Metrics, Output, component

from agentic_workflows.restaurant_data import parse_restaurant_mr
from agentic_workflows.restaurant_evaluation import score_restaurant_predictions


@component(
    base_image="python:3.11-slim",
    packages_to_install=["rouge-score==0.1.2", "pandas>=2.3.3"],
    additional_funcs=[parse_restaurant_mr, score_restaurant_predictions],
)
def evaluate_restaurant_predictions(
    predictions: Input[Dataset], evaluation_results: Output[Dataset], metrics: Output[Metrics],
) -> None:
    from pathlib import Path

    import pandas as pd

    frame = pd.read_csv(predictions.path, keep_default_na=False)
    results, summary = score_restaurant_predictions(frame)
    Path(evaluation_results.path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(evaluation_results.path, index=False)
    for name, value in summary.items():
        metrics.log_metric(name, value)
    metrics.metadata["protocol"] = "Maximum ROUGE F1 per reference, macro average per unique MR; Porter stemming. Entity coverage is a literal diagnostic."
