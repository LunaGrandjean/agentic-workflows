"""Local non-LLM Ragas scoring helpers; no API key is required."""

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ragas.metrics import SingleTurnMetric


def compute_metrics(
    user_input: str, response: str, reference: str,
    metric_definitions: list["SingleTurnMetric"],
) -> dict[str, float]:
    from ragas import SingleTurnSample

    sample = SingleTurnSample(user_input=user_input, response=response, reference=reference)
    return {metric.name: float(metric.single_turn_score(sample)) for metric in metric_definitions}


def aggregate_metrics(per_sample_metrics: pd.DataFrame) -> dict[str, float]:
    if per_sample_metrics.empty:
        raise ValueError("Cannot aggregate an empty evaluation set.")
    return {name: float(value) for name, value in per_sample_metrics.select_dtypes(include="number").mean().items()}
