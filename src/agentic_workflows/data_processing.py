from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"Could not find any of {candidates} in dataset columns: {columns}"
    )


def format_yoda_dataset(
    raw_dataset_uri: str,
    train_output_path: str,
    test_output_path: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[str, str]:
    dataframe = pd.read_csv(raw_dataset_uri)

    source_column = _pick_column(
        dataframe.columns.tolist(),
        ("sentence", "english", "source", "input", "original"),
    )
    target_column = _pick_column(
        dataframe.columns.tolist(),
        ("yoda", "translation", "target", "output", "yoda_sentence"),
    )

    dataset = Dataset.from_pandas(dataframe, preserve_index=False)

    def to_conversation(example: dict) -> dict:
        return {
            "messages": [
                {"role": "user", "content": str(example[source_column])},
                {"role": "assistant", "content": str(example[target_column])},
            ]
        }

    formatted_dataset = dataset.map(
        to_conversation,
        remove_columns=dataset.column_names,
    )
    split_dataset = formatted_dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )

    Path(train_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_output_path).parent.mkdir(parents=True, exist_ok=True)

    split_dataset["train"].to_csv(train_output_path, index=False)
    split_dataset["test"].to_csv(test_output_path, index=False)

    return train_output_path, test_output_path
