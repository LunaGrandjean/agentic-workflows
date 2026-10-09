from kfp.dsl import OutputPath, component


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "datasets>=4.1.1,<6",
        "gcsfs>=2025.9.0,<2026.8.0",
        "pandas>=2.3.3",
    ],
)
def transform_yoda_data(
    raw_dataset_uri: str,
    test_size: float,
    seed: int,
    train_dataset: OutputPath("Dataset"),
    test_dataset: OutputPath("Dataset"),
) -> None:
    import logging
    from pathlib import Path

    import pandas as pd
    from datasets import Dataset

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def pick_column(columns: list[str], candidates: tuple[str, ...]) -> str:
        normalized = {column.lower().strip(): column for column in columns}
        for candidate in candidates:
            if candidate in normalized:
                return normalized[candidate]
        raise ValueError(
            f"Could not find any of {candidates} in dataset columns: {columns}"
        )

    logger.info("Reading raw dataset from %s", raw_dataset_uri)
    dataframe = pd.read_csv(raw_dataset_uri)
    logger.info("Loaded %s rows and columns %s", len(dataframe), dataframe.columns.tolist())

    source_column = pick_column(
        dataframe.columns.tolist(),
        ("sentence", "english", "source", "input", "original"),
    )
    target_column = pick_column(
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
    split_dataset = formatted_dataset.train_test_split(test_size=test_size, seed=seed)

    Path(train_dataset).parent.mkdir(parents=True, exist_ok=True)
    Path(test_dataset).parent.mkdir(parents=True, exist_ok=True)

    split_dataset["train"].to_csv(train_dataset, index=False)
    split_dataset["test"].to_csv(test_dataset, index=False)

    logger.info("Wrote train dataset to %s", train_dataset)
    logger.info("Wrote test dataset to %s", test_dataset)
