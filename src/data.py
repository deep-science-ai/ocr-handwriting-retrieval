from __future__ import annotations

from pathlib import Path

import polars as pl

from config import (
    DEFAULT_TEST_LIMIT,
    TESTING_LABELS,
    TESTING_WORDS,
    TRAINING_LABELS,
    TRAINING_WORDS,
    VALIDATION_LABELS,
    VALIDATION_WORDS,
)


def load_split_labels(labels_path: Path, words_dir: Path, limit: int | None = None) -> pl.DataFrame:
    labels = pl.read_csv(labels_path)
    if limit is not None:
        labels = labels.head(limit)
    return labels.with_columns((pl.lit(str(words_dir)) + "/" + pl.col("IMAGE")).alias("image_path"))


def load_testing_labels(limit: int | None = DEFAULT_TEST_LIMIT) -> pl.DataFrame:
    return load_split_labels(TESTING_LABELS, TESTING_WORDS, limit)


def load_training_labels(limit: int | None = None) -> pl.DataFrame:
    return load_split_labels(TRAINING_LABELS, TRAINING_WORDS, limit)


def load_validation_labels(limit: int | None = None) -> pl.DataFrame:
    return load_split_labels(VALIDATION_LABELS, VALIDATION_WORDS, limit)


def image_path(filename: str, split: str = "Testing") -> Path:
    words_dir = {
        "Training": TRAINING_WORDS,
        "Validation": VALIDATION_WORDS,
        "Testing": TESTING_WORDS,
    }[split]
    return words_dir / filename


def validate_testing_paths(limit: int | None = DEFAULT_TEST_LIMIT) -> pl.DataFrame:
    labels = load_testing_labels(limit)
    return labels.with_columns(
        pl.col("image_path").map_elements(lambda p: Path(p).exists(), return_dtype=pl.Boolean).alias("exists")
    )
