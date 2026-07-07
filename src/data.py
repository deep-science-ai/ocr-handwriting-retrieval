from __future__ import annotations

from pathlib import Path

import polars as pl

from config import DEFAULT_TEST_LIMIT, TESTING_LABELS, TESTING_WORDS


def load_testing_labels(limit: int | None = DEFAULT_TEST_LIMIT) -> pl.DataFrame:
    labels = pl.read_csv(TESTING_LABELS)
    if limit is not None:
        labels = labels.head(limit)
    return labels.with_columns(
        (pl.lit(str(TESTING_WORDS)) + "/" + pl.col("IMAGE")).alias("image_path")
    )


def image_path(filename: str) -> Path:
    return TESTING_WORDS / filename


def validate_testing_paths(limit: int | None = DEFAULT_TEST_LIMIT) -> pl.DataFrame:
    labels = load_testing_labels(limit)
    return labels.with_columns(
        pl.col("image_path").map_elements(lambda p: Path(p).exists(), return_dtype=pl.Boolean).alias("exists")
    )
