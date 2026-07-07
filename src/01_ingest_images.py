#!/usr/bin/env python
"""Create the LanceDB table and store Testing images as raw bytes.

This is the first durable pipeline stage: each handwritten PNG is stored in the
table itself, alongside validation-only labels, then the local table is compacted.
"""

from __future__ import annotations

import argparse

from config import SOURCE_DATASET
from data import load_testing_labels
from db import create_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Testing image bytes into LanceDB.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit. Defaults to all Testing rows.")
    parser.add_argument("--mode", choices=["overwrite", "create"], default="overwrite")
    args = parser.parse_args()

    labels = load_testing_labels(args.limit)
    rows = []
    for i, row in enumerate(labels.iter_rows(named=True)):
        image_path = row["image_path"]
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        rows.append(
            {
                "id": f"test_{i:05d}",
                "split": "Testing",
                "image_filename": row["IMAGE"],
                "image": image_bytes,
                "medicine_name": row["MEDICINE_NAME"],
                "generic_name": row["GENERIC_NAME"],
                "source_dataset": SOURCE_DATASET,
            }
        )

    table = create_table(mode=args.mode)
    table.add(rows)
    table.optimize()
    print(f"Ingested {len(rows)} Testing images into LanceDB table {table.name!r}.")
    print("The image column contains raw PNG bytes; downstream stages read images back from LanceDB.")
    print("Optimized the local LanceDB table after ingestion.")


if __name__ == "__main__":
    main()
