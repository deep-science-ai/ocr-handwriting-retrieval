#!/usr/bin/env python
"""Create the LanceDB table and store Testing images as raw bytes.

This is the first durable pipeline stage: each handwritten PNG is stored in the
table itself, alongside validation-only labels, then the local table is compacted.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import geneva

from config import SOURCE_DATASET, TABLE_NAME, load_local_env
from data import load_testing_labels
from db import DoctorHandwriting, create_table


def build_rows(limit: int | None) -> list[dict[str, Any]]:
    labels = load_testing_labels(limit)
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
    return rows


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Ingest Testing image bytes into LanceDB.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit. Defaults to all Testing rows.")
    parser.add_argument(
        "--mode",
        choices=["overwrite", "create"],
        default=None,
        help="Defaults to overwrite locally and create remotely.",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Create/populate the table in LanceDB Enterprise using .env.",
    )
    # For Enterprise iteration, pass a fresh run table name instead of
    # drop/recreate on the canonical table; cleanup of dropped storage prefixes can lag.
    parser.add_argument(
        "--table-name",
        default=TABLE_NAME,
        help=f"Destination table name. Defaults to {TABLE_NAME!r}.",
    )
    args = parser.parse_args()

    rows = build_rows(args.limit)
    mode = args.mode or ("create" if args.remote else "overwrite")

    if args.remote:
        db = geneva.connect(
            os.environ["LANCEDB_URI"],
            api_key=os.environ["LANCEDB_API_KEY"],
            region=os.environ["LANCEDB_REGION"],
            host_override=os.environ["LANCEDB_HOST_OVERRIDE"],
        )
        table = db.create_table(args.table_name, schema=DoctorHandwriting, mode=mode)
    else:
        table = create_table(mode=mode, table_name=args.table_name)
    table.add(rows)

    print(f"Ingested {len(rows)} Testing images into LanceDB table {table.name!r}.")
    print("The image column contains raw PNG bytes; downstream stages read images back from LanceDB.")
    if args.remote:
        print("Remote Enterprise ingest complete; skipped local optimize().")
    else:
        table.optimize()
        print("Optimized the local LanceDB table after ingestion.")


if __name__ == "__main__":
    main()
