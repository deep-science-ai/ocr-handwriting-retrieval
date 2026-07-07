#!/usr/bin/env python
"""Embed searchable summaries and build LanceDB indexes.

This stage fills the vector column and creates scalar, full-text, and vector
indexes that power the retrieval scripts.
"""

from __future__ import annotations

import argparse

from lancedb.index import BTree, Bitmap, FTS, IvfFlat

from db import merge_update, open_table
from embeddings import encode_texts, load_embedder


def rows_for_embedding(limit: int | None, overwrite: bool) -> list[dict]:
    table = open_table()
    row_limit = limit or table.count_rows()
    where = "searchable_summary IS NOT NULL"
    if not overwrite:
        where += " AND vector IS NULL"
    return table.search().select(["id", "searchable_summary"]).where(where).limit(row_limit).to_list()


def build_indexes(table) -> None:
    print("Building scalar indexes.")
    for column, config in [("id", BTree()), ("category", Bitmap()), ("needs_human_review", Bitmap())]:
        try:
            table.create_index(column, config=config, replace=True)
            print(f"Created scalar index on {column}.")
        except Exception as exc:
            print(f"Could not create scalar index on {column}: {exc}")

    print("Building full-text index on searchable_summary.")
    try:
        table.create_index("searchable_summary", config=FTS(), replace=True)
        print("Created full-text index on searchable_summary.")
    except Exception as exc:
        print(f"Could not create full-text index: {exc}")

    print("Building vector index on vector.")
    table.create_index("vector", config=IvfFlat(distance_type="cosine", num_partitions=16), replace=True)
    print("Created cosine vector index on vector.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed searchable summaries and build LanceDB indexes.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit. Defaults to all pending rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-embeddings", action="store_true", help="Fail instead of using local fallback embeddings.")
    args = parser.parse_args()

    rows = rows_for_embedding(args.limit, args.overwrite)
    if not rows:
        print("No rows need embeddings.")
    else:
        model = load_embedder(allow_fallback=not args.strict_embeddings)
        texts = [row["searchable_summary"] for row in rows]
        vectors = encode_texts(model, texts)
        updates = [{"id": row["id"], "vector": vector} for row, vector in zip(rows, vectors)]
        merge_update(updates)
        print(f"Wrote vectors for {len(updates)} rows using {getattr(model, 'name', type(model).__name__)}.")

    table = open_table()
    rows_with_vectors = table.search().select(["id"]).where("vector IS NOT NULL").limit(table.count_rows()).to_list()
    if not rows_with_vectors:
        raise RuntimeError("No vectors found. Run src/03_extract.py before indexing.")
    build_indexes(table)
    print(f"Indexed {len(rows_with_vectors)} vectorized rows.")


if __name__ == "__main__":
    main()
