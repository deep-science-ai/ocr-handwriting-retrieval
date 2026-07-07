#!/usr/bin/env python
"""Run the baseline dense retrieval path against LanceDB.

Queries are embedded once, searched through LanceDB's vector index, and returned
with decoded image previews plus labels, OCR text, and extraction metadata.
"""

from __future__ import annotations

import argparse

from db import open_table
from embeddings import encode_texts, load_embedder
from rendering import preview_dir, save_image_preview, slugify

DEFAULT_QUERIES = [
    "antibiotic medication",
    "pain medication",
    "fever",
    "medicine taken twice daily",
    "prescription term related to headache",
]


def search(query: str, limit: int, *, strict_embeddings: bool) -> list[dict]:
    model = load_embedder(allow_fallback=not strict_embeddings)
    query_vector = encode_texts(model, [query])[0]
    table = open_table()
    return (
        table.search(query_vector)
        .metric("cosine")
        .select(
            [
                "id",
                "image",
                "medicine_name",
                "generic_name",
                "ocr_text",
                "normalized_text",
                "category",
                "needs_human_review",
                "searchable_summary",
                "_distance",
            ]
        )
        .limit(limit)
        .to_list()
    )


def print_results(query: str, rows: list[dict]) -> None:
    out_dir = preview_dir("search", query)
    print(f"\nQuery: {query}")
    print(f"Image previews: {out_dir}")
    for rank, row in enumerate(rows, start=1):
        preview = save_image_preview(row["image"], out_dir, f"{rank:02d}_{slugify(row['id'])}.png")
        distance = row.get("_distance")
        score = "" if distance is None else f" distance={distance:.4f}"
        print(
            f"{rank:02d}.{score} id={row['id']} label={row['medicine_name']} "
            f"generic={row['generic_name']} ocr={row.get('ocr_text')!r} "
            f"normalized={row.get('normalized_text')!r} category={row.get('category')} "
            f"review={row.get('needs_human_review')} preview={preview}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline dense semantic search over LanceDB.")
    parser.add_argument("queries", nargs="*", default=DEFAULT_QUERIES)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--strict-embeddings", action="store_true")
    args = parser.parse_args()

    for query in args.queries:
        print_results(query, search(query, args.limit, strict_embeddings=args.strict_embeddings))


if __name__ == "__main__":
    main()
