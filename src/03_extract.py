#!/usr/bin/env python
"""Turn OCR text into structured fields for search and review.

This stage writes normalized text, category, medical relevance, review flags,
and searchable summaries back to the same LanceDB rows.
"""

from __future__ import annotations

import argparse
import os

from config import load_local_env
from db import merge_update, open_table
from dspy_programs import coerce_bool, heuristic_extract, make_medical_term_extractor


def rows_for_extraction(limit: int | None, overwrite: bool) -> list[dict]:
    table = open_table()
    row_limit = limit or table.count_rows()
    query = (
        table.search()
        .select(["id", "ocr_text", "generic_name", "needs_human_review"])
        .where("ocr_text IS NOT NULL")
        .limit(row_limit)
    )
    if not overwrite:
        query = query.where("ocr_text IS NOT NULL AND searchable_summary IS NULL")
    return query.to_list()


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser(description="Run structured extraction over OCR text.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit. Defaults to all pending rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--heuristic", action="store_true", help="Use deterministic local extraction instead of DSPy LM.")
    args = parser.parse_args()

    rows = rows_for_extraction(args.limit, args.overwrite)
    if not rows:
        print("No rows need extraction.")
        return

    use_heuristic = args.heuristic or not os.environ.get("OPENAI_API_KEY")
    extractor = None if use_heuristic else make_medical_term_extractor()
    if use_heuristic:
        print("Using deterministic heuristic extraction. Set OPENAI_API_KEY to use DSPy extraction.")

    updates = []
    for i, row in enumerate(rows, start=1):
        if use_heuristic:
            pred = heuristic_extract(row["ocr_text"], row["generic_name"], row["needs_human_review"])
            update = {
                "id": row["id"],
                "normalized_text": pred.normalized_text,
                "category": pred.category,
                "is_medical": pred.is_medical,
                "needs_human_review": pred.needs_human_review,
                "searchable_summary": pred.searchable_summary,
            }
        else:
            pred = extractor(ocr_text=row["ocr_text"])
            update = {
                "id": row["id"],
                "normalized_text": str(pred.normalized_text).strip(),
                "category": str(pred.category).strip().lower(),
                "is_medical": coerce_bool(pred.is_medical),
                "needs_human_review": coerce_bool(pred.needs_human_review) or bool(row["needs_human_review"]),
                "searchable_summary": str(pred.searchable_summary).strip(),
            }
        updates.append(update)
        if i % 50 == 0 or i == len(rows):
            print(f"Prepared extraction updates for {i}/{len(rows)} rows.")

    merge_update(updates)
    print(f"Wrote structured extraction fields for {len(updates)} rows.")


if __name__ == "__main__":
    main()
