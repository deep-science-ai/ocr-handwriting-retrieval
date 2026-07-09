#!/usr/bin/env python
"""Publish held-out Testing metrics for an OCR/extraction pipeline run.

This reads OCR/extraction outputs from LanceDB, compares OCR text with ground
truth labels, and writes a Markdown metrics table.
"""

from __future__ import annotations

import argparse
from statistics import mean, median

from config import ensure_output_dir
from db import open_table
from metrics import edit_distance, normalize_text


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{(100 * numerator / denominator):.1f}%"


def bool_count(rows: list[dict], column: str) -> int:
    return sum(1 for row in rows if bool(row.get(column)))


def rows_for_evaluation(limit: int | None, split: str) -> list[dict]:
    table = open_table()
    row_limit = limit or table.count_rows()
    return (
        table.search()
        .select(
            [
                "id",
                "split",
                "medicine_name",
                "generic_name",
                "ocr_text",
                "ocr_model",
                "normalized_text",
                "category",
                "is_medical",
                "needs_human_review",
                "searchable_summary",
            ]
        )
        .where(f"split = '{split}' AND ocr_text IS NOT NULL")
        .limit(row_limit)
        .to_list()
    )


def evaluate(rows: list[dict], split: str, run_name: str) -> tuple[str, list[dict]]:
    evaluated = []
    for row in rows:
        exact = (row.get("ocr_text") or "").strip() == row["medicine_name"].strip()
        normalized = normalize_text(row.get("ocr_text")) == normalize_text(row["medicine_name"])
        distance = edit_distance(row.get("ocr_text"), row["medicine_name"])
        extraction_complete = all(
            row.get(column) is not None
            for column in ["normalized_text", "category", "is_medical", "needs_human_review", "searchable_summary"]
        )
        evaluated.append(
            {
                **row,
                "exact_match": exact,
                "normalized_match": normalized,
                "edit_distance": distance,
                "extraction_complete": extraction_complete,
            }
        )

    total = len(evaluated)
    distances = [row["edit_distance"] for row in evaluated]
    models = sorted({row.get("ocr_model") or "unknown" for row in evaluated})
    model_label = ", ".join(models)
    table = [
        "| Run | OCR model | Split | Rows | Exact OCR match | Normalized OCR match | Avg edit distance | Median edit distance | Human review flagged | Extraction coverage |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {run_name} | `{model_label}` | {split} | {total} | "
            f"{percent(bool_count(evaluated, 'exact_match'), total)} | "
            f"{percent(bool_count(evaluated, 'normalized_match'), total)} | "
            f"{mean(distances):.2f} | {median(distances):.2f} | "
            f"{percent(bool_count(evaluated, 'needs_human_review'), total)} | "
            f"{percent(bool_count(evaluated, 'extraction_complete'), total)} |"
        ),
    ]
    return "\n".join(table), evaluated


def disagreement_table(rows: list[dict], limit: int) -> str:
    disagreements = [row for row in rows if not row["normalized_match"]]
    disagreements.sort(key=lambda row: row["edit_distance"], reverse=True)
    lines = [
        "| ID | Label | OCR text | Generic name | Edit distance | Needs review |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in disagreements[:limit]:
        lines.append(
            f"| `{row['id']}` | {row['medicine_name']} | {row.get('ocr_text') or ''} | "
            f"{row['generic_name']} | {row['edit_distance']} | {row.get('needs_human_review')} |"
        )
    if len(lines) == 2:
        lines.append("| n/a | No normalized OCR disagreements found. |  |  |  |  |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR/extraction metrics for a dataset split.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=["Testing", "Training", "Validation"],
        default="Testing",
        help="Dataset split to evaluate. Defaults to held-out Testing rows.",
    )
    parser.add_argument("--allow-mock", action="store_true", help="Allow reporting mock-label OCR rows.")
    parser.add_argument("--disagreements", type=int, default=10)
    parser.add_argument("--run-name", default="Unoptimized baseline", help="Run label to write in the metrics table.")
    parser.add_argument("--output", default=None, help="Markdown output path. Defaults to outputs/baseline_results.md.")
    args = parser.parse_args()

    rows = rows_for_evaluation(args.limit, args.split)
    if not rows:
        raise RuntimeError("No OCR rows found. Run src/02_ocr.py first.")

    models = {row.get("ocr_model") for row in rows}
    if "mock-labels" in models and not args.allow_mock:
        raise RuntimeError(
            "This table includes rows produced by --mock-from-labels. "
            "Run live OCR with src/02_ocr.py, or pass --allow-mock for smoke-test reporting only."
        )

    summary, evaluated = evaluate(rows, args.split, args.run_name)
    body = [
        "# OCR Evaluation Results",
        "",
        f"These metrics evaluate {args.run_name} on the {args.split} split.",
        "",
        summary,
        "",
        "## Largest OCR Disagreements",
        "",
        disagreement_table(evaluated, args.disagreements),
        "",
    ]
    markdown = "\n".join(body)
    output = args.output or str(ensure_output_dir() / "baseline_results.md")
    with open(output, "w", encoding="utf-8") as output_file:
        output_file.write(markdown)
    print(markdown)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
