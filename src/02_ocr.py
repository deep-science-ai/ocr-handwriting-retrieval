#!/usr/bin/env python
"""Run VLM handwriting OCR over images read back from LanceDB.

The model sees only the image bytes stored in the table. Results are written
back in LanceDB merge batches after the async DSPy calls finish.
"""

from __future__ import annotations

import argparse
import asyncio
import io
from pathlib import Path
import time

import pyarrow as pa
from PIL import Image

from config import VISION_MODEL
from db import merge_update, open_table
from dspy_programs import make_handwriting_reader
from metrics import ocr_scores


OCR_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("ocr_text", pa.string()),
        pa.field("ocr_model", pa.string()),
        pa.field("ocr_exact_match", pa.bool_()),
        pa.field("ocr_normalized_match", pa.bool_()),
        pa.field("ocr_edit_distance", pa.int64()),
    ]
)


def ensure_ocr_columns():
    table = open_table()
    existing = set(table.schema.names)
    missing = [field for field in OCR_OUTPUT_SCHEMA if field.name not in existing]
    if missing:
        table.add_columns(pa.schema(missing))
        table = open_table()
    return table


def rows_for_ocr(limit: int | None, overwrite: bool, model_name: str, split: str) -> list[dict]:
    table = ensure_ocr_columns()
    row_limit = limit or table.count_rows()
    query = (
        table.search()
        .select(["id", "split", "image", "medicine_name", "ocr_model"])
        .where(f"split = '{split}'")
        .limit(row_limit)
    )
    if not overwrite:
        query = query.where(
            f"split = '{split}' AND (ocr_text IS NULL OR ocr_model IS NULL OR ocr_model != '{model_name}')"
        )
    return query.to_list()


def flush_updates(updates: list[dict]) -> int:
    if not updates:
        return 0
    merge_update(updates)
    return len(updates)


async def transcribe_row(row: dict, *, dspy, reader, mock_from_labels: bool, ocr_model: str) -> dict:
    if mock_from_labels:
        ocr_text = row["medicine_name"]
    else:
        image = Image.open(io.BytesIO(row["image"]))
        image.load()
        pred = await reader.aforward(image=dspy.Image(image))
        ocr_text = pred.ocr_text.strip()

    update = {
        "id": row["id"],
        "ocr_text": ocr_text,
        "ocr_model": ocr_model,
    }
    update.update(ocr_scores(ocr_text, row["medicine_name"]))
    return update


async def transcribe_rows(rows: list[dict], *, mock_from_labels: bool, program_path: str | None, ocr_model: str) -> list[dict]:
    if mock_from_labels:
        print("Using --mock-from-labels. Do not report this as model OCR quality.", flush=True)
        dspy = None
        reader = None
    else:
        dspy, reader = make_handwriting_reader(program_path=program_path)

    tasks = [
        transcribe_row(row, dspy=dspy, reader=reader, mock_from_labels=mock_from_labels, ocr_model=ocr_model)
        for row in rows
    ]
    return await asyncio.gather(*tasks)


def model_label(mock_from_labels: bool, program_path: str | None) -> str:
    if mock_from_labels:
        return "mock-labels"
    if program_path:
        return f"{VISION_MODEL}+{Path(program_path).stem}"
    return VISION_MODEL


def write_updates(updates: list[dict], batch_size: int) -> int:
    written = 0
    for start in range(0, len(updates), batch_size):
        batch = updates[start : start + batch_size]
        written += flush_updates(batch)
        print(f"Wrote OCR updates for {written}/{len(updates)} rows.", flush=True)
    return written


async def run_ocr(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    ocr_model = model_label(args.mock_from_labels, args.program_path)
    rows = rows_for_ocr(args.limit, args.overwrite, ocr_model, args.split)
    if not rows:
        print("No rows need OCR.")
        print("END_TO_END_SECONDS=0.00")
        return
    print(
        f"Processing {len(rows)} {args.split} OCR rows with async DSPy calls, batch_size={args.batch_size}.",
        flush=True,
    )

    updates = await transcribe_rows(
        rows,
        mock_from_labels=args.mock_from_labels,
        program_path=args.program_path,
        ocr_model=ocr_model,
    )
    written = write_updates(updates, args.batch_size)
    exact = sum(int(update["ocr_exact_match"]) for update in updates)
    normalized = sum(int(update["ocr_normalized_match"]) for update in updates)
    print(f"Wrote OCR results for {written} rows.", flush=True)
    print(f"Exact match: {exact}/{written}; normalized match: {normalized}/{written}.", flush=True)
    elapsed = time.perf_counter() - started
    print(f"END_TO_END_SECONDS={elapsed:.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSPy VLM OCR over images stored in LanceDB.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit. Defaults to all pending rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--split",
        choices=["Testing", "Training", "Validation"],
        default="Testing",
        help="Dataset split to OCR. Defaults to held-out Testing rows.",
    )
    parser.add_argument("--batch-size", type=int, default=40, help="Rows to merge into LanceDB per write.")
    parser.add_argument(
        "--program-path",
        default=None,
        help="Optional DSPy program state produced by src/08_optimize_ocr.py.",
    )
    parser.add_argument(
        "--mock-from-labels",
        action="store_true",
        help="Offline smoke-test mode only: copies validation labels into ocr_text.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    asyncio.run(run_ocr(args))


if __name__ == "__main__":
    main()
