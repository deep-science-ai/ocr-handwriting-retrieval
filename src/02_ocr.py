#!/usr/bin/env python
"""Run VLM handwriting OCR over images read back from LanceDB.

The model sees only the image bytes stored in the table. Results are written
back in LanceDB merge batches while OpenAI/DSPy calls run with bounded concurrency.
"""

from __future__ import annotations

import argparse
import asyncio
import io

from PIL import Image

from config import OPENAI_VISION_MODEL
from db import merge_update, open_table
from dspy_programs import make_handwriting_reader
from metrics import ocr_scores


def rows_for_ocr(limit: int | None, overwrite: bool, model_name: str) -> list[dict]:
    table = open_table()
    row_limit = limit or table.count_rows()
    query = table.search().select(["id", "image", "medicine_name", "ocr_model"]).limit(row_limit)
    if not overwrite:
        query = query.where(f"ocr_text IS NULL OR ocr_model IS NULL OR ocr_model != '{model_name}'")
    return query.to_list()


def flush_updates(updates: list[dict]) -> int:
    if not updates:
        return 0
    merge_update(updates)
    return len(updates)


async def transcribe_row(row: dict, *, dspy, reader, mock_from_labels: bool, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        if mock_from_labels:
            ocr_text = row["medicine_name"]
        else:
            image = Image.open(io.BytesIO(row["image"]))
            image.load()
            pred = await reader.acall(image=dspy.Image(image))
            ocr_text = pred.ocr_text.strip()

        update = {
            "id": row["id"],
            "ocr_text": ocr_text,
            "ocr_model": "mock-labels" if mock_from_labels else OPENAI_VISION_MODEL,
        }
        update.update(ocr_scores(ocr_text, row["medicine_name"]))
        return update


async def run_ocr(args: argparse.Namespace) -> None:
    ocr_model = "mock-labels" if args.mock_from_labels else OPENAI_VISION_MODEL
    rows = rows_for_ocr(args.limit, args.overwrite, ocr_model)
    if not rows:
        print("No rows need OCR.")
        return
    print(
        f"Processing {len(rows)} OCR rows with concurrency={args.concurrency}, batch_size={args.batch_size}.",
        flush=True,
    )

    if args.mock_from_labels:
        print("Using --mock-from-labels. Do not report this as model OCR quality.", flush=True)
        reader = None
        dspy = None
    else:
        dspy, reader = make_handwriting_reader()

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            transcribe_row(row, dspy=dspy, reader=reader, mock_from_labels=args.mock_from_labels, semaphore=semaphore)
        )
        for row in rows
    ]

    updates = []
    written = 0
    completed = 0
    exact = 0
    normalized = 0
    for task in asyncio.as_completed(tasks):
        update = await task
        completed += 1
        updates.append(update)
        exact += int(update["ocr_exact_match"])
        normalized += int(update["ocr_normalized_match"])
        if completed % args.progress_every == 0 or completed == len(rows):
            print(f"Completed OCR calls for {completed}/{len(rows)} rows.", flush=True)

        if len(updates) >= args.batch_size or completed == len(rows):
            flushed = flush_updates(updates)
            written += flushed
            updates.clear()
            print(f"Wrote OCR updates for {written}/{len(rows)} rows.", flush=True)

    print(f"Wrote OCR results for {written} rows.", flush=True)
    print(f"Exact match: {exact}/{written}; normalized match: {normalized}/{written}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSPy VLM OCR over images stored in LanceDB.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit. Defaults to all pending rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=25, help="Rows to merge into LanceDB per write.")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent DSPy/OpenAI OCR requests.")
    parser.add_argument("--progress-every", type=int, default=5, help="Print progress after this many OCR calls.")
    parser.add_argument(
        "--mock-from-labels",
        action="store_true",
        help="Offline smoke-test mode only: copies validation labels into ocr_text.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1.")
    if args.batch_size % args.concurrency != 0:
        parser.error("--batch-size must be a clean multiple of --concurrency.")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1.")
    asyncio.run(run_ocr(args))


if __name__ == "__main__":
    main()
