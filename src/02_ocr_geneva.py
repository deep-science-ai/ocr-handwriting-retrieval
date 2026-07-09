#!/usr/bin/env python
"""Run OCR as a local Geneva backfill on the LanceDB table."""

from __future__ import annotations

import argparse
import math
import os
import time

import geneva
from geneva.table import Table

from config import DB_PATH, TABLE_NAME, VISION_MODEL, load_local_env
from dspy_programs import api_key_env_for_model
from udf import HandwritingOCR

OCR_COLUMN = "ocr_text"
MIN_TASK_SIZE = 128


def split_where(split: str) -> str:
    return f"split = '{split}'"


def build_ocr_udf() -> HandwritingOCR:
    # Fail fast in the driver process if the model API key is missing, instead
    # of discovering it only after Geneva starts worker processes.
    api_key_env = api_key_env_for_model(VISION_MODEL)
    if api_key_env and not os.environ.get(api_key_env):
        raise RuntimeError(f"{api_key_env} is required for Geneva OCR with {VISION_MODEL}.")
    return HandwritingOCR()


def configure_ocr_column(tbl: Table, column: str, ocr_udf: HandwritingOCR) -> bool:
    # Register the UDF as a computed column. If the column already exists, alter
    # it so this run uses the current UDF implementation.
    column_exists = column in tbl.schema.names
    if column_exists:
        tbl.alter_columns({"path": column, "udf": ocr_udf})
    else:
        tbl.add_columns({column: ocr_udf})
    return column_exists


def verify_outputs(tbl: Table, *, column: str, split: str, expected: int) -> int:
    # Keep the sample honest: every row in the selected split should have a
    # non-empty OCR output after the backfill completes.
    rows = tbl.search().select(["id", column]).where(split_where(split)).limit(expected).to_list()
    non_empty = sum(bool((row.get(column) or "").strip()) for row in rows)
    if non_empty != expected:
        raise RuntimeError(f"Expected {expected} non-empty {column!r} values for {split}, found {non_empty}.")
    return non_empty


def default_task_size(total_rows: int, concurrency: int) -> int:
    # Local runs benefit from a few more tasks than workers so work can be
    # distributed, but tiny task sizes add overhead. Keep the default coarse.
    target_chunks = max(1, concurrency * 2)
    return max(MIN_TASK_SIZE, math.ceil(total_rows / target_chunks))


def main() -> None:
    load_local_env()
    # These flags make the local benchmark reproducible without changing the
    # core Geneva flow: connect, register a UDF column, and backfill it.
    parser = argparse.ArgumentParser(description="Run OCR as a local Geneva backfill over LanceDB rows.")
    parser.add_argument(
        "--db-path",
        default=str(DB_PATH),
        help=f"Local LanceDB path. Defaults to {str(DB_PATH)!r}.",
    )
    parser.add_argument(
        "--table-name",
        default=TABLE_NAME,
        help=f"Table name. Defaults to {TABLE_NAME!r}.",
    )
    parser.add_argument(
        "--column",
        default=OCR_COLUMN,
        help=f"OCR column to create/backfill. Defaults to {OCR_COLUMN!r}.",
    )
    parser.add_argument(
        "--split",
        choices=["Testing", "Training", "Validation"],
        default="Testing",
        help="Dataset split to OCR. Defaults to held-out Testing rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute the target column for every row in the selected split.",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Geneva backfill worker concurrency.")
    parser.add_argument(
        "--intra-applier-concurrency",
        type=int,
        default=1,
        help="Threads per Geneva worker. Defaults to one model call per worker process.",
    )
    parser.add_argument(
        "--task-size",
        type=int,
        default=None,
        help=(
            f"Rows per Geneva read task. Defaults to at least {MIN_TASK_SIZE} rows, "
            "with enough chunks to distribute the selected split across workers."
        ),
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip table optimization/compaction after the backfill completes.",
    )
    parser.add_argument(
        "--refresh-status-secs",
        type=float,
        default=15.0,
        help="Seconds between Geneva progress refreshes.",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1.")
    if args.intra_applier_concurrency < 1:
        parser.error("--intra-applier-concurrency must be at least 1.")
    if args.task_size is not None and args.task_size < 1:
        parser.error("--task-size must be at least 1.")

    # Open the local LanceDB table and scope the run to one dataset split. The
    # default keeps the demo pointed at the held-out Testing rows.
    db = geneva.connect(args.db_path)
    tbl = db.open_table(args.table_name)
    total_rows = tbl.count_rows(split_where(args.split))
    if total_rows == 0:
        raise RuntimeError(f"No rows found for split {args.split!r} in table {args.table_name!r}.")
    task_size = args.task_size or default_task_size(total_rows, args.concurrency)

    # Attach the OCR UDF to the target column. Geneva will materialize this
    # column during backfill by calling the UDF on each selected row.
    ocr_udf = build_ocr_udf()
    column_existed = configure_ocr_column(tbl, args.column, ocr_udf)
    tbl = db.open_table(args.table_name)

    # Overwrite recomputes the selected split. Without it, existing values are
    # left alone and only missing outputs are filled in.
    if args.overwrite or not column_existed:
        where = split_where(args.split)
    else:
        where = f"{split_where(args.split)} AND {args.column} IS NULL"

    started = time.perf_counter()
    print(
        f"Starting local Geneva backfill: table={args.table_name} split={args.split} "
        f"rows={total_rows} column={args.column} concurrency={args.concurrency} "
        f"intra_applier_concurrency={args.intra_applier_concurrency} task_size={task_size} "
        f"overwrite={args.overwrite}",
        flush=True,
    )
    # `local_ray_context()` is used here: it starts a local execution context so
    # the UDF backfill pattern can run on a laptop. For LanceDB Enterprise, connect
    # to your Enterprise deployment and follow the Geneva quickstart:
    # https://docs.lancedb.com/geneva/getting-started#quickstart
    with db.local_ray_context():
        result = tbl.backfill(
            args.column,
            udf=ocr_udf,
            where=where,
            concurrency=args.concurrency,
            intra_applier_concurrency=args.intra_applier_concurrency,
            task_size=task_size,
            refresh_status_secs=args.refresh_status_secs,
        )
    backfill_elapsed = time.perf_counter() - started

    optimize_elapsed = 0.0
    tbl = db.open_table(args.table_name)
    if not args.skip_optimize:
        # The backfill writes a new table version. Optimize afterwards so the
        # local table is compacted again before the next pipeline stage.
        optimize_started = time.perf_counter()
        tbl.optimize()
        optimize_elapsed = time.perf_counter() - optimize_started
        tbl = db.open_table(args.table_name)

    # Report backfill and compaction separately: the first is model-call
    # throughput, while the second is storage maintenance.
    non_empty = verify_outputs(tbl, column=args.column, split=args.split, expected=total_rows)
    total_elapsed = time.perf_counter() - started

    print(f"Geneva backfill complete: job_id={result.job_id}")
    print(f"Verified non-empty {args.column!r} values for {non_empty}/{total_rows} {args.split} rows.")
    print(f"BACKFILL_SECONDS={backfill_elapsed:.2f}")
    print(f"OPTIMIZE_SECONDS={optimize_elapsed:.2f}")
    print(f"END_TO_END_SECONDS={total_elapsed:.2f}")


if __name__ == "__main__":
    main()
