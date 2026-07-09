from __future__ import annotations

from typing import Iterable

import lancedb
from lancedb.pydantic import LanceModel, Vector

from config import DB_PATH, SOURCE_DATASET, TABLE_NAME, VECTOR_DIM


class DoctorHandwriting(LanceModel):
    id: str
    split: str = "Testing"
    image_filename: str
    image: bytes
    medicine_name: str
    generic_name: str
    normalized_text: str | None = None
    category: str | None = None
    is_medical: bool | None = None
    needs_human_review: bool | None = None
    searchable_summary: str | None = None
    source_dataset: str = SOURCE_DATASET
    vector: Vector(VECTOR_DIM) | None = None


def connect_db():
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DB_PATH))


def table_exists() -> bool:
    db = connect_db()
    return TABLE_NAME in db.table_names()


def create_table(mode: str = "overwrite", table_name: str = TABLE_NAME):
    db = connect_db()
    return db.create_table(table_name, schema=DoctorHandwriting, mode=mode)


def open_table():
    db = connect_db()
    if TABLE_NAME not in db.table_names():
        raise RuntimeError(
            f"LanceDB table {TABLE_NAME!r} does not exist. Run src/01_ingest_images.py first."
        )
    return db.open_table(TABLE_NAME)


def bounded_rows(
    columns: list[str],
    *,
    limit: int,
    where: str | None = None,
) -> list[dict]:
    builder = open_table().search().select(columns).limit(limit)
    if where:
        builder = builder.where(where)
    return builder.to_list()


def merge_update(rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    table = open_table()
    table.merge_insert(on="id").when_matched_update_all().execute(rows)
    return len(rows)
