from __future__ import annotations

import io
import re
from pathlib import Path

from PIL import Image

from config import ensure_output_dir


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "query"


def save_image_preview(image_bytes: bytes, output_dir: Path, filename: str) -> Path:
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    path = output_dir / filename
    image.save(path)
    return path


def preview_dir(stage: str, query: str | None = None) -> Path:
    return ensure_output_dir(stage, *( [slugify(query)] if query else [] ))
