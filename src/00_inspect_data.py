#!/usr/bin/env python
"""Inspect the held-out Testing split and export a few image previews.

This is the quick sanity-check stage: it verifies that label rows resolve to
real word-crop PNGs before anything is written to LanceDB.
"""

from __future__ import annotations

import argparse

from PIL import Image

from config import DEFAULT_PREVIEW_SAMPLES
from data import validate_testing_paths
from rendering import preview_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Testing labels and word images.")
    parser.add_argument("--limit", type=int, default=DEFAULT_PREVIEW_SAMPLES)
    args = parser.parse_args()

    labels = validate_testing_paths(args.limit)
    missing = labels.filter(~labels["exists"])
    print(f"Loaded {labels.height} Testing rows for inspection.")
    print(f"Missing image files: {missing.height}")

    out_dir = preview_dir("inspection")
    for i, row in enumerate(labels.iter_rows(named=True), start=1):
        image = Image.open(row["image_path"])
        preview_path = out_dir / f"{i:02d}_{row['IMAGE']}"
        image.save(preview_path)
        print(
            f"{i:02d}. {row['IMAGE']} size={image.size} "
            f"medicine={row['MEDICINE_NAME']!r} generic={row['GENERIC_NAME']!r} "
            f"preview={preview_path}"
        )


if __name__ == "__main__":
    main()
