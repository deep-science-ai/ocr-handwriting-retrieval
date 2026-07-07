from __future__ import annotations

import re


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def edit_distance(a: str | None, b: str | None) -> int:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm:
        return len(b_norm)
    if not b_norm:
        return len(a_norm)

    prev = list(range(len(b_norm) + 1))
    for i, ca in enumerate(a_norm, start=1):
        curr = [i]
        for j, cb in enumerate(b_norm, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def ocr_scores(ocr_text: str | None, medicine_name: str) -> dict:
    normalized_match = normalize_text(ocr_text) == normalize_text(medicine_name)
    distance = edit_distance(ocr_text, medicine_name)
    return {
        "ocr_exact_match": (ocr_text or "").strip() == medicine_name.strip(),
        "ocr_normalized_match": normalized_match,
        "ocr_edit_distance": distance,
        "needs_human_review": not normalized_match and distance > 1,
    }
