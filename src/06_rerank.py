#!/usr/bin/env python
"""Rerank LanceDB dense-search candidates with late interaction.

LanceDB supplies the fast top-k candidate set; a ColBERT-style MaxSim pass then
reranks those candidates using token-level query/document evidence.
"""

from __future__ import annotations

import argparse
from typing import cast

from lancedb.query import LanceVectorQueryBuilder

from config import LATE_INTERACTION_MODEL
from db import open_table
from embeddings import encode_texts, load_embedder
from late_interaction import LateInteractionReranker
from rendering import preview_dir, save_image_preview, slugify

DEFAULT_QUERIES = [
    "antibiotic for infection",
    "fever and pain medication",
    "blood pressure measurement",
    "medicine instruction twice daily",
    "prescription term related to headache",
]


def dense_candidates(query: str, candidate_limit: int, *, strict_embeddings: bool) -> list[dict]:
    model = load_embedder(allow_fallback=not strict_embeddings)
    query_vector = encode_texts(model, [query])[0]
    table = open_table()
    builder = cast(LanceVectorQueryBuilder, table.search(query_vector))
    return (
        builder.distance_type("cosine")
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
        .limit(candidate_limit)
        .to_list()
    )


def rerank(query: str, candidates: list[dict], reranker: LateInteractionReranker) -> list[dict]:
    reranked = []
    for dense_rank, row in enumerate(candidates, start=1):
        text = " ".join(
            str(row.get(field) or "")
            for field in ["searchable_summary", "ocr_text", "medicine_name", "generic_name"]
        )
        updated = dict(row)
        updated["dense_rank"] = dense_rank
        updated["late_interaction_score"] = reranker.score(query, text)
        reranked.append(updated)
    return sorted(reranked, key=lambda row: row["late_interaction_score"], reverse=True)


def print_comparison(query: str, dense_rows: list[dict], reranked_rows: list[dict], limit: int) -> None:
    out_dir = preview_dir("rerank", query)
    print(f"\nQuery: {query}")
    print("Dense top results:")
    for rank, row in enumerate(dense_rows[:limit], start=1):
        distance = row.get("_distance")
        score = "" if distance is None else f" distance={distance:.4f}"
        print(f"  {rank:02d}.{score} {row['id']} {row['medicine_name']} ({row['generic_name']})")

    print("Late-interaction reranked results:")
    for rank, row in enumerate(reranked_rows[:limit], start=1):
        preview = save_image_preview(row["image"], out_dir, f"{rank:02d}_{slugify(row['id'])}.png")
        print(
            f"  {rank:02d}. late={row['late_interaction_score']:.4f} "
            f"dense_rank={row['dense_rank']} id={row['id']} label={row['medicine_name']} "
            f"generic={row['generic_name']} ocr={row.get('ocr_text')!r} preview={preview}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank dense candidates with ColBERT-style late interaction.")
    parser.add_argument("queries", nargs="*", default=DEFAULT_QUERIES)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--late-interaction-model", default=LATE_INTERACTION_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--strict-embeddings", action="store_true")
    args = parser.parse_args()

    reranker = LateInteractionReranker(model_name=args.late_interaction_model, device=args.device)
    for query in args.queries:
        candidates = dense_candidates(query, args.candidate_limit, strict_embeddings=args.strict_embeddings)
        print_comparison(query, candidates, rerank(query, candidates, reranker), args.limit)


if __name__ == "__main__":
    main()
