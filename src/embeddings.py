from __future__ import annotations

import hashlib
import math
from typing import Iterable

from config import EMBEDDING_MODEL, VECTOR_DIM


class HashingEmbedder:
    name = "local-hashing-fallback"

    def encode(self, texts: str | Iterable[str], normalize_embeddings: bool = True):
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        vectors = [self._encode_one(text, normalize_embeddings) for text in values]
        return vectors[0] if single else vectors

    def _encode_one(self, text: str, normalize_embeddings: bool) -> list[float]:
        vector = [0.0] * VECTOR_DIM
        tokens = (text or "").lower().split()
        if not tokens:
            tokens = [""]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            for offset, byte in enumerate(digest):
                idx = (byte + offset * 257) % VECTOR_DIM
                vector[idx] += 1.0 if byte % 2 == 0 else -1.0
        if normalize_embeddings:
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vector = [v / norm for v in vector]
        return vector


def load_embedder(*, allow_fallback: bool = True):
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        model.name = EMBEDDING_MODEL
        return model
    except Exception as exc:
        if not allow_fallback:
            raise
        print(f"Could not load {EMBEDDING_MODEL!r}: {exc}")
        print("Using deterministic local hashing embeddings for this run.")
        return HashingEmbedder()


def encode_texts(model, texts: list[str]) -> list[list[float]]:
    vectors = model.encode(texts, normalize_embeddings=True)
    if hasattr(vectors, "tolist"):
        return vectors.tolist()
    return [list(v) for v in vectors]
