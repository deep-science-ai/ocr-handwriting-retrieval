from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import LATE_INTERACTION_MODEL


@dataclass
class LateInteractionReranker:
    model_name: str = LATE_INTERACTION_MODEL
    device: str | None = None

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def score(self, query: str, document: str) -> float:
        query_vectors = self._token_vectors(query)
        document_vectors = self._token_vectors(document)
        if query_vectors.numel() == 0 or document_vectors.numel() == 0:
            return float("-inf")

        similarities = query_vectors @ document_vectors.T
        return similarities.max(dim=1).values.sum().item()

    @torch.no_grad()
    def _token_vectors(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        vectors = outputs.last_hidden_state[0]
        mask = encoded["attention_mask"][0].bool()

        special_ids = set(self.tokenizer.all_special_ids)
        if special_ids:
            input_ids = encoded["input_ids"][0]
            special_mask = torch.tensor(
                [token_id.item() in special_ids for token_id in input_ids],
                dtype=torch.bool,
                device=self.device,
            )
            mask = mask & ~special_mask

        vectors = vectors[mask]
        return F.normalize(vectors, p=2, dim=1)
