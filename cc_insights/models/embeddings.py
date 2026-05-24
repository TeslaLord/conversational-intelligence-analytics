"""Sentence-transformers embedding wrapper."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False):
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ).astype("float32")
