from __future__ import annotations

import numpy as np

from .config import DEFAULT_EMBED_MODEL, EMBED_BATCH_SIZE


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

    def search(
        self,
        query: str,
        corpus_ids: list[int],
        corpus_vectors: np.ndarray,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """Return (prompt_id, similarity_score) pairs, highest first."""
        query_vec = self.encode([query])[0]
        # Vectors are already L2-normalized, so dot product = cosine similarity
        scores = corpus_vectors @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(corpus_ids[i], float(scores[i])) for i in top_indices]
