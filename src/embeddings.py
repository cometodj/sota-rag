from __future__ import annotations

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(float).tolist()
