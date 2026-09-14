"""Embedding backends."""

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Local embeddings. The model is loaded once, on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts)).tolist()
