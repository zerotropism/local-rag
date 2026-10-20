"""Embedding backends, chosen by a "backend:model" name such as "ollama:nomic-embed-text"."""

from collections.abc import Sequence

import ollama

BACKENDS = ("sentence-transformers", "ollama")

# Retrieval models trained with an instruction in front of queries, of documents, or both;
# without it they rank noticeably worse. From each model's card.
PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
}


class SentenceTransformerEmbedder:
    """Local embeddings through sentence-transformers. The model is loaded once, on first use.

    sentence-transformers pulls in torch: it is an optional extra, imported only here.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed: "
                    "uv sync --extra sentence-transformers, or use an ollama: encoder"
                ) from None
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self.model.encode(list(texts)).tolist()

    def encode_query(self, text: str) -> list[float]:
        return self.encode([text])[0]


class OllamaEmbedder:
    """Embeddings from a local Ollama server, sent in batches."""

    def __init__(self, model: str, batch_size: int = 64, client: ollama.Client | None = None):
        self.model = model
        self.batch_size = batch_size
        self.query_prefix, self.document_prefix = PREFIXES.get(model.split(":")[0], ("", ""))
        self._client = client or ollama.Client()
        self._dimension: int | None = None

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            list(vector) for vector in self._client.embed(model=self.model, input=texts).embeddings
        ]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self._embed(["dimension probe"])[0])
        return self._dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        marked = [self.document_prefix + text for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(marked), self.batch_size):
            vectors.extend(self._embed(marked[start : start + self.batch_size]))
        return vectors

    def encode_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]


def make_embedder(name: str) -> SentenceTransformerEmbedder | OllamaEmbedder:
    """Build an embedder from "backend:model"; the model part may itself contain a colon."""
    backend, separator, model = name.partition(":")
    if not separator or not model or backend not in BACKENDS:
        raise ValueError(f"Encoder {name!r}: expected one of {BACKENDS} then ':model'.")
    if backend == "ollama":
        return OllamaEmbedder(model)
    return SentenceTransformerEmbedder(model)
