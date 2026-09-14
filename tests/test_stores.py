"""Store behaviour, with a toy embedder: no model download, no torch."""

from collections.abc import Sequence

import pytest

from local_rag.models import Document
from local_rag.protocols import Embedder, VectorStore
from local_rag.stores import QdrantStore

DOCUMENTS = [
    Document(text="crisp white wine from Burgundy", payload={"name": "Chablis"}),
    Document(text="bold red shiraz from Barossa", payload={"name": "Shiraz"}),
    Document(text="sparkling wine from Champagne", payload={"name": "Champagne"}),
]


class ToyEmbedder:
    """Deterministic bag-of-words vectors, stable across runs."""

    dimension = 16

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for word in text.lower().split():
                vector[sum(map(ord, word)) % self.dimension] += 1.0
            vectors.append(vector)
        return vectors


@pytest.fixture
def store() -> QdrantStore:
    store = QdrantStore(ToyEmbedder(), collection_name="test")
    store.index(DOCUMENTS)
    return store


def test_implementations_satisfy_the_protocols(store) -> None:
    assert isinstance(ToyEmbedder(), Embedder)
    assert isinstance(store, VectorStore)


def test_indexing_reports_the_stored_count(store) -> None:
    assert store.count() == 3


def test_search_returns_the_closest_document_first(store) -> None:
    """The regression this replaces: qdrant-client 1.19 dropped .search, and the
    decorator turned the resulting AttributeError into a silent empty result."""
    hits = store.search("bold red shiraz from Barossa", limit=2)
    assert hits[0].payload["name"] == "Shiraz"


def test_search_respects_the_limit(store) -> None:
    assert len(store.search("wine", limit=1)) == 1


def test_payload_is_returned_verbatim(store) -> None:
    """The store must not know which fields a corpus carries."""
    assert set(store.search("wine", limit=1)[0].payload) == {"name"}
