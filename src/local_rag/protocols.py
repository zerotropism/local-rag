"""The two swappable pieces of a RAG pipeline."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from local_rag.models import Document, SearchHit


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Swap sentence-transformers for an API without touching the store."""

    @property
    def dimension(self) -> int:
        """Vector size, needed to declare the collection."""
        ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch. A single text is a batch of one."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Holds vectors and answers queries. Swap Qdrant for another backend."""

    def index(self, documents: Sequence[Document]) -> int:
        """Index documents, returning how many are stored afterwards."""
        ...

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        """Return the closest documents, best first."""
        ...

    def count(self) -> int:
        """How many points the collection holds."""
        ...
