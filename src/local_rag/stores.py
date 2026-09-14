"""Qdrant-backed vector store."""

from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from local_rag.models import Document, SearchHit
from local_rag.protocols import Embedder


class QdrantStore:
    """Knows nothing about the corpus: documents carry their own text and payload."""

    def __init__(
        self,
        embedder: Embedder,
        location: str = ":memory:",
        collection_name: str = "documents",
    ) -> None:
        self._embedder = embedder
        self._collection_name = collection_name
        self._client = QdrantClient(location)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection_name):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=self._embedder.dimension, distance=models.Distance.COSINE
            ),
        )

    def index(self, documents: Sequence[Document]) -> int:
        vectors = self._embedder.encode([document.text for document in documents])
        self._client.upload_points(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(id=index, vector=vector, payload=document.payload)
                for index, (document, vector) in enumerate(zip(documents, vectors, strict=True))
            ],
            wait=True,
        )
        return self.count()

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        """Uses query_points: the search method was removed in qdrant-client 1.19."""
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=self._embedder.encode([query])[0],
            limit=limit,
        )
        return [SearchHit(score=float(p.score), payload=p.payload or {}) for p in response.points]

    def count(self) -> int:
        return self._client.get_collection(self._collection_name).points_count or 0
