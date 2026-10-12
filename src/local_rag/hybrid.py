"""Reciprocal rank fusion: each retriever votes with the rank it gives a document, not its score.

Scores from BM25 and from cosine similarity are on unrelated scales; ranks are comparable.
"""

from collections.abc import Sequence

from local_rag.models import SearchHit
from local_rag.protocols import Retriever

# The constant from Cormack et al. (2009): it damps the weight of the very first ranks
RRF_K = 60
# How deep each retriever is read before fusing
DEPTH = 50


class HybridRetriever:
    """Fuses retrievers that indexed the same documents in the same order (hits share ids)."""

    def __init__(self, retrievers: Sequence[Retriever], k: int = RRF_K, depth: int = DEPTH):
        self.retrievers = list(retrievers)
        self.k = k
        self.depth = depth

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        fused: dict[int, float] = {}
        payloads: dict[int, dict] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, limit=max(limit, self.depth)), 1):
                if hit.id is None:
                    raise ValueError(f"{type(retriever).__name__} returned a hit without an id")
                fused[hit.id] = fused.get(hit.id, 0.0) + 1 / (self.k + rank)
                payloads[hit.id] = hit.payload
        best = sorted(fused, key=lambda key: fused[key], reverse=True)[:limit]
        return [SearchHit(score=fused[key], payload=payloads[key], id=key) for key in best]
