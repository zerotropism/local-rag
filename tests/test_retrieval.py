"""BM25 and rank fusion, on documents small enough to check by hand."""

import pytest

from local_rag.hybrid import HybridRetriever
from local_rag.lexical import BM25Index, tokens
from local_rag.models import Document, SearchHit
from local_rag.protocols import Retriever, VectorStore

DOCUMENTS = [
    Document(text="Chablis: crisp white wine from Burgundy", payload={"name": "Chablis"}),
    Document(text="Shiraz: bold red from Barossa", payload={"name": "Shiraz"}),
    Document(text="Château Margaux: red Bordeaux first growth", payload={"name": "Margaux"}),
]


@pytest.fixture
def lexical() -> BM25Index:
    index = BM25Index()
    index.index(DOCUMENTS)
    return index


def test_tokens_ignore_case_and_accents() -> None:
    assert tokens("Château Côte-Rôtie 2010") == ["chateau", "cote", "rotie", "2010"]


def test_bm25_satisfies_the_protocols(lexical) -> None:
    assert isinstance(lexical, Retriever)
    assert isinstance(lexical, VectorStore)
    assert lexical.count() == 3


def test_bm25_finds_exact_words_and_names(lexical) -> None:
    assert lexical.search("chateau margaux", limit=1)[0].payload["name"] == "Margaux"
    assert lexical.search("Barossa", limit=1)[0].payload["name"] == "Shiraz"


def test_bm25_returns_nothing_without_a_shared_word(lexical) -> None:
    assert lexical.search("sparkling", limit=3) == []


def test_bm25_hits_carry_the_document_position(lexical) -> None:
    assert lexical.search("Barossa", limit=1)[0].id == 1


class Fixed:
    """A retriever that always answers the same ids, in order."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        return [SearchHit(score=1.0, payload={"id": i}, id=i) for i in self.ids[:limit]]


def test_rank_fusion_favours_documents_both_retrievers_rank_well() -> None:
    hybrid = HybridRetriever([Fixed([1, 2, 3]), Fixed([2, 3, 1])])
    assert [hit.id for hit in hybrid.search("q", limit=3)] == [2, 1, 3]


def test_rank_fusion_keeps_documents_only_one_retriever_found() -> None:
    hybrid = HybridRetriever([Fixed([1]), Fixed([2])])
    assert {hit.id for hit in hybrid.search("q", limit=5)} == {1, 2}


def test_rank_fusion_needs_ids() -> None:
    class Anonymous:
        def search(self, query, limit=3):
            return [SearchHit(score=1.0, payload={})]

    with pytest.raises(ValueError, match="without an id"):
        HybridRetriever([Anonymous()]).search("q")
