"""BM25 over the same documents as the vector store: exact words, names and rare terms."""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from local_rag.models import Document, SearchHit

K1 = 1.5
B = 0.75


def tokens(text: str) -> list[str]:
    """Lowercase words without accents: 'Château' and 'chateau' are the same token."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", plain)


class BM25Index:
    """Okapi BM25. In memory: indexing 1,300 documents takes a fraction of a second."""

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._bags: list[Counter[str]] = []
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 0.0

    def index(self, documents: Sequence[Document]) -> int:
        self._documents = list(documents)
        self._bags = [Counter(tokens(document.text)) for document in self._documents]
        self._document_frequency = Counter(term for bag in self._bags for term in bag)
        lengths = [sum(bag.values()) for bag in self._bags]
        self._average_length = sum(lengths) / len(lengths) if lengths else 0.0
        return self.count()

    def count(self) -> int:
        return len(self._documents)

    def _idf(self, term: str) -> float:
        n = self._document_frequency[term]
        return math.log(1 + (self.count() - n + 0.5) / (n + 0.5))

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        terms = set(tokens(query))
        scored = []
        for position, bag in enumerate(self._bags):
            matched = terms & bag.keys()
            if not matched:
                continue
            norm = K1 * (1 - B + B * sum(bag.values()) / self._average_length)
            score = sum(
                self._idf(term) * bag[term] * (K1 + 1) / (bag[term] + norm) for term in matched
            )
            scored.append((score, position))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(score=score, payload=self._documents[position].payload, id=position)
            for score, position in scored[:limit]
        ]
