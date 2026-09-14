"""What the pipeline moves around: documents in, hits out."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """One indexable item: the text that gets embedded, plus whatever metadata goes with it."""

    text: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHit:
    score: float
    payload: dict[str, Any]
