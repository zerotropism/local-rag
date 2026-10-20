"""Embedder selection and the Ollama embedder, against a fake Ollama client."""

from types import SimpleNamespace

import pytest

from local_rag.embedders import (
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    make_embedder,
)
from local_rag.protocols import Embedder


class FakeOllama:
    """Records each batch and answers one vector per text: [length of the text, 1.0]."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(self, model: str, input: list[str]):
        self.batches.append(list(input))
        return SimpleNamespace(embeddings=[[float(len(text)), 1.0] for text in input])


def test_encoders_are_named_backend_then_model() -> None:
    assert isinstance(make_embedder("ollama:nomic-embed-text:latest"), OllamaEmbedder)
    assert make_embedder("ollama:nomic-embed-text:latest").model == "nomic-embed-text:latest"
    assert isinstance(make_embedder("sentence-transformers:all-MiniLM-L6-v2"), Embedder)
    assert isinstance(
        make_embedder("sentence-transformers:all-MiniLM-L6-v2"), SentenceTransformerEmbedder
    )


@pytest.mark.parametrize("name", ["all-MiniLM-L6-v2", "openai:text-embedding-3", "ollama:"])
def test_unknown_encoders_are_refused(name) -> None:
    with pytest.raises(ValueError, match="backend:model|expected one of"):
        make_embedder(name)


def test_queries_and_documents_get_the_prefixes_the_model_expects() -> None:
    client = FakeOllama()
    embedder = OllamaEmbedder("nomic-embed-text", client=client)
    embedder.encode(["a red wine"])
    embedder.encode_query("which red?")
    assert client.batches == [["search_document: a red wine"], ["search_query: which red?"]]


def test_models_without_known_prefixes_get_none() -> None:
    client = FakeOllama()
    OllamaEmbedder("bge-m3", client=client).encode_query("which red?")
    assert client.batches == [["which red?"]]


def test_documents_are_sent_in_batches_and_keep_their_order() -> None:
    client = FakeOllama()
    vectors = OllamaEmbedder("bge-m3", batch_size=2, client=client).encode(["a", "bb", "ccc"])
    assert [len(batch) for batch in client.batches] == [2, 1]
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0]


def test_the_dimension_is_read_from_one_probe_then_kept() -> None:
    client = FakeOllama()
    embedder = OllamaEmbedder("bge-m3", client=client)
    assert embedder.dimension == 2
    assert embedder.dimension == 2
    assert len(client.batches) == 1
