"""Building the configured pipeline: retrieval modes, optional dependencies, entry points."""

import sys
from importlib.metadata import entry_points

import pytest

from local_rag.cli import build_retriever, load_documents
from local_rag.config import load_config
from local_rag.embedders import SentenceTransformerEmbedder
from local_rag.lexical import BM25Index


def test_bm25_is_the_default_and_needs_no_model() -> None:
    config = load_config()
    assert config["retrieval"] == "lexical"
    retriever = build_retriever(config)
    assert isinstance(retriever, BM25Index)
    assert retriever.count() == len(load_documents(config))


def test_an_unknown_retrieval_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        build_retriever({**load_config(), "retrieval": "magic"})


def test_a_missing_sentence_transformers_says_how_to_install_it(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="uv sync --extra sentence-transformers"):
        _ = SentenceTransformerEmbedder().model


def test_console_scripts_resolve() -> None:
    expected = {"local-rag", "local-rag-eval", "local-rag-mcp"}
    scripts = [ep for ep in entry_points(group="console_scripts") if ep.name in expected]
    assert {ep.name for ep in scripts} == expected
    for ep in scripts:
        assert callable(ep.load())
