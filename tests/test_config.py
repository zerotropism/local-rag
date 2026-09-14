"""The shipped configuration must stay loadable from any working directory."""

import pytest

from local_rag.config import DEFAULT_CONFIG, load_config, resolve


def test_the_shipped_config_loads() -> None:
    config = load_config()
    assert {"data", "vectordb", "chatbot"} <= set(config)


def test_the_configured_corpus_exists() -> None:
    assert resolve(load_config()["data"]["path"]).is_file()


def test_relative_paths_resolve_from_the_project_root() -> None:
    assert resolve("data/x.csv") == DEFAULT_CONFIG.parent / "data/x.csv"


def test_a_missing_config_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
