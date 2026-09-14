"""Smoke test: the package modules import cleanly."""

import pytest

MODULES = [
    "models",
    "protocols",
    "embedders",
    "stores",
    "corpus",
    "config",
    "chatbot",
    "cli",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    __import__(f"local_rag.{name}")
