"""Smoke tests: every module in code/ must import cleanly."""

import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE))

MODULES = ["chatbot", "decorators", "main", "vectordb"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    __import__(name)
