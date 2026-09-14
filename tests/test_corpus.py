"""Corpus loading: quoted multi-line fields, skipped rows, missing columns."""

import pytest

from local_rag.corpus import load_csv

CSV = """name,region,variety,notes
Shiraz 2004,"Barossa, Australia",Red Wine,"Classic vintage conditions.
Rainfall kept the vines in balance."
Cabernet 2005,"Napa, California",Red Wine,"Notes of blackcurrant."
No Variety,"Somewhere",,"Should be skipped."
No Notes,"Somewhere",Red Wine,""
"""


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "wines.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


def test_multiline_quoted_fields_are_read_whole(corpus) -> None:
    documents = load_csv(corpus, text_field="notes")
    assert "Rainfall kept the vines" in documents[0].text


def test_rows_missing_a_required_field_are_skipped(corpus) -> None:
    documents = load_csv(corpus, text_field="notes", required_fields=["variety"])
    assert [d.payload["name"] for d in documents] == ["Shiraz 2004", "Cabernet 2005"]


def test_rows_with_empty_text_are_skipped(corpus) -> None:
    documents = load_csv(corpus, text_field="notes")
    assert "No Notes" not in [d.payload["name"] for d in documents]


def test_the_whole_row_is_kept_as_payload(corpus) -> None:
    """A different corpus carries different columns: the loader must not filter them."""
    payload = load_csv(corpus, text_field="notes")[0].payload
    assert set(payload) == {"name", "region", "variety", "notes"}


def test_a_missing_column_is_reported(corpus) -> None:
    with pytest.raises(KeyError, match="absent"):
        load_csv(corpus, text_field="description")


def test_a_missing_file_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nope.csv", text_field="notes")


def test_a_single_field_is_prefixed_with_its_name(corpus) -> None:
    documents = load_csv(corpus, text_field="notes")
    assert documents[0].text.startswith("notes: ")


def test_several_fields_are_concatenated(corpus) -> None:
    """Embedding only the free-text column leaves metadata unsearchable."""
    documents = load_csv(corpus, text_field=["name", "region", "notes"])
    text = documents[0].text

    assert "name: Shiraz 2004" in text
    assert "region: Barossa, Australia" in text
    assert "Rainfall kept the vines" in text
