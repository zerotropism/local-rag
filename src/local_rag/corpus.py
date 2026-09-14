"""Loading a corpus into Documents. Nothing here is wine-specific."""

import csv
from collections.abc import Sequence
from pathlib import Path

from local_rag.models import Document


def load_csv(
    path: str | Path,
    text_field: str,
    required_fields: Sequence[str] = (),
) -> list[Document]:
    """Read a CSV into Documents.

    Args:
        path: CSV file with a header row.
        text_field: column whose value gets embedded.
        required_fields: columns that must be non-empty, or the row is skipped.
    Returns:
        One Document per usable row, the whole row kept as payload.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Corpus file '{csv_path}' not found.")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [f for f in (text_field, *required_fields) if f not in columns]
        if missing:
            raise KeyError(f"Columns {missing} are absent from '{csv_path}'. Found: {columns}")

        documents = []
        for row in reader:
            if any(not (row.get(field) or "").strip() for field in required_fields):
                continue
            text = (row.get(text_field) or "").strip()
            if not text:
                continue
            documents.append(Document(text=text, payload=row))

    if not documents:
        raise ValueError(f"No usable row in '{csv_path}'.")
    return documents
