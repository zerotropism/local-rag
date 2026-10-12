"""Retrieval metrics checked by hand, the harness on a scripted retriever, the CLI on BM25."""

import json

import pytest

from local_rag.config import resolve
from local_rag.evaluation import (
    EvalItem,
    evaluate,
    load_eval,
    main,
    markdown_table,
    ranked_ids,
    recall_at,
    reciprocal_rank,
    summarise,
)
from local_rag.models import SearchHit


def test_recall_and_reciprocal_rank_by_hand() -> None:
    ranked = ["a", "b", "c", "d"]
    assert recall_at(ranked, ["b", "z"], 2) == 0.5
    assert recall_at(ranked, ["b", "z"], 1) == 0.0
    assert reciprocal_rank(ranked, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ["z"]) == 0.0


def test_duplicate_rows_of_a_document_count_once() -> None:
    hits = [SearchHit(1.0, {"name": n}) for n in ("Krug", "Krug", "Margaux")]
    assert ranked_ids(hits, "name") == ["Krug", "Margaux"]


def test_an_item_without_relevant_documents_is_rejected(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps({"question": "q", "relevant": []}) + "\n")
    with pytest.raises(ValueError, match="no relevant document"):
        load_eval(path)


class Scripted:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        return [SearchHit(1.0, {"name": n}) for n in self.names[:limit]]


def test_results_are_summarised_per_kind_and_overall() -> None:
    items = [
        EvalItem("q1", ["a"], "name"),
        EvalItem("q2", ["b"], "paraphrase"),
    ]
    outcomes = evaluate({"scripted": Scripted(["a", "x", "b"])}, items, "name")
    rows = {s.kind: s for s in summarise(outcomes)}
    assert rows["name"].mrr == 1.0
    assert rows["paraphrase"].mrr == pytest.approx(0.333)
    assert rows["all"].questions == 2
    assert "| scripted | all | 2 |" in markdown_table(list(rows.values()))


def test_the_cli_measures_bm25_on_the_bundled_data(tmp_path, capsys) -> None:
    """BM25 needs no model: the command runs on the wine corpus and the bundled questions."""
    main(["--retrievers", "lexical", "--output", str(tmp_path)])
    outcomes = next(tmp_path.glob("outcomes-*.jsonl")).read_text().splitlines()
    assert len(outcomes) == len(load_eval(resolve("data/eval.jsonl")))
    table = capsys.readouterr().out
    assert "| lexical | all |" in table
    # the paraphrases share no content word with their wine: BM25 cannot find them
    assert "| lexical | paraphrase | 10 | 0.000 |" in table
