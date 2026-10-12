"""`local-rag-eval`: how often each retriever puts the expected documents first.

Each line of the evaluation file is a question, the ids of the documents that answer it, and a
kind, so results can be read per kind of question. The id is a payload field (`id_field`), so
the file survives a reordering of the corpus.
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from local_rag.config import load_config, resolve
from local_rag.hybrid import HybridRetriever
from local_rag.lexical import BM25Index
from local_rag.models import SearchHit
from local_rag.protocols import Retriever

RETRIEVERS = ("lexical", "dense", "hybrid")
DEPTH = 10


@dataclass(frozen=True)
class EvalItem:
    question: str
    relevant: list[str]
    kind: str = "all"


def load_eval(path: str | Path) -> list[EvalItem]:
    items = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = EvalItem(**json.loads(line))
        if not item.relevant:
            raise ValueError(f"line {number}: no relevant document for {item.question!r}")
        items.append(item)
    return items


def ranked_ids(hits: Sequence[SearchHit], id_field: str) -> list[str]:
    """Hit ids in rank order, each once: duplicate rows of one document count once."""
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(str(hit.payload.get(id_field)), None)
    return list(seen)


def recall_at(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Share of the relevant documents found in the first k."""
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first relevant document, 0 when none is retrieved."""
    for position, name in enumerate(ranked, 1):
        if name in relevant:
            return 1 / position
    return 0.0


@dataclass(frozen=True)
class Outcome:
    retriever: str
    kind: str
    question: str
    ranked: list[str]
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    milliseconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(
    retrievers: dict[str, Retriever],
    items: Sequence[EvalItem],
    id_field: str,
    report: Callable[[Outcome], None] = lambda outcome: None,
) -> list[Outcome]:
    outcomes = []
    for name, retriever in retrievers.items():
        for item in items:
            started = time.perf_counter()
            # read past DEPTH: duplicate rows of one document collapse into one id
            hits = retriever.search(item.question, limit=2 * DEPTH)
            elapsed = (time.perf_counter() - started) * 1000
            ranked = ranked_ids(hits, id_field)[:DEPTH]
            outcome = Outcome(
                retriever=name,
                kind=item.kind,
                question=item.question,
                ranked=ranked,
                recall_at_1=recall_at(ranked, item.relevant, 1),
                recall_at_5=recall_at(ranked, item.relevant, 5),
                recall_at_10=recall_at(ranked, item.relevant, 10),
                reciprocal_rank=reciprocal_rank(ranked, item.relevant),
                milliseconds=round(elapsed, 1),
            )
            report(outcome)
            outcomes.append(outcome)
    return outcomes


@dataclass(frozen=True)
class Summary:
    retriever: str
    kind: str
    questions: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    milliseconds: float


def summarise(outcomes: Sequence[Outcome]) -> list[Summary]:
    """One row per retriever and kind of question, then one row per retriever for all kinds."""
    groups: dict[tuple[str, str], list[Outcome]] = {}
    for outcome in outcomes:
        groups.setdefault((outcome.retriever, outcome.kind), []).append(outcome)
        groups.setdefault((outcome.retriever, "all"), []).append(outcome)

    def row(key: tuple[str, str], group: list[Outcome]) -> Summary:
        return Summary(
            retriever=key[0],
            kind=key[1],
            questions=len(group),
            recall_at_1=round(statistics.mean(o.recall_at_1 for o in group), 3),
            recall_at_5=round(statistics.mean(o.recall_at_5 for o in group), 3),
            recall_at_10=round(statistics.mean(o.recall_at_10 for o in group), 3),
            mrr=round(statistics.mean(o.reciprocal_rank for o in group), 3),
            milliseconds=round(statistics.median(o.milliseconds for o in group), 1),
        )

    return [row(key, group) for key, group in groups.items()]


def markdown_table(summaries: Sequence[Summary]) -> str:
    lines = [
        "| Retriever | Questions | n | Recall@1 | Recall@5 | Recall@10 | MRR | ms / query |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(summaries, key=lambda s: (s.retriever, s.kind != "all", s.kind))
    for s in ordered:
        lines.append(
            f"| {s.retriever} | {s.kind} | {s.questions} | {s.recall_at_1:.3f} "
            f"| {s.recall_at_5:.3f} | {s.recall_at_10:.3f} | {s.mrr:.3f} | {s.milliseconds:g} |"
        )
    return "\n".join(lines)


def build_retrievers(config: dict, names: Sequence[str]) -> dict[str, Retriever]:
    """The retrievers asked for, over the same documents in the same order."""
    from local_rag.cli import build_dense_store, load_documents

    documents = load_documents(config)
    lexical = BM25Index()
    lexical.index(documents)
    dense = build_dense_store(config, documents) if {"dense", "hybrid"} & set(names) else None
    available: dict[str, Callable[[], Retriever]] = {
        "lexical": lambda: lexical,
        "dense": lambda: dense,
        "hybrid": lambda: HybridRetriever([lexical, dense]),
    }
    return {name: available[name]() for name in names}


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="local-rag-eval", description=__doc__)
    parser.add_argument("--config", default=None, help="Path to the YAML configuration")
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVERS, default=list(RETRIEVERS))
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    parser.add_argument(
        "--system-certs",
        action="store_true",
        help="Verify TLS against the OS trust store (needed behind a TLS-inspecting proxy)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    if args.system_certs:
        from local_rag.cli import use_system_certificates

        use_system_certificates()
    config = load_config(args.config)
    settings = config["evaluation"]
    items = load_eval(resolve(settings["path"]))
    retrievers = build_retrievers(config, args.retrievers)

    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw = args.output / f"outcomes-{stamp}.jsonl"
    with raw.open("w") as sink:

        def record(outcome: Outcome) -> None:
            sink.write(json.dumps(outcome.to_dict(), ensure_ascii=False) + "\n")

        outcomes = evaluate(retrievers, items, settings["id_field"], report=record)
    table = markdown_table(summarise(outcomes))
    model = config["vectordb"]["encoder_model"]
    (args.output / f"summary-{stamp}.md").write_text(f"{table}\n\nEncoder: {model}\n")
    print(table)
    print(f"\noutcomes: {raw}", file=sys.stderr)
