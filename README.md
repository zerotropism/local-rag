# local-rag

A local retrieval-augmented chatbot: index a CSV corpus into a vector store, retrieve against a
question, and let a local model answer from what was retrieved. Nothing leaves the machine.

The embedder and the vector store sit behind protocols, so either can be swapped without
touching the other — and the corpus is configuration, not code. The shipped example is a wine
catalogue; pointing it at another CSV takes three lines of YAML.

## Installation

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and a running [Ollama](https://ollama.com/).

```bash
uv sync
ollama pull gemma3:12b
uv run local-rag
```

The first run downloads the embedding model from Hugging Face. Behind a TLS-inspecting
corporate proxy, Python rejects the intercepted certificate while the system trust store accepts
it; `--system-certs` makes verification use the OS trust store instead of the bundled one:

```bash
uv run local-rag --system-certs
```

## Configuration

`config.yaml` describes the corpus, the vector store and the chatbot:

```yaml
data:
  path: "data/top_rated_wines.csv"
  text_field: ["name", "region", "variety", "notes"]
  required_fields: ["variety"]
evaluation:
  path: "data/eval.jsonl"
  id_field: "name"
vectordb:
  encoder: "sentence-transformers:all-MiniLM-L6-v2"   # or "ollama:<model>"
  location: ":memory:"
  collection_name: "top_wines"
chatbot:
  theme: "wine"
  initial_template: >
    You are a wine specialist. Answer using only the search results provided below.
    Never mention a wine that is not in those results.
  llm_model: "gemma3:12b"
```

`text_field` accepts one column or several. Several columns are concatenated as `field: value`
lines before embedding, which is what makes metadata searchable: with only the free-text column
indexed, a query mentioning a region or a grape variety has nothing to match against.

Paths are resolved from the project root, so the command works from any directory.

## Using another corpus

Any CSV with a header row works. Point `data.path` at it, name the columns to embed in
`text_field`, and list in `required_fields` the columns that must be non-empty for a row to be
kept. The whole row is stored as payload and returned with each hit, so no code knows which
columns your corpus carries.

## Architecture

```
src/local_rag/
├── models.py      Document (text + payload) and SearchHit
├── protocols.py   Embedder, Retriever and VectorStore
├── embedders.py   SentenceTransformerEmbedder, OllamaEmbedder, make_embedder("backend:model")
├── stores.py      QdrantStore
├── lexical.py     BM25Index, in memory
├── hybrid.py      HybridRetriever: reciprocal rank fusion of several retrievers
├── evaluation.py  recall@k and MRR per retriever and kind of question, local-rag-eval
├── corpus.py      CSV to Documents
├── config.py      YAML loading, root-relative paths
├── chatbot.py     retrieval, prompt building, conversation
└── cli.py         entry point
```

`Document.payload` is what keeps the store corpus-agnostic: the store embeds `text` and returns
`payload` untouched, so it never learns that a row has a `region` or a `variety`.

`Chatbot` takes a `Retriever` and a chat callable, both injected. That is what lets the
conversation tests run with a fake store and a fake model, asserting on the prompt that gets
built rather than on a live answer.

## Retrieval quality

The chatbot searches by vectors only for now: the question is embedded and compared by cosine
similarity. On the wine catalogue this retrieves plausible neighbours but discriminates poorly
on names and places, which a 384-dimension embedding of two hundred words of tasting notes
blurs. The prompt compensates: the model answers only from the retrieved documents and says so
when none fit.

Three retrievers can be compared on the same documents:

| Retriever | How it ranks |
|---|---|
| `dense` | cosine similarity between embeddings, in Qdrant |
| `lexical` | BM25 over the same text, accents and case ignored |
| `hybrid` | reciprocal rank fusion of both: each ranks its first 50, a document scores the sum of `1 / (60 + rank)` |

## Evaluation

`data/eval.jsonl` holds 30 questions, each with the wines that answer it and a kind:

| Kind | Questions | What it tests | Relevant wines |
|---|---|---|---|
| `name` | 10 | producer and wine names, as a customer types them | every wine whose name contains the producer and wine named, all vintages |
| `notes` | 10 | tasting vocabulary taken from the notes | the described wine, plus any other row with the same notes (60 % of words or more shared) |
| `paraphrase` | 10 | the same kind of description, reworded | as for `notes`; each question shares no content word with its wine, which leaves only meaning to match on |

Relevance is by `name` (`evaluation.id_field` in `config.yaml`), so duplicate rows of a wine
count once. Each question reports recall at 1, 5 and 10 (share of its relevant wines found) and
its reciprocal rank; results are averaged per kind and overall.

```bash
uv run local-rag-eval                          # the three retrievers, configured encoder
uv run local-rag-eval --retrievers lexical     # BM25 only: no model, a few seconds
uv run local-rag-eval --encoders sentence-transformers:all-MiniLM-L6-v2 ollama:nomic-embed-text
```

`--encoders` measures the dense and hybrid retrievers once per encoder, in one run. An encoder
is `sentence-transformers:<model>`, downloaded from Hugging Face on first use (add
`--system-certs` behind a TLS-inspecting proxy), or `ollama:<model>`, served by a local Ollama
after `ollama pull <model>`. Models trained with an instruction in front of queries or documents
get it automatically: `nomic-embed-text` and `mxbai-embed-large`.

Each question's outcome, with its ranked wines, goes to `evaluation/outcomes-<time>.jsonl`, and
the table to `summary-<time>.md`.

Measured on 20 October 2026, three encoders in one run (`evaluation/summary-20261020T091626Z.md`
holds every row, the `.jsonl` next to it every question). Mean reciprocal rank per kind of
question:

| Retriever | Encoder | name | notes | paraphrase | all | Recall@10, all | ms / query |
|---|---|---|---|---|---|---|---|
| lexical | — | 1.000 | 1.000 | 0.000 | 0.667 | 0.644 | 1.6 |
| dense | all-MiniLM-L6-v2 | 1.000 | 0.233 | 0.250 | 0.494 | 0.556 | 6 |
| dense | nomic-embed-text | 1.000 | 0.250 | 0.061 | 0.437 | 0.500 | 18.4 |
| dense | mxbai-embed-large | 0.933 | 0.089 | 0.183 | 0.402 | 0.533 | 35 |
| hybrid | all-MiniLM-L6-v2 | 1.000 | 0.599 | 0.043 | 0.548 | 0.722 | 7.8 |
| hybrid | nomic-embed-text | 1.000 | 0.506 | 0.048 | 0.518 | 0.661 | 19.4 |
| hybrid | mxbai-embed-large | 1.000 | 0.589 | 0.046 | 0.545 | 0.672 | 38.5 |

- BM25 ranks best overall, and no encoder beats it. The two larger encoders do worse than
  `all-MiniLM-L6-v2`; `mxbai-embed-large` never ranks the wine described by its notes first
  (recall@1 of 0). Every wine is described in the same vocabulary, and a whole tasting note
  embedded as one vector says "a fine red wine" more than it says which one.
- The dense retrievers are the only ones that find paraphrased descriptions, at best
  `all-MiniLM-L6-v2`, with three in ten within its first ten hits.
- Rank fusion finds the most relevant wines within ten (0.722) but ranks them lower than BM25
  alone. BM25 always returns 50 hits, since common words such as "in" or "fruit" match hundreds
  of wines; equal-weight fusion then lets a wine that both lists rank in the middle overtake one
  that a single list ranks first. The chatbot passes its first three hits to the model, so the
  first ranks are what matter.
- Recall@1 on `name` is capped below 1: those questions have several relevant vintages. The
  paraphrases score 0 with BM25 by construction.
- The `all-MiniLM-L6-v2` and BM25 rows reproduce the 12 October run exactly.

Decision: BM25 becomes the default retriever. It ranks best, needs no model and answers in two
milliseconds; the dense and hybrid retrievers stay available for corpora where wording varies
more. The questions were written for this evaluation by one author: they compare the
retrievers with each other, not with real users, and the paraphrases were built to defeat BM25.

## Tests

```bash
uv run pytest
```

No model download, no Qdrant server, no Ollama: the store tests use a deterministic toy
embedder, the chatbot tests a fake store and a fake chat backend, and the evaluation tests run
BM25 on the bundled questions.

## Dependencies

| Package                 | Role                                    |
|-------------------------|-----------------------------------------|
| `qdrant-client`         | Vector store, in-memory or on a server  |
| `sentence-transformers` | Local embeddings                        |
| `ollama`                | Local chat model and embeddings         |
| `pyyaml`                | Configuration                           |
| `truststore`            | TLS verification against the OS store   |
