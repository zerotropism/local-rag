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
  encoder_model: "all-MiniLM-L6-v2"
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
├── embedders.py   SentenceTransformerEmbedder
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
uv run local-rag-eval                          # the three retrievers; downloads the encoder once
uv run local-rag-eval --retrievers lexical     # BM25 only: no model, a few seconds
```

Behind a TLS-inspecting proxy, add `--system-certs`. Each question's outcome, with its ranked
wines, goes to `evaluation/outcomes-<time>.jsonl`, and the table to `summary-<time>.md`.

BM25 on the bundled questions:

| Retriever | Questions | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|---|
| lexical | all | 30 | 0.472 | 0.632 | 0.644 | 0.667 |
| lexical | name | 10 | 0.566 | 0.896 | 0.933 | 1.000 |
| lexical | notes | 10 | 0.850 | 1.000 | 1.000 | 1.000 |
| lexical | paraphrase | 10 | 0.000 | 0.000 | 0.000 | 0.000 |

Recall@1 on `name` is capped below 1: those questions have several relevant vintages. The
paraphrases score 0 by construction. The questions were written for this evaluation by one
author: they compare the retrievers with each other, not with real users.

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
| `ollama`                | Local chat model                        |
| `pyyaml`                | Configuration                           |
| `truststore`            | TLS verification against the OS store   |
