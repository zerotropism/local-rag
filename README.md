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
├── protocols.py   Embedder and VectorStore
├── embedders.py   SentenceTransformerEmbedder
├── stores.py      QdrantStore
├── corpus.py      CSV to Documents
├── config.py      YAML loading, root-relative paths
├── chatbot.py     retrieval, prompt building, conversation
└── cli.py         entry point
```

`Document.payload` is what keeps the store corpus-agnostic: the store embeds `text` and returns
`payload` untouched, so it never learns that a row has a `region` or a `variety`.

`Chatbot` takes a `VectorStore` and a chat callable, both injected. That is what lets the
conversation tests run with a fake store and a fake model, asserting on the prompt that gets
built rather than on a live answer.

## Retrieval quality

Search is dense-only: the question is embedded and compared by cosine similarity. On the wine
catalogue this retrieves plausible neighbours but discriminates poorly — a query for a white
wine from one region can return a red one from the same region, because the colour is a weak
signal inside two hundred words of tasting notes.

Hybrid search (BM25 combined with vectors) and a recall@k evaluation set are the next step.
Until there is a measurement, tuning the retrieval would be guesswork.

The prompt compensates in the meantime: the model is instructed to answer only from the
retrieved documents and to say so when none fit, rather than answering from its own knowledge.

## Tests

```bash
uv run pytest
```

No model download, no Qdrant server, no Ollama: the store tests use a deterministic toy
embedder, the chatbot tests a fake store and a fake chat backend.

## Dependencies

| Package                 | Role                                    |
|-------------------------|-----------------------------------------|
| `qdrant-client`         | Vector store, in-memory or on a server  |
| `sentence-transformers` | Local embeddings                        |
| `ollama`                | Local chat model                        |
| `pyyaml`                | Configuration                           |
| `truststore`            | TLS verification against the OS store   |
