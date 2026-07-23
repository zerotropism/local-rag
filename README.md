# local-rag

Reimplementation of [@alfredodeza](https://github.com/alfredodeza)'s introduction to RAG course.

A fully **local** Retrieval-Augmented Generation (RAG) chatbot that recommends wines,
combining semantic search (Qdrant) with a local LLM (Ollama) — no cloud API required.

## Requirements

* Python 3.12
* [Ollama](https://ollama.com/) installed and running
* Install Python dependencies:
  ```bash
  pip install -r requirements.txt
  ```
* Pull the LLM model used by the chatbot:
  ```bash
  ollama pull gemma3:12b
  ```

## Definition

Custom local RAG setup with **Qdrant** (in-memory vector database) and **Ollama**
(local LLM inference). Wine tasting notes are embedded with a
`sentence-transformers` model and retrieved by semantic similarity to ground the
LLM's answers.

## How it works

1. **Index** — each wine from the CSV is encoded into a vector (`all-MiniLM-L6-v2`)
   and stored in an in-memory Qdrant collection.
2. **Retrieve** — the user's question is embedded and the top matching wines are
   fetched via cosine similarity.
3. **Generate** — the retrieved wines and the conversation history are injected into
   a prompt sent to the local LLM (`gemma3:12b`), which answers as a wine specialist.

## Usage

```bash
python code/main.py
```

Then chat in the terminal. Type `exit` to quit.

## Content

* `data/top_rated_wines.csv` — the wines dataset
* `notebook.ipynb` — building the initial reasoning
* `code/` — factored version of the code
  * `main.py` — entry point: loads config, builds the vector DB, starts the chatbot
  * `vectordb.py` — `VectorDB` class: embeddings, Qdrant collection & similarity search
  * `chatbot.py` — `Chatbot` class: conversation loop and LLM calls via LangChain
  * `decorators.py` — `@handle_exception` error-handling decorator
  * `config.yaml` — dataset paths, encoder model, collection name and LLM settings

## Configuration

All runtime settings live in `code/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `data` | `data/top_rated_wines.csv` | Source dataset |
| `vectordb.encoder_model` | `all-MiniLM-L6-v2` | Sentence-transformers embedding model |
| `vectordb.instance_mode` | `:memory:` | Qdrant mode (in-memory, rebuilt each run) |
| `vectordb.collection_name` | `top_wines` | Qdrant collection name |
| `chatbot.llm_model` | `gemma3:12b` | Ollama model used for generation |
| `chatbot.theme` | `wine` | Chatbot persona |