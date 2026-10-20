"""Command line entry point: build the pipeline from the configuration, then converse."""

import argparse

from local_rag.chatbot import Chatbot
from local_rag.config import load_config, resolve
from local_rag.corpus import load_csv
from local_rag.embedders import make_embedder
from local_rag.hybrid import HybridRetriever
from local_rag.lexical import BM25Index
from local_rag.models import Document
from local_rag.protocols import Retriever
from local_rag.stores import QdrantStore

RETRIEVALS = ("lexical", "dense", "hybrid")


def load_documents(config: dict) -> list[Document]:
    data = config["data"]
    return load_csv(
        resolve(data["path"]),
        text_field=data["text_field"],
        required_fields=data.get("required_fields", ()),
    )


def build_dense_store(
    config: dict, documents: list[Document], encoder: str | None = None
) -> QdrantStore:
    """Index the documents with `encoder`, by default the one configured."""
    vectordb = config["vectordb"]
    store = QdrantStore(
        make_embedder(encoder or vectordb["encoder"]),
        location=vectordb.get("location", ":memory:"),
        collection_name=vectordb["collection_name"],
    )
    store.index(documents)
    return store


def build_retriever(config: dict, documents: list[Document] | None = None) -> Retriever:
    """The configured retriever over the corpus: `retrieval` is lexical, dense or hybrid.

    BM25 is the default: it ranked best on the evaluation and needs no model (README).
    """
    mode = config.get("retrieval", "lexical")
    if mode not in RETRIEVALS:
        raise ValueError(f"retrieval {mode!r}: expected one of {RETRIEVALS}")
    documents = load_documents(config) if documents is None else documents
    lexical = BM25Index()
    lexical.index(documents)
    if mode == "lexical":
        return lexical
    dense = build_dense_store(config, documents)
    return dense if mode == "dense" else HybridRetriever([lexical, dense])


def build_chatbot(config: dict, store: Retriever) -> Chatbot:
    chatbot = config["chatbot"]
    return Chatbot(
        store=store,
        template=chatbot["initial_template"],
        model=chatbot["llm_model"],
        theme=chatbot.get("theme", ""),
    )


def use_system_certificates() -> None:
    """Verify TLS against the OS trust store, for model downloads behind a TLS-inspecting proxy."""
    import truststore

    truststore.inject_into_ssl()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG chatbot")
    parser.add_argument("--config", default=None, help="Path to the YAML configuration")
    parser.add_argument(
        "--system-certs",
        action="store_true",
        help="Verify TLS against the OS trust store (needed behind a TLS-inspecting proxy)",
    )
    args = parser.parse_args()

    if args.system_certs:
        use_system_certificates()

    config = load_config(args.config)
    documents = load_documents(config)
    print(
        f"Indexing {len(documents)} documents ({config.get('retrieval', 'lexical')} retrieval)..."
    )
    build_chatbot(config, build_retriever(config, documents)).run()


if __name__ == "__main__":
    main()
