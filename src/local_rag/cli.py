"""Command line entry point: build the pipeline from the configuration, then converse."""

import argparse

from local_rag.chatbot import Chatbot
from local_rag.config import load_config, resolve
from local_rag.corpus import load_csv
from local_rag.embedders import make_embedder
from local_rag.models import Document
from local_rag.protocols import VectorStore
from local_rag.stores import QdrantStore


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


def build_store(config: dict) -> VectorStore:
    """Load the corpus and index it, reporting how many documents made it in."""
    documents = load_documents(config)
    print(f"Indexing {len(documents)} documents...")
    store = build_dense_store(config, documents)
    print(f"Collection '{config['vectordb']['collection_name']}' holds {store.count()} points.")
    return store


def build_chatbot(config: dict, store: VectorStore) -> Chatbot:
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
    build_chatbot(config, build_store(config)).run()


if __name__ == "__main__":
    main()
