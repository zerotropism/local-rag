"""Command line entry point: build the pipeline from the configuration, then converse."""

import argparse

from local_rag.chatbot import Chatbot
from local_rag.config import load_config, resolve
from local_rag.corpus import load_csv
from local_rag.embedders import SentenceTransformerEmbedder
from local_rag.protocols import VectorStore
from local_rag.stores import QdrantStore


def build_store(config: dict) -> VectorStore:
    """Load the corpus and index it, reporting how many documents made it in."""
    data = config["data"]
    vectordb = config["vectordb"]

    documents = load_csv(
        resolve(data["path"]),
        text_field=data["text_field"],
        required_fields=data.get("required_fields", ()),
    )
    store = QdrantStore(
        SentenceTransformerEmbedder(vectordb["encoder_model"]),
        location=vectordb.get("location", ":memory:"),
        collection_name=vectordb["collection_name"],
    )
    print(f"Indexing {len(documents)} documents...")
    print(f"Collection '{vectordb['collection_name']}' holds {store.index(documents)} points.")
    return store


def build_chatbot(config: dict, store: VectorStore) -> Chatbot:
    chatbot = config["chatbot"]
    return Chatbot(
        store=store,
        template=chatbot["initial_template"],
        model=chatbot["llm_model"],
        theme=chatbot.get("theme", ""),
    )


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
        import truststore

        truststore.inject_into_ssl()

    config = load_config(args.config)
    build_chatbot(config, build_store(config)).run()


if __name__ == "__main__":
    main()
