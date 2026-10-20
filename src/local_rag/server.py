"""MCP adapter: the corpus the chatbot answers from, as a search tool and an ask tool.

The corpus is indexed once, at start-up, with the retriever `config.yaml` names. Answering calls
a local model, which blocks: it runs in a worker thread under a timeout, so a slow model can
neither stall the server nor hold a client indefinitely.
"""

import argparse
import asyncio
import os
from collections.abc import Callable
from typing import Any

import ollama
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel

from local_rag.chatbot import Chatbot
from local_rag.cli import build_chatbot, build_retriever, use_system_certificates
from local_rag.config import load_config
from local_rag.models import SearchHit
from local_rag.protocols import Retriever

TIMEOUT_ENV_VAR = "LOCAL_RAG_TIMEOUT"
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_K = 20
READ_ONLY = {"readOnlyHint": True, "openWorldHint": False}


class Hit(BaseModel):
    """One retrieved document: its score and every field of its row."""

    score: float
    document: dict[str, Any]


class Answer(BaseModel):
    answer: str
    sources: list[Hit]


def as_hits(hits: list[SearchHit]) -> list[Hit]:
    return [Hit(score=round(hit.score, 4), document=hit.payload) for hit in hits]


def create_server(
    retriever: Retriever,
    make_chatbot: Callable[[], Chatbot] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    name: str = "local-rag",
) -> FastMCP:
    """Build the server around an indexed retriever. `make_chatbot` returns a fresh Chatbot."""
    mcp = FastMCP(name)

    @mcp.tool(annotations=READ_ONLY)
    def search(query: str, k: int = 5) -> list[Hit]:
        """The k documents of the corpus closest to the query, best first."""
        if not 1 <= k <= MAX_K:
            raise ToolError(f"k must be between 1 and {MAX_K}")
        return as_hits(retriever.search(query, limit=k))

    @mcp.tool(annotations=READ_ONLY)
    async def ask(question: str) -> Answer:
        """Answer from the corpus only, with the documents the answer was built from.

        Each call is independent: no conversation history is kept between calls.
        """
        if make_chatbot is None:
            raise ToolError("no chat model is configured")
        chatbot = make_chatbot()
        try:
            text, hits = await asyncio.wait_for(
                asyncio.to_thread(chatbot.answer, question), timeout
            )
        except TimeoutError:
            raise ToolError(f"the model did not answer within {timeout:g}s") from None
        except ConnectionError as exc:
            raise ToolError(f"Ollama is unreachable ({exc}): start it, or set OLLAMA_HOST") from exc
        except ollama.ResponseError as exc:
            raise ToolError(f"the model failed: {exc}") from exc
        return Answer(answer=text, sources=as_hits(hits))

    return mcp


def main(argv: list[str] | None = None) -> None:
    """Console entry point: index the configured corpus, then serve over stdio."""
    parser = argparse.ArgumentParser(prog="local-rag-mcp", description=__doc__)
    parser.add_argument("--config", default=None, help="Path to the YAML configuration")
    parser.add_argument("--system-certs", action="store_true", help="Use the OS trust store")
    args = parser.parse_args(argv)
    if args.system_certs:
        use_system_certificates()

    config = load_config(args.config)
    retriever = build_retriever(config)
    timeout = float(os.environ.get(TIMEOUT_ENV_VAR, DEFAULT_TIMEOUT_SECONDS))
    theme = config.get("chatbot", {}).get("theme", "")
    server = create_server(
        retriever,
        make_chatbot=lambda: build_chatbot(config, retriever),
        timeout=timeout,
        name=f"local-rag {theme}".strip(),
    )
    server.run(show_banner=False)


if __name__ == "__main__":
    main()
