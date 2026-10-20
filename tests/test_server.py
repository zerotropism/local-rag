"""The MCP adapter through the in-memory FastMCP client: BM25 on a few documents, fake model."""

import time

import pytest
from fastmcp import Client

from local_rag.chatbot import Chatbot
from local_rag.lexical import BM25Index
from local_rag.models import Document
from local_rag.server import create_server

DOCUMENTS = [
    Document(text="Chablis: crisp white wine from Burgundy", payload={"name": "Chablis"}),
    Document(text="Shiraz: bold red from Barossa", payload={"name": "Shiraz"}),
]


@pytest.fixture
def retriever() -> BM25Index:
    index = BM25Index()
    index.index(DOCUMENTS)
    return index


def chatbot_factory(retriever, reply=lambda model, prompt: "Try the Shiraz."):
    return lambda: Chatbot(retriever, template="Answer from the results.", model="m", chat=reply)


async def test_tools_are_listed_and_read_only(retriever) -> None:
    async with Client(create_server(retriever)) as client:
        tools = await client.list_tools()
    assert {tool.name for tool in tools} == {"search", "ask"}
    assert all(tool.annotations.read_only_hint for tool in tools)


async def test_search_returns_scored_documents_best_first(retriever) -> None:
    async with Client(create_server(retriever)) as client:
        result = await client.call_tool("search", {"query": "Barossa shiraz", "k": 1})
    hits = result.structured_content["result"]
    assert [hit["document"]["name"] for hit in hits] == ["Shiraz"]
    assert hits[0]["score"] > 0


async def test_k_is_bounded(retriever) -> None:
    async with Client(create_server(retriever)) as client:
        result = await client.call_tool("search", {"query": "wine", "k": 0}, raise_on_error=False)
    assert result.is_error
    assert "between 1 and 20" in result.content[0].text


async def test_ask_answers_with_its_sources(retriever) -> None:
    prompts = []

    def reply(model, prompt):
        prompts.append(prompt)
        return "Try the Shiraz."

    server = create_server(retriever, chatbot_factory(retriever, reply))
    async with Client(server) as client:
        result = await client.call_tool("ask", {"question": "a bold red from Barossa?"})
    assert result.structured_content["answer"] == "Try the Shiraz."
    assert result.structured_content["sources"][0]["document"]["name"] == "Shiraz"
    assert "Barossa" in prompts[0]


async def test_each_question_starts_without_history(retriever) -> None:
    prompts = []

    def reply(model, prompt):
        prompts.append(prompt)
        return f"answer {len(prompts)}"

    server = create_server(retriever, chatbot_factory(retriever, reply))
    async with Client(server) as client:
        await client.call_tool("ask", {"question": "first?"})
        await client.call_tool("ask", {"question": "second?"})
    assert "answer 1" not in prompts[1]


async def test_an_unreachable_model_server_is_named(retriever) -> None:
    def down(model, prompt):
        raise ConnectionError("connection refused")

    server = create_server(retriever, chatbot_factory(retriever, down))
    async with Client(server) as client:
        result = await client.call_tool("ask", {"question": "q"}, raise_on_error=False)
    assert result.is_error
    assert "Ollama is unreachable" in result.content[0].text


async def test_a_slow_model_is_cut_off(retriever) -> None:
    def slow(model, prompt):
        time.sleep(1)
        return "late"

    server = create_server(retriever, chatbot_factory(retriever, slow), timeout=0.1)
    async with Client(server) as client:
        result = await client.call_tool("ask", {"question": "q"}, raise_on_error=False)
    assert result.is_error
    assert "within 0.1s" in result.content[0].text
