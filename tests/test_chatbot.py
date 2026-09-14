"""Conversation logic, with a fake store and a fake model: no Qdrant, no Ollama."""

from local_rag.chatbot import Chatbot, format_hits
from local_rag.models import SearchHit

HITS = [
    SearchHit(score=0.9, payload={"name": "Chablis", "region": "Burgundy"}),
    SearchHit(score=0.4, payload={"name": "Shiraz", "region": "Barossa"}),
]


class FakeStore:
    """Records queries and replays fixed hits."""

    def __init__(self, hits=HITS):
        self.hits = hits
        self.queries = []

    def index(self, documents) -> int:
        return len(documents)

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        self.queries.append((query, limit))
        return self.hits[:limit]

    def count(self) -> int:
        return len(self.hits)


def make_bot(store=None, replies=("an answer",)):
    answers = list(replies)
    prompts = []

    def fake_chat(model: str, prompt: str) -> str:
        prompts.append(prompt)
        return answers.pop(0)

    bot = Chatbot(store or FakeStore(), template="TEMPLATE", model="m", chat=fake_chat)
    return bot, prompts


def test_hits_are_rendered_whatever_their_payload() -> None:
    rendered = format_hits([SearchHit(score=0.5, payload={"title": "x", "year": 2024})])
    assert "title: x" in rendered
    assert "year: 2024" in rendered


def test_no_hit_is_stated_rather_than_left_empty() -> None:
    assert "no relevant document" in format_hits([])


def test_the_question_reaches_the_store() -> None:
    store = FakeStore()
    bot, _ = make_bot(store)
    bot.answer("which white wine?")
    assert store.queries == [("which white wine?", 3)]


def test_the_prompt_carries_template_hits_and_question() -> None:
    bot, prompts = make_bot()
    bot.answer("which white wine?")

    assert "TEMPLATE" in prompts[0]
    assert "Chablis" in prompts[0]
    assert "which white wine?" in prompts[0]


def test_history_accumulates_across_turns() -> None:
    bot, prompts = make_bot(replies=("first", "second"))
    bot.answer("one")
    bot.answer("two")

    assert "first" in prompts[1]
    assert "one" in prompts[1]


def test_retrieval_failure_is_not_swallowed() -> None:
    """The regression this guards: handle_exception turned a broken store into empty results."""

    class BrokenStore(FakeStore):
        def search(self, query, limit=3):
            raise RuntimeError("qdrant is down")

    bot, _ = make_bot(BrokenStore())
    try:
        bot.answer("anything")
    except RuntimeError:
        return
    raise AssertionError("a store failure must reach the caller")
