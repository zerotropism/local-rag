"""Retrieval-augmented conversation over any VectorStore."""

from collections.abc import Callable, Sequence

import ollama

from local_rag.models import SearchHit
from local_rag.protocols import VectorStore

DEFAULT_LIMIT = 3


def ollama_chat(model: str, prompt: str) -> str:
    """Default chat backend. Injected, so the conversation can be tested without a server."""
    return ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])["message"][
        "content"
    ]


def format_hits(hits: Sequence[SearchHit]) -> str:
    """Render hits for the prompt. Works for any payload, whatever its fields."""
    if not hits:
        return "(no relevant document found)"
    return "\n\n".join(
        "\n".join([f"score: {hit.score:.3f}", *(f"{k}: {v}" for k, v in hit.payload.items())])
        for hit in hits
    )


class Chatbot:
    """Holds the conversation history and asks the store before asking the model."""

    def __init__(
        self,
        store: VectorStore,
        template: str,
        model: str = "llama3.1",
        theme: str = "",
        limit: int = DEFAULT_LIMIT,
        chat: Callable[[str, str], str] = ollama_chat,
    ) -> None:
        self.store = store
        self.template = template
        self.model = model
        self.theme = theme
        self.limit = limit
        self.chat = chat
        self.history = ""

    def build_prompt(self, question: str, hits: Sequence[SearchHit]) -> str:
        """Pure function of its inputs, so the prompt can be asserted on in tests."""
        return (
            f"{self.template}\n\n"
            f"Here is the conversation history:\n{self.history}\n\n"
            f"Here are the top search results:\n{format_hits(hits)}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def answer(self, question: str) -> tuple[str, list[SearchHit]]:
        """Retrieve, answer, and record the exchange in the history."""
        hits = self.store.search(question, limit=self.limit)
        response = self.chat(self.model, self.build_prompt(question, hits))
        self.history += f"\nYou: {question}\nChatbot: {response}"
        return response, hits

    def run(self, show_hits: bool = True) -> None:
        """Interactive loop. Type 'exit' to leave."""
        print(f"Welcome to the local {self.theme} chatbot! Type 'exit' to quit.")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if question.lower() == "exit":
                return
            if not question:
                continue

            response, hits = self.answer(question)
            if show_hits:
                print(f"\nVector DB search results:\n{format_hits(hits)}\n")
            print("Chatbot:", response)
