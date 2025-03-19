from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


class Chatbot:
    def __init__(
        self,
        question: str,
        search: str,
        theme: str,
        template: str,
        model: str = "llama3.1",
    ):
        self.question = question
        self.search_results = search
        self.theme = theme
        self.llm = OllamaLLM(model=model)
        self.context = ""
        self.template = (
            template
            + f"""
        Here is the conversation history: {self.context}

        Here are the search results: {self.search_results}

        Question: {self.question}

        Answer:
        """
        )
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.llm

    def start_conversation(self):
        print(f"Welcome to the local {self.theme} chatbot! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            result = self.chain.invoke(
                {
                    "context": self.context,
                    "search_results": str(self.search_results),
                    "question": user_input,
                }
            )
            print("Chatbot:", result)
            self.context += f"\nYou: {user_input}\nChatbot: {result}"
