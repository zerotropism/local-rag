import json
from vectordb import VectorDB
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


class Chatbot:
    def __init__(
        self,
        theme: str,
        template: str,
        model: str = "llama3.1",
        rag: VectorDB = None,
    ):
        self.theme = theme
        self.llm = OllamaLLM(model=model)
        self.rag = rag
        self.template = template
        self.search_results = None
        self.context = ""
        self.prompt = None
        self.chain = None

    def handle_conversation(self):
        # start conversation
        print(f"Welcome to the local {self.theme} chatbot! Type 'exit' to quit.")
        while True:
            # collect user input query
            user_input = input("You: ")
            # exit if user types 'exit'
            if user_input.lower() == "exit":
                break
            # get search retrieval results in the vector database
            self.search_results = self.rag.search_vector_db(
                user_input, return_limit=3, text_output=True
            )
            # set user prompt with the search results
            self.prompt = ChatPromptTemplate.from_template(
                self.template
                + f"""
            Here is the conversation history: {self.context}

            Here are the top search results: {self.search_results}

            Question: {user_input}

            Answer:
            """
            )
            print(self.prompt)
            # set the chain of response generation
            self.chain = self.prompt | self.llm
            # invoke the chain of response generation
            response = self.chain.invoke(
                {
                    "context": self.context,
                    "search_results": str(self.search_results),
                    "question": user_input,
                }
            )
            # print the response to user
            print("Chatbot:", response)
            # update the conversation history as context for next iteration
            self.context += f"\nYou: {user_input}\nChatbot: {response}"
