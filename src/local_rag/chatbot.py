import ollama
from decorators import handle_exception
from vectordb import VectorDB


class Chatbot:
    def __init__(
        self,
        theme: str,
        template: str,
        model: str = "llama3.1",
        rag: VectorDB = None,
    ):
        """Initialize the Chatbot class.
        Args:
            theme (str): Theme of the chatbot.
            template (str): Template for the chatbot conversation.
            model (str, optional): Model name. Defaults to "llama3.1".
            rag (VectorDB, optional): Vector database object. Defaults to None.
        """
        self._theme = theme
        self._model = model
        self._rag = rag
        self._template = template
        self._search_results = None
        self._context = ""
        self._prompt = None

    @property
    def theme(self) -> str:
        """Get the theme of the chatbot
        Args:
            value (str): Theme of the chatbot.
        """
        return self._theme

    @handle_exception
    def handle_conversation(self):
        """Handle the conversation with the chatbot.
        This method initiates the conversation with the user and handles the interaction.
        It uses the vector database to retrieve relevant information based on user queries.
        """
        # start conversation
        print(f"Welcome to the local {self._theme} chatbot! Type 'exit' to quit.")
        while True:
            # collect user input query
            user_input = input("You: ")
            # exit if user types 'exit'
            if user_input.lower() == "exit":
                break
            # get search retrieval results in the vector database
            self._search_results = self._rag.search_vector_db(
                user_input, return_limit=3, text_output=True
            )
            # we print the search results to the console for control purposes
            print(
                "\n\n",
                "Vector DB search results:",
                "\n\n",
                self._search_results,
                "\n\n",
            )
            # set user prompt as a concatenation of initial template, conversation history,
            # the search results & user input
            self._prompt = (
                self._template
                + f"""
            Here is the conversation history: {self._context}

            Here are the top search results: {self._search_results}

            Question: {user_input}

            Answer:
            """
            )
            # call the ollama API to get the response from the model
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": self._prompt}],
            )["message"]["content"]
            # print the response to user
            print("Chatbot:", response)
            # update the conversation history as context for next iteration
            self._context += f"\nYou: {user_input}\nChatbot: {response}"
