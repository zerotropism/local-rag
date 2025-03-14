from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

llm = OllamaLLM(model="llama3.1")
template = """
You are chatbot, a wine specialist. 
Your top priority is to help guide users into selecting amazing wine and guide them with their requests.
Answer the question below using the conversation history and the search results content provided.
Here is the conversation history: {context}
Here are the search results: {search_results}
Question: {question}
Answer:
"""
context = ""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm
response = chain.invoke(
    {"context": context, "search_results": str(search_results), "question": user_prompt}
)
print(response)
