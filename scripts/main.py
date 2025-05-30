from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import add_documents, get_documents
import os

model = OllamaLLM(
    model="gemma3:4b",
)
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    print("Adding documents !")
    add_documents()
else:
    print("reading vector store !")
    vector_store = get_documents()


template = """
You are an expert in answering qwestions about the a pizza restaurant. 
Here are some user reviews : {reviews}.
Here is the question that you have to answer : {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    user_input = input("Enter a review (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    # Append the user input to the reviews list
    reviews = vector_store.as_retriever(search_kwargs={"k": 5})
    reviews.invoke(user_input)
    print("reviews!!!", reviews)
    result = chain.invoke({"reviews": reviews, "question": user_input})

    print(result)
