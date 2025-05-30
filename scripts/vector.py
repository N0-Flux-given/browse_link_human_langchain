from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("data/zomato.csv")
df = df.head(10)
print("csv shape :", df.shape)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"


def add_documents():
    print("df shape : ", df.shape)
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["restaurant name"],
            metadata={
                "restaurant_type": row["restaurant type"],
                "rating": row["rate (out of 5)"],
                "cuisines": row["cuisines type"],
            },
            id=str(i),
        )
        ids.append(str(i))
        documents.append(document)
        print("Appended document:", str(i))

        vector_store = Chroma(
            collection_name="restaurant_reviews",
            embedding_function=embeddings,
            persist_directory=db_location,
        )

        vector_store.add_documents(documents=documents, ids=ids)
        # vector_store.persist()


def get_documents():
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings,
    )
    return vector_store


if __name__ == "__main__":
    add_documents()
