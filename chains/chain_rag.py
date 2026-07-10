import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from config import COLLECTION_NAME, EMBEDDING_MODEL, TOP_K


load_dotenv()


modelo_embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL)

db_vetorial = QdrantVectorStore.from_existing_collection(
    collection_name=COLLECTION_NAME,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    embedding=modelo_embedding)

retriever = db_vetorial.as_retriever(
    search_kwargs={
        "k": TOP_K})

def buscar_documentos(pergunta):

    documentos = retriever.invoke(pergunta)

    print()

    print("===== DOCUMENTOS RECUPERADOS =====")

    for i, documento in enumerate(documentos, start=1):

        print(f"\nChunk {i}")

        print("-" * 40)

        print(documento.page_content)

    return documentos


if __name__ == "__main__":

    pergunta = input("Digite sua pergunta: ")

    buscar_documentos(pergunta)