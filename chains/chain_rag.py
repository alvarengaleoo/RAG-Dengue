import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
)

# Carrega as variáveis do .env
load_dotenv()

# Modelo de Embeddings (o mesmo usado na indexação)
modelo_embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# Conecta na coleção já existente do Qdrant
db_vetorial = QdrantVectorStore.from_existing_collection(
    collection_name=COLLECTION_NAME,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    embedding=modelo_embedding,
)

# Cria o Retriever
retriever = db_vetorial.as_retriever(
    search_kwargs={"k": TOP_K}
)

# Modelo LLM da Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# Prompt que será enviado para a LLM
prompt = ChatPromptTemplate.from_template(
    """
Você é um assistente especializado em dengue.

Responda apenas utilizando as informações presentes no contexto fornecido.

Se a resposta não estiver no contexto, informe que essa informação não foi encontrada na base de conhecimento.

Contexto:
{contexto}

Pergunta:
{pergunta}
"""
)


def criar_contexto(documentos):
    """
    Junta todos os chunks recuperados em um único texto.
    """
    return "\n\n".join(
        documento.page_content
        for documento in documentos
    )


def responder(pergunta):
    """
    Busca os documentos mais relevantes e gera a resposta.
    """

    # Busca os documentos no Qdrant
    documentos = retriever.invoke(pergunta)

    # Apenas para estudo (pode remover depois)
    print("\n===== CHUNKS RECUPERADOS =====\n")

    for i, doc in enumerate(documentos, start=1):
        print(f"Chunk {i}")
        print("-" * 50)
        print(doc.page_content)
        print()

    # Concatena os chunks em um único contexto
    contexto = criar_contexto(documentos)

    # Envia o contexto para a LLM
    chain = prompt | llm | StrOutputParser()

    resposta = chain.invoke(
        {
            "contexto": contexto,
            "pergunta": pergunta,
        }
    )

    return resposta


if __name__ == "__main__":

    pergunta = input("Digite sua pergunta: ")

    resposta = responder(pergunta)

    print("\n===== RESPOSTA =====\n")

    print(resposta)