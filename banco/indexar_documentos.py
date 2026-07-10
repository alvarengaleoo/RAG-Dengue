import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DOCUMENTOS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
)

# Carrega as variáveis do .env
load_dotenv()


class IndexadorDocumentos:

    def __init__(self):

        self.loader = PyPDFLoader(
            str(DOCUMENTOS_PATH / "DENGUE.pdf")
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # Modelo utilizado para gerar os embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    def carregar_documentos(self):
        documentos = self.loader.load()

        print(f"Total de páginas carregadas: {len(documentos)}")

        return documentos

    def criar_chunks(self, documentos):
        chunks = self.text_splitter.split_documents(documentos)

        print(f"Total de chunks criados: {len(chunks)}")

        return chunks

    # Envia os documentos para o Qdrant
    def indexar_documentos(self, chunks):

        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=COLLECTION_NAME,
        )

        print("Documentos indexados com sucesso!")


if __name__ == "__main__":

    indexador = IndexadorDocumentos()

    documentos = indexador.carregar_documentos()

    chunks = indexador.criar_chunks(documentos)

    indexador.indexar_documentos(chunks)