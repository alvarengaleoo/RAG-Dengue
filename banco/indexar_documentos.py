from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from config import DOCUMENTOS_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, COLLECTION_NAME


load_dotenv()


class IndexadorDocumentos:
    def __init__(self):
        self.loader = PyPDFLoader(
            str(DOCUMENTOS_PATH / "DENGUE.pdf"))

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,)
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL)

    def carregar_documentos(self):
        documentos = self.loader.load()
        print(f"Total de paginas carregadas: {len(documentos)}")
        return documentos
    
    def criar_chunks(self, documentos):
        chunks = self.text_splitter.split_documents(documentos)
        print(f"Total de chunks criados: {len(chunks)}")
        return chunks
    
    def indexar_documentos(self, chunks):
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=COLLECTION_NAME)

        print("Documentos indexados com sucesso!")


if __name__ == "__main__":

    carregador = IndexadorDocumentos()

    documentos = carregador.carregar_documentos()

    chunks = carregador.criar_chunks(documentos)

    carregador.indexar_documentos(chunks)
        