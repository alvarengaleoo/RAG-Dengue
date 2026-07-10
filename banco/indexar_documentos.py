from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCUMENTOS_PATH, CHUNK_SIZE, CHUNK_OVERLAP


class CarregadorDocumento:
    def __init__(self):
        self.loader = PyPDFLoader(
            str(DOCUMENTOS_PATH / "DENGUE.pdf")
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def carregar_documentos(self):
        documentos = self.loader.load()
        print(f"Total de paginas carregadas: {len(documentos)}")
        return documentos
    
    def criar_chunks(self, documentos):
        chunks = self.text_splitter.split_documents(documentos)
        print(f"Total de chunks criados: {len(chunks)}")
        return chunks
    

if __name__ == "__main__":

    carregador = CarregadorDocumento()

    documentos = carregador.carregar_documentos()

    chunks = carregador.criar_chunks(documentos)

    print()

    print(chunks[0].page_content)

    print()

    print(chunks[0].metadata)
        