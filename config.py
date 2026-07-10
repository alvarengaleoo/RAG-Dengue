from pathlib import Path

# Caminho principal do projeto
BASE_DIR = Path(__file__).parent

# Pasta dos documentos
DOCUMENTOS_PATH = BASE_DIR / "documentos"

# Banco vetorial
COLLECTION_NAME = "chatbot_dengue"

# Modelo de embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configuração dos chunks
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Quantidade de documentos retornados
TOP_K = 4