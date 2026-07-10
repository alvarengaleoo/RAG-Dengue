from pathlib import Path

BASE_DIR = Path(__file__).parent

DOCUMENTOS_PATH = BASE_DIR / "documentos"

COLLECTION_NAME = "chatbot_dengue"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TOP_K = 4
