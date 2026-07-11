"""Application configuration loaded from environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load the .env from the project root (one level above rag-app/)
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# --- Paths -------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parents[1]          # .../rag-app
STORAGE_DIR = APP_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
INDEX_DIR = STORAGE_DIR / "faiss_index"
FRONTEND_DIR = APP_DIR / "frontend"

for _d in (STORAGE_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Secrets / models --------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Admin password for the ingestion panel. Override in .env with ADMIN_PASSWORD.
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Chat model served through Groq (fast + free tier). Change if you like.
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama-3.3-70b-versatile")

# Local sentence-transformers embedding model (no API cost).
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Retrieval / chunking ----------------------------------------------------
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
