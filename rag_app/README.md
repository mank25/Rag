# RAG App

A minimal Retrieval-Augmented Generation application:

- **Admin panel** (`/admin`) — password-protected document ingestion (PDF, DOCX, TXT, MD).
- **Public chat** (`/`) — anyone can ask questions about the ingested documents.

## Stack

| Layer | Choice |
|-------|--------|
| Web   | FastAPI + Uvicorn |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost) |
| Vector store | FAISS (persisted to `storage/faiss_index`) |
| LLM | Groq `llama-3.3-70b-versatile` |

## Configuration

Reads the project-root `.env`. Relevant variables:

```
GROQ_API_KEY=...          # required for chat
ADMIN_PASSWORD=admin123   # default; change this!
RAG_LLM_MODEL=llama-3.3-70b-versatile
RAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=150
RAG_TOP_K=4
```

## Run

```bash
./rag_app/run.sh
# or, from the project root:
uv run uvicorn rag_app.backend.main:app --host 0.0.0.0 --port 8000
```

Then open:
- Chat:  http://localhost:8000/
- Admin: http://localhost:8000/admin

## Flow

1. Log in on `/admin` with `ADMIN_PASSWORD`.
2. Upload a document → it is chunked, embedded, and stored in FAISS.
3. On `/`, ask questions → relevant chunks are retrieved and passed to the LLM, which answers with source citations.

The index persists to disk, so ingested documents survive restarts.
