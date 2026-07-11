# RAG App

A Retrieval-Augmented Generation application that applies the techniques from this repo's notebooks end to end:

- **Admin panel** (`/admin`) — password-protected document ingestion (PDF, DOCX, TXT, MD).
- **Public chat** (`/`) — anyone can ask questions about the ingested documents.
- **About page** (`/about`) — explains RAG and every technique in the pipeline.

## Pipelines

**Ingestion:** load & parse → semantic chunking (recursive fallback) → local MiniLM embeddings → FAISS + BM25 indexing.

**Query:** LLM query enhancement (expansion + HyDE) → hybrid retrieval (FAISS with MMR + BM25) → Reciprocal Rank Fusion → cross-encoder reranking → grounded generation with citations.

## Stack

| Layer | Choice |
|-------|--------|
| Web   | FastAPI + Uvicorn |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost) |
| Vector store | FAISS (persisted to `storage/faiss_index`) + BM25 sparse index |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| LLM | Groq `llama-3.3-70b-versatile` |

## Configuration

Reads the project-root `.env`. Relevant variables:

```
GROQ_API_KEY=...          # required for chat
ADMIN_PASSWORD=admin123   # default; change this!
RAG_LLM_MODEL=llama-3.3-70b-versatile
RAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_CHUNKING=semantic     # or "recursive"
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=150
RAG_FETCH_K=12            # candidates per retriever before fusion/reranking
RAG_DENSE_WEIGHT=0.5      # dense weight in hybrid fusion
RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_TOP_K=4               # chunks passed to the LLM after reranking
RAG_QUERY_ENHANCEMENT=1   # set 0 to disable expansion + HyDE
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
- About: http://localhost:8000/about

## Flow

1. Log in on `/admin` with `ADMIN_PASSWORD`.
2. Upload a document → it is semantically chunked, embedded locally, and indexed in FAISS and BM25.
3. On `/`, ask questions → the query is enhanced (expansion + HyDE), candidates are retrieved by hybrid search, fused with RRF, reranked by a cross-encoder, and the LLM answers strictly from the winning chunks with source citations.

The FAISS index persists to disk (the BM25 index is rebuilt from it at startup), so ingested documents survive restarts. Embeddings, chunking, and reranking run locally; only query enhancement and the final answer call the Groq API.
