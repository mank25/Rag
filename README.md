# RAG — Learning Notebooks & App

A hands-on exploration of Retrieval-Augmented Generation (RAG), built in two parts:

1. **Learning notebooks** (`0-` through `6-`) — a step-by-step progression through the RAG pipeline, from data ingestion to multimodal retrieval.
2. **RAG App** (`rag_app/`) — a working web application that puts the notebook techniques into practice: semantic chunking, hybrid dense+sparse search, reranking, query enhancement, and grounded chat with citations.

## Repository layout

| Folder | Topic |
|--------|-------|
| `0-DataIngestParsing/` | Loading and parsing data sources: PDF, DOCX, CSV/Excel, JSON, and databases |
| `1-VectorEmbeddingsAndDatabases/` | Text embeddings with local (HuggingFace) and OpenAI models |
| `2-Vector Stores/` | Vector databases: ChromaDB, FAISS, and others |
| `3-AdvancedChunking/` | Semantic chunking strategies |
| `4-Hybrid Search Statergies/` | Dense + sparse (BM25) retrieval, reranking, and MMR |
| `5-Query Enhancement/` | Query expansion, query decomposition, and HyDE |
| `6-multimodal/` | Multimodal RAG over PDFs with images (OpenAI) |
| `rag_app/` | The complete RAG web application (see below) |

## Getting started

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # install dependencies
```

Create a `.env` in the project root with the API keys you need:

```env
GROQ_API_KEY=...                 # required for the app's chat
OPENAI_API_KEY=...               # used by some notebooks (OpenAI embeddings, multimodal)
ADMIN_PASSWORD=change-me         # app admin login; defaults to admin123
```

The notebooks can then be run with the project's environment as the Jupyter kernel (`ipykernel` is included).

---

# RAG App (`rag_app/`)

A RAG web application that applies the techniques from the notebooks end to end. Admins upload documents through a password-protected panel; anyone can then chat with an LLM that answers questions grounded in those documents, with source citations. An **About page** (`/about`) explains RAG and every technique in the pipeline.

## Pipelines

**Ingestion** (admin uploads a document):

```
Load & parse → Semantic chunking → Embed (MiniLM, local) → Index (FAISS + BM25)
```

**Query** (anyone asks a question):

```
Query enhancement (expansion + HyDE) → Hybrid retrieval (FAISS-MMR + BM25)
    → Reciprocal Rank Fusion → Cross-encoder reranking → Grounded generation
```

## Techniques applied

| Technique | From notebook | How it's used |
|-----------|---------------|---------------|
| Semantic chunking | `3-AdvancedChunking` | Splits at topic shifts using embedding similarity; oversized chunks re-split recursively |
| Hybrid dense + sparse search | `4-Hybrid Search Statergies` | FAISS vector search and BM25 keyword search run in parallel |
| MMR (Maximal Marginal Relevance) | `4-Hybrid Search Statergies` | Dense retrieval favors relevant *and* diverse chunks |
| Reciprocal Rank Fusion | `4-Hybrid Search Statergies` | Merges ranked lists from all retrievers and query variants by position |
| Reranking | `4-Hybrid Search Statergies` | A local cross-encoder rescores fused candidates before the LLM sees them |
| Query expansion | `5-Query Enhancement` | The LLM rewrites the question with synonyms/related terms; both versions are searched |
| HyDE | `5-Query Enhancement` | The LLM drafts a hypothetical answer, which is embedded and used for dense search |
| Grounded generation | — | The LLM answers strictly from retrieved context, citing source documents |

## Stack

| Layer | Choice |
|-------|--------|
| Web framework | FastAPI + Uvicorn |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | FAISS (persisted to disk) + BM25 sparse index |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Frontend | Plain HTML/JS (`rag_app/frontend/`) |

## Run

```bash
./rag_app/run.sh
# or, from the project root:
uv run uvicorn rag_app.backend.main:app --host 0.0.0.0 --port 8000
```

Then open:

- **Chat:** http://localhost:8000/
- **Admin:** http://localhost:8000/admin
- **About:** http://localhost:8000/about

## Usage

1. Open `/admin` and log in with your `ADMIN_PASSWORD`.
2. Upload a document (PDF, DOCX, DOC, TXT, MD, CSV) — it is semantically chunked (CSV: one chunk per row), embedded locally, and indexed in FAISS and BM25.
3. Open `/` and ask questions — the full query pipeline runs and the LLM answers with source citations.

## Configuration

All settings are read from the project-root `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | Required for chat responses |
| `ADMIN_PASSWORD` | `admin123` | Admin panel login (change this!) |
| `RAG_LLM_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `RAG_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `RAG_CHUNKING` | `semantic` | Chunking strategy: `semantic` or `recursive` |
| `RAG_CHUNK_SIZE` | `1000` | Character budget per chunk (recursive / oversize re-split) |
| `RAG_CHUNK_OVERLAP` | `150` | Overlap between recursive chunks |
| `RAG_FETCH_K` | `12` | Candidates fetched per retriever before fusion/reranking |
| `RAG_DENSE_WEIGHT` | `0.5` | Dense retriever weight in fusion (sparse gets the rest) |
| `RAG_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `RAG_TOP_K` | `4` | Chunks passed to the LLM after reranking |
| `RAG_QUERY_ENHANCEMENT` | `1` | Set `0` to disable query expansion + HyDE |

## API

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/admin/login` | — | Exchange password for a token |
| `POST` | `/api/admin/ingest` | `X-Admin-Password` header | Upload and index a document |
| `GET` | `/api/admin/sources` | `X-Admin-Password` header | List ingested documents |
| `POST` | `/api/chat` | — | Ask a question (`{"message": "..."}`) |
| `GET` | `/api/status` | — | Index readiness and document list |

## App structure

```
rag_app/
├── backend/
│   ├── main.py      # FastAPI routes (admin ingestion + public chat + pages)
│   ├── rag.py       # RAG engine: chunking, hybrid retrieval, fusion, reranking, chat
│   └── config.py    # Environment-based configuration
├── frontend/
│   ├── chat.html    # Public chat UI
│   ├── admin.html   # Admin upload panel
│   └── about.html   # Explains RAG and the pipeline techniques
├── storage/         # Uploads and persisted FAISS index (gitignored data)
└── run.sh           # Launch script
```

## Notes

- Embeddings, chunking, and reranking all run **locally** via sentence-transformers — only query enhancement and the final answer call the Groq API.
- The FAISS index persists to `rag_app/storage/faiss_index`; the BM25 index is rebuilt from it at startup, so ingested documents survive restarts.
- The cross-encoder and embedding models are downloaded from HuggingFace on first use.
