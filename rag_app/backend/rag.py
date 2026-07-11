"""Core RAG engine: ingestion, hybrid retrieval, reranking, and grounded chat.

The pipeline applies the techniques explored in the repo notebooks:

- Semantic chunking (3-AdvancedChunking) with a recursive fallback
- Hybrid dense + sparse retrieval: FAISS (MMR) + BM25, fused with
  Reciprocal Rank Fusion (4-Hybrid Search Statergies)
- Cross-encoder reranking of the fused candidates (4-Hybrid Search Statergies)
- LLM query enhancement: query expansion + HyDE hypothetical answer
  (5-Query Enhancement)
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from . import config

# A single lock guards mutations to the shared FAISS/BM25 indexes.
_lock = threading.Lock()


class RAGEngine:
    """Owns the embedding model, indexes, and the ingestion/query flows."""

    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.semantic_splitter = None
        if config.CHUNKING == "semantic":
            try:
                from langchain_experimental.text_splitter import SemanticChunker

                self.semantic_splitter = SemanticChunker(
                    self.embeddings, breakpoint_threshold_type="percentile"
                )
            except Exception:  # noqa: BLE001 - optional technique, fall back
                self.semantic_splitter = None
        self._reranker: Any = None
        self.vs: FAISS | None = None
        self.bm25: BM25Retriever | None = None
        self._load_index()
        self._rebuild_bm25()

    # -- Persistence ---------------------------------------------------------
    def _load_index(self) -> None:
        if config.INDEX_DIR.exists():
            self.vs = FAISS.load_local(
                str(config.INDEX_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

    def _save_index(self) -> None:
        if self.vs is not None:
            self.vs.save_local(str(config.INDEX_DIR))

    # -- Sparse index (BM25) ---------------------------------------------------
    def _all_chunks(self) -> list[Document]:
        if self.vs is None:
            return []
        return list(self.vs.docstore._dict.values())

    def _rebuild_bm25(self) -> None:
        chunks = self._all_chunks()
        self.bm25 = (
            BM25Retriever.from_documents(chunks, k=config.FETCH_K) if chunks else None
        )

    # -- Loading -------------------------------------------------------------
    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader: Any = PyMuPDFLoader(str(path))
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(str(path))
        elif suffix in (".txt", ".md"):
            loader = TextLoader(str(path), encoding="utf-8")
        elif suffix == ".csv":
            # One Document per row, with column names embedded in the text.
            loader = CSVLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        return loader.load()

    # -- Chunking --------------------------------------------------------------
    def _split(self, docs: list[Document]) -> list[Document]:
        """Semantic chunking when available; oversized semantic chunks are
        re-split recursively so no chunk blows past the embedding budget."""
        if self.semantic_splitter is None:
            return self.recursive_splitter.split_documents(docs)
        try:
            chunks = self.semantic_splitter.split_documents(docs)
        except Exception:  # noqa: BLE001 - degrade to the simple splitter
            return self.recursive_splitter.split_documents(docs)
        result: list[Document] = []
        for c in chunks:
            if len(c.page_content) > 2 * config.CHUNK_SIZE:
                result.extend(self.recursive_splitter.split_documents([c]))
            else:
                result.append(c)
        return result

    # -- Public API ----------------------------------------------------------
    def ingest(self, path: Path) -> int:
        """Load, chunk, embed, and index a document. Returns number of chunks added."""
        docs = self._load_file(path)
        for d in docs:
            d.metadata["source"] = path.name
        chunks = self._split(docs)
        if not chunks:
            return 0
        with _lock:
            if self.vs is None:
                self.vs = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vs.add_documents(chunks)
            self._save_index()
            self._rebuild_bm25()
        return len(chunks)

    def sources(self) -> list[str]:
        """Return the distinct document names currently in the index."""
        names = {d.metadata.get("source", "unknown") for d in self._all_chunks()}
        return sorted(names)

    def is_empty(self) -> bool:
        return self.vs is None

    # -- Retrieval -------------------------------------------------------------
    def _dense(self, query: str) -> list[Document]:
        """Dense retrieval with Maximal Marginal Relevance for diverse results."""
        if self.vs is None:
            return []
        return self.vs.max_marginal_relevance_search(
            query, k=config.FETCH_K, fetch_k=config.FETCH_K * 3
        )

    def _sparse(self, query: str) -> list[Document]:
        if self.bm25 is None:
            return []
        return self.bm25.invoke(query)

    def hybrid_retrieve(self, query: str) -> list[Document]:
        """Dense (MMR) + sparse (BM25) candidates fused with weighted RRF."""
        return _rrf_fuse(
            [self._dense(query), self._sparse(query)],
            weights=[config.DENSE_WEIGHT, 1.0 - config.DENSE_WEIGHT],
        )

    # -- Reranking ---------------------------------------------------------------
    def rerank(self, question: str, docs: list[Document], k: int) -> list[Document]:
        """Cross-encoder reranking; falls back to fused order if unavailable."""
        if len(docs) <= k:
            return docs
        try:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(config.RERANK_MODEL)
            scores = self._reranker.predict(
                [(question, d.page_content) for d in docs]
            )
            ranked = sorted(zip(scores, docs), key=lambda p: -p[0])
            return [d for _, d in ranked[:k]]
        except Exception:  # noqa: BLE001 - reranker is an enhancement, not a hard dep
            return docs[:k]


def _rrf_fuse(
    ranked_lists: list[list[Document]], weights: list[float], c: int = 60
) -> list[Document]:
    """Weighted Reciprocal Rank Fusion over multiple ranked candidate lists."""
    scored: dict[str, tuple[float, Document]] = {}
    for docs, w in zip(ranked_lists, weights):
        for rank, d in enumerate(docs):
            key = d.page_content
            prev = scored.get(key, (0.0, d))
            scored[key] = (prev[0] + w / (c + rank + 1), prev[1])
    return [d for _, d in sorted(scored.values(), key=lambda p: -p[0])]


# --- Chat (LLM) --------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided context from uploaded documents"
)

_ENHANCE_PROMPT = """You improve search queries for a document retrieval system.

User question: "{question}"

Reply with exactly two lines:
EXPANDED: the question rewritten with synonyms and related terms (query expansion)
HYPOTHETICAL: a short one-paragraph hypothetical answer to the question (HyDE)"""


def build_llm():
    from langchain_groq import ChatGroq

    return ChatGroq(model=config.LLM_MODEL, temperature=0.2, api_key=config.GROQ_API_KEY)


def enhance_query(llm, question: str) -> dict[str, str]:
    """Query expansion + HyDE in a single LLM call. Empty dict on any failure."""
    try:
        text = llm.invoke(_ENHANCE_PROMPT.format(question=question)).content
        out: dict[str, str] = {}
        for line in text.splitlines():
            if line.startswith("EXPANDED:"):
                out["expanded"] = line[len("EXPANDED:"):].strip()
            elif line.startswith("HYPOTHETICAL:"):
                out["hypothetical"] = line[len("HYPOTHETICAL:"):].strip()
        return out
    except Exception:  # noqa: BLE001 - enhancement must never break the chat
        return {}


def answer(engine: RAGEngine, question: str) -> dict[str, Any]:
    """Run the full pipeline: enhance -> hybrid retrieve -> rerank -> generate."""
    if engine.is_empty():
        return {
            "answer": "No documents have been ingested yet. Please ask an admin to upload documents.",
            "sources": [],
        }

    llm = build_llm()

    # 1. Query enhancement: expansion + HyDE (one LLM call, best-effort).
    enhanced = enhance_query(llm, question) if config.QUERY_ENHANCEMENT else {}

    # 2. Hybrid retrieval for each query variant, fused with RRF.
    candidate_lists = [engine.hybrid_retrieve(question)]
    if enhanced.get("expanded"):
        candidate_lists.append(engine.hybrid_retrieve(enhanced["expanded"]))
    if enhanced.get("hypothetical"):
        # HyDE: the hypothetical answer lives in "answer space", so dense-only.
        candidate_lists.append(engine._dense(enhanced["hypothetical"]))
    candidates = _rrf_fuse(candidate_lists, weights=[1.0] * len(candidate_lists))

    # 3. Cross-encoder reranking down to the final context window.
    docs = engine.rerank(question, candidates, k=config.TOP_K)

    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )
    used_sources = sorted({d.metadata.get("source", "unknown") for d in docs})

    # 4. Grounded generation with citations.
    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    resp = llm.invoke(messages)
    return {"answer": resp.content, "sources": used_sources}
