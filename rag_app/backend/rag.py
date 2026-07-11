"""Core RAG engine: document ingestion, vector store, and retrieval-augmented chat."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from . import config

# A single lock guards mutations to the shared FAISS index.
_lock = threading.Lock()


class RAGEngine:
    """Owns the embedding model, FAISS index, and the ingestion/query flows."""

    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.vs: FAISS | None = None
        self._load_index()

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

    # -- Loading -------------------------------------------------------------
    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader: Any = PyMuPDFLoader(str(path))
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(str(path))
        elif suffix in (".txt", ".md"):
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        return loader.load()

    # -- Public API ----------------------------------------------------------
    def ingest(self, path: Path) -> int:
        """Load, chunk, embed, and index a document. Returns number of chunks added."""
        docs = self._load_file(path)
        for d in docs:
            d.metadata["source"] = path.name
        chunks = self.splitter.split_documents(docs)
        if not chunks:
            return 0
        with _lock:
            if self.vs is None:
                self.vs = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vs.add_documents(chunks)
            self._save_index()
        return len(chunks)

    def sources(self) -> list[str]:
        """Return the distinct document names currently in the index."""
        if self.vs is None:
            return []
        names = {
            d.metadata.get("source", "unknown")
            for d in self.vs.docstore._dict.values()
        }
        return sorted(names)

    def is_empty(self) -> bool:
        return self.vs is None

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        if self.vs is None:
            return []
        return self.vs.similarity_search(query, k=k or config.TOP_K)


# --- Chat (LLM) --------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided context from uploaded documents"
)


def build_llm():
    from langchain_groq import ChatGroq

    return ChatGroq(model=config.LLM_MODEL, temperature=0.2, api_key=config.GROQ_API_KEY)


def answer(engine: RAGEngine, question: str) -> dict[str, Any]:
    """Retrieve context and generate an answer. Returns answer + sources."""
    if engine.is_empty():
        return {
            "answer": "No documents have been ingested yet. Please ask an admin to upload documents.",
            "sources": [],
        }

    docs = engine.retrieve(question)
    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )
    used_sources = sorted({d.metadata.get("source", "unknown") for d in docs})

    llm = build_llm()
    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    resp = llm.invoke(messages)
    return {"answer": resp.content, "sources": used_sources}
