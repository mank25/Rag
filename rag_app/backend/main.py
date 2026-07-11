"""FastAPI app: admin document ingestion + public RAG chat."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .rag import RAGEngine, answer

app = FastAPI(title="RAG App")

# The engine (embedding model + FAISS index) is loaded once at startup.
engine: RAGEngine | None = None

ALLOWED_EXT = {".pdf", ".docx", ".doc", ".txt", ".md"}


@app.on_event("startup")
def _startup() -> None:
    global engine
    engine = RAGEngine()


def get_engine() -> RAGEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return engine


# --- Auth --------------------------------------------------------------------
def require_admin(x_admin_password: str = Header(default="")) -> None:
    if x_admin_password != config.ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin password")


class LoginRequest(BaseModel):
    password: str


@app.post("/api/admin/login")
def login(req: LoginRequest) -> dict:
    if req.password != config.ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    # The password itself is used as the bearer token for subsequent calls.
    return {"token": req.password}


# --- Admin: ingestion --------------------------------------------------------
@app.post("/api/admin/ingest", dependencies=[Depends(require_admin)])
async def ingest(file: UploadFile = File(...), eng: RAGEngine = Depends(get_engine)) -> dict:
    name = file.filename or "upload"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}",
        )

    dest = config.UPLOAD_DIR / name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = eng.ingest(dest)
    except Exception as e:  # noqa: BLE001 - surface any loader/embed error to the UI
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {"filename": name, "chunks_added": chunks, "sources": eng.sources()}


@app.get("/api/admin/sources", dependencies=[Depends(require_admin)])
def list_sources(eng: RAGEngine = Depends(get_engine)) -> dict:
    return {"sources": eng.sources()}


# --- Public: chat ------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest, eng: RAGEngine = Depends(get_engine)) -> dict:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    return answer(eng, req.message)


@app.get("/api/status")
def status(eng: RAGEngine = Depends(get_engine)) -> dict:
    return {"documents": eng.sources(), "ready": not eng.is_empty()}


# --- Frontend ----------------------------------------------------------------
@app.get("/")
def root() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "chat.html")


@app.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "admin.html")


@app.get("/about")
def about_page() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "about.html")


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")
