#!/usr/bin/env bash
# Launch the RAG app. Run from the project root or from rag-app/.
set -e
cd "$(dirname "$0")/.."          # project root (where .venv and .env live)
exec uv run uvicorn rag_app.backend.main:app --host 0.0.0.0 --port 8000
