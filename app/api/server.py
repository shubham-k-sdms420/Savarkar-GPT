"""
Savarkar GPT - FastAPI Backend Server.

Exposes the RAG pipeline as a REST API for the frontend.
Serves the frontend static files and provides the /api/query endpoint.
"""

import os
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: pre-load heavy models once at startup
# ---------------------------------------------------------------------------
_rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load RAG chain (embedding model + Pinecone) once at startup."""
    global _rag_chain
    print("\n--- Loading RAG chain (embedding model + Pinecone index) ---")
    start = time.time()
    from app.rag.chain import RAGChain
    _rag_chain = RAGChain()
    elapsed = time.time() - start
    print(f"--- RAG chain ready in {elapsed:.1f}s ---\n")
    yield
    _rag_chain = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Savarkar GPT API",
    description="Historical Q&A about Vinayak Damodar Savarkar",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceInfo(BaseModel):
    title: str
    author: str
    chapter: str
    relevance_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]
    num_passages_retrieved: int


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Run a RAG query and return the historian's answer with sources."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if _rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain is still loading.")

    try:
        result = _rag_chain.query(
            question=req.question.strip(),
            top_k=req.top_k,
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": settings.LLM_MODEL_NAME,
        "index": settings.PINECONE_INDEX_NAME,
        "ready": _rag_chain is not None,
    }


@app.get("/api/usage")
async def usage():
    """Return token usage statistics from the JSONL log."""
    from app.monitoring.token_logger import get_usage_summary

    return get_usage_summary()


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------
_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"

if _frontend_dir.exists():
    app.mount("/css", StaticFiles(directory=_frontend_dir / "css"), name="css")
    app.mount("/js", StaticFiles(directory=_frontend_dir / "js"), name="js")

    @app.get("/")
    async def serve_index():
        return FileResponse(_frontend_dir / "index.html")
