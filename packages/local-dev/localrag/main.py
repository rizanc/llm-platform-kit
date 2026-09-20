"""FastAPI app — local-first RAG over your PDFs."""
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import logging

from .config import settings
from .ingest import ingest_pdf
from .rag import RAG

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Local-First RAG", version="0.1.0")
rag = RAG(settings.lancedb_uri, settings.ollama_host)


class Query(BaseModel):
    q: str
    k: int = 4


@app.get("/health")
def health():
    return {"status": "ok", "ollama": settings.ollama_host, "embed": settings.embed_model, "llm": settings.llm_model}


@app.get("/stats")
def stats():
    return rag.stats()


@app.post("/ingest")
async def ingest(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF supported")
    path = Path(settings.docs_dir) / file.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(await file.read())
    n = ingest_pdf(path, rag)
    return JSONResponse({"file": file.filename, "chunks": n})


@app.post("/query")
def query(q: Query):
    hits = rag.retrieve(q.q, k=q.k)
    if not hits:
        return {"answer": "No documents indexed. POST /ingest first.", "sources": []}
    answer = rag.generate(q.q, hits)
    return {"answer": answer, "sources": [{"page": h["page"], "text": h["text"][:200]} for h in hits]}