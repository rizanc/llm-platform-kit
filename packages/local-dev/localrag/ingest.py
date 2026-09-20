"""PDF → chunks → embeddings → LanceDB."""
from pathlib import Path
import logging
from pypdf import PdfReader

from .rag import RAG

log = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Char-based chunking; simple but predictable. Project #1 will swap in token-based."""
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_pdf(path: Path, rag: RAG) -> int:
    reader = PdfReader(str(path))
    rows = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in chunk_text(text):
            rows.append({"text": chunk, "page": page_no, "source": path.name})
    if not rows:
        log.warning("No extractable text in %s", path)
        return 0
    rag.add(rows)
    log.info("Ingested %s → %d chunks across %d pages", path.name, len(rows), len(reader.pages))
    return len(rows)