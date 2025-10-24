from __future__ import annotations
import os, logging
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from Backend.Ingestion.embedder import MedicalEmbedder
from Backend.Ingestion.vector_store import MedicalChromaStore
from Backend.retrieval.retriever import MedicalRetriever
from Backend.generator.generator import MedicalGenerator


# === ENV SETUP ===
load_dotenv(find_dotenv(usecwd=True), override=True)
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

os.environ.update({
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
})

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DB_PATH, COLLECTION = "./chroma_db", "medical_rag"
UPLOADS_DIR = Path("Backend/data/uploads")
REQ_ENV = ["TOGETHER_API_KEY"]

app = FastAPI(title="Medical RAG Agent", version="1.0.0")


# === UTILS ===
def ensure_env(keys: List[str]):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise HTTPException(400, f"Missing env vars: {missing}")


def ensure_dirs():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    Path(DB_PATH).mkdir(parents=True, exist_ok=True)


def load_pdf(pdf_path: Path) -> List[Document]:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        raise HTTPException(500, "Missing dependency: pip install pymupdf")

    if not pdf_path.exists():
        raise HTTPException(404, f"PDF not found: {pdf_path}")

    with fitz.open(str(pdf_path)) as pdf:
        return [
            Document(text=p.get_text("text").strip(), metadata={"source": str(pdf_path), "page_label": i + 1})
            for i, p in enumerate(pdf) if p.get_text("text").strip()
        ]


# === SCHEMAS ===
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    include_context: bool = True


class QueryResponse(BaseModel):
    answer: str
    contexts: Optional[List[Dict[str, Any]]] = None


# === ROUTES ===
@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): return PlainTextResponse("", status_code=204)


@app.get("/")
def root(): return {"service": "Medical RAG Agent", "docs": "/docs"}


@app.get("/readyz")
def readyz():
    ready = UPLOADS_DIR.exists() and Path(DB_PATH).exists()
    missing = [k for k in REQ_ENV if not os.getenv(k)]
    return {"ready": ready and not missing, "missing_env": missing}


@app.get("/stats")
def stats():
    t0 = perf_counter()
    try:
        count = MedicalChromaStore(COLLECTION, DB_PATH).col.count()
    except Exception:
        count = -1
    return {
        "db": DB_PATH,
        "collection": COLLECTION,
        "vector_count": count,
        "ping_ms": int((perf_counter() - t0) * 1000),
    }


@app.post("/ingest_path")
def ingest_path(pdf_path: str):
    """Ingest a local PDF, embed, and store vectors in Chroma."""
    ensure_dirs()
    ensure_env(REQ_ENV)

    docs = load_pdf(Path(pdf_path))
    nodes: List[TextNode] = SentenceSplitter(chunk_size=440, chunk_overlap=40).get_nodes_from_documents(docs)
    nodes = MedicalEmbedder().embed_nodes(nodes)
    store = MedicalChromaStore(COLLECTION, DB_PATH)
    stored = store.store_nodes(nodes)

    return {
        "filename": Path(pdf_path).name,
        "docs_count": len(docs),
        "nodes_count": len(nodes),
        "stored_vectors": stored,
        "collection_count": store.col.count(),
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest = Body(...)):
    """Run retrieval + generation to answer a question."""
    ensure_env(REQ_ENV)

    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty.")

    retriever = MedicalRetriever(db_path=DB_PATH, collection=COLLECTION)
    context = retriever.query(question, top_k=req.top_k)
    generator = MedicalGenerator(db_path=DB_PATH, collection=COLLECTION)
    answer = await generator.generate_async(question, top_k=max(1, len(context)))

    return {"answer": answer, "contexts": context if req.include_context else None}
