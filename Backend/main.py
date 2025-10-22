# Backend/main.py
from __future__ import annotations
import os, re, hashlib, logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from time import perf_counter

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from Backend.Ingestion.document_parser import LlamaParsePDFLoader
from Backend.Ingestion.semantic_chunker import SemanticChunker
from Backend.Ingestion.embedder import MedicalEmbedder
from Backend.Ingestion.vector_store import MedicalChromaStore
from Backend.retrieval.retriever import MedicalRetriever
from Backend.generator.generator import MedicalGenerator

# ---------- App/bootstrap ----------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

APP_API_KEY = os.getenv("APP_API_KEY", "").strip()
_ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "FRONTEND_ORIGINS",
    "http://127.0.0.1:5173,http://localhost:5173"
).split(",") if o.strip()]

app = FastAPI(title="Medical RAG Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "X-API-Key"],
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DATA_DIR, UPLOADS_DIR = ROOT / "data", ROOT / "data" / "uploads"
app.state.db_path, app.state.collection = "./chroma_db", "medical_rag"
REQ_PARSE, REQ_GEN = ["LLAMA_CLOUD_API_KEY"], ["TOGETHER_API_KEY"]

# ---------- Security & Safety ----------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
def require_key(key: Optional[str] = Security(api_key_header)):
    if APP_API_KEY and (key or "").strip() != APP_API_KEY:
        raise HTTPException(401, "Invalid API key")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.middleware("http")
async def _rate_limit_mw(request, call_next):
    return await limiter._middleware(request, call_next)

@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Rate limit exceeded", status_code=429)

_BAD = re.compile(
    r"(ignore (all|previous) instructions|system:|developer:|you are now|"
    r"onerror=|<script|</script>|data:text/html|javascript:|base64,|"
    r"pretend to|bypass|jailbreak|do anything now)",
    re.I
)
_URL = re.compile(r"https?://", re.I)

PII_PATTERNS = [
    re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
    re.compile(r'\b\d{10}\b'),
    re.compile(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}', re.I),
]
BANNED_INTENT = [
    "diagnose", "diagnosis", "what is wrong with me",
    "prescribe", "prescription", "dose", "dosage", "titrate",
    "start medication", "stop medication", "emergency advice",
]
DISCLAIMER = ("Educational use only. Not medical advice. "
              "For diagnosis or treatment, consult a licensed clinician.")

def redact_pii(s: str) -> str:
    out = s or ""
    for pat in PII_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out

def looks_clinical_instruction(q: str) -> bool:
    x = (q or "").lower()
    return any(k in x for k in BANNED_INTENT)

# ---------- Helpers ----------
def _need_env(keys: List[str]):
    miss = [k for k in keys if not os.getenv(k)]
    if miss: raise HTTPException(400, f"Missing env vars: {miss}")

def _ens_dirs():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    Path(app.state.db_path).mkdir(parents=True, exist_ok=True)

def _dedup(items: List[dict]) -> List[dict]:
    seen, out = set(), []
    for d in items:
        t = " ".join((d.get("text") or "").split())
        k = hashlib.sha1(t[:250].encode()).hexdigest()
        if k in seen: continue
        seen.add(k); out.append(d)
    return out

def _sanitize(items: List[dict], k: int = 5) -> List[dict]:
    clean = []
    for d in items:
        t = (d.get("text") or "").strip()
        if not t or _BAD.search(t): 
            continue
        if sum(1 for _ in _URL.finditer(t)) > 5:
            continue
        t = re.sub(r"<[^>]+>", "", t)
        d["text"] = t
        clean.append(d)
        if len(clean) >= k: break
    return clean

# ---------- Models ----------
class IngestResponse(BaseModel):
    filename: str; docs_count: int; nodes_count: int; stored_vectors: int; collection_count: int

class QueryRequest(BaseModel):
    question: str; top_k: int = 5; include_context: bool = True

class QueryResponse(BaseModel):
    answer: str; contexts: Optional[List[dict]] = None; citations: Optional[List[dict]] = None

# ---------- Lazy singletons ----------
def _store() -> MedicalChromaStore:
    if not hasattr(app.state, "store"):
        app.state.store = MedicalChromaStore(app.state.collection, app.state.db_path)
    return app.state.store

def _retriever() -> MedicalRetriever:
    if not hasattr(app.state, "retriever"):
        app.state.retriever = MedicalRetriever(db_path=app.state.db_path, collection=app.state.collection)
    return app.state.retriever

def _generator() -> MedicalGenerator:
    if not hasattr(app.state, "generator"):
        app.state.generator = MedicalGenerator(db_path=app.state.db_path, collection=app.state.collection)
    return app.state.generator

# ---------- Routes ----------
@app.get("/")
def root():
    return {"name":"Medical RAG Agent","db":app.state.db_path,"collection":app.state.collection,"docs":"/docs"}

@app.get("/health")
def health(): 
    return {"status":"ok"}

@app.get("/readyz")
def readyz():
    miss = [k for k in REQ_PARSE + REQ_GEN if not os.getenv(k)]
    return {"ready": UPLOADS_DIR.exists() and Path(app.state.db_path).exists() and not miss, "missing_env": miss}

@app.get("/stats")
def stats():
    t0 = perf_counter()
    try:
        _ = _retriever().col.count()
    except:
        pass
    ping = int((perf_counter()-t0)*1000)
    return {"db": app.state.db_path, "collection": app.state.collection, "vector_count": _store().collection.count(), "ping_ms": ping}

def _ingest_pdf(p: Path) -> IngestResponse:
    docs = LlamaParsePDFLoader(str(p)).load()
    if not docs: raise HTTPException(400, "No text extracted")
    nodes = SemanticChunker(440, 40).chunk(docs)
    nodes = MedicalEmbedder("BAAI/bge-large-en-v1.5", 50).embed_nodes(nodes)
    st = _store(); stored = st.store_nodes(nodes, batch_size=100)
    return IngestResponse(filename=p.name, docs_count=len(docs), nodes_count=len(nodes), stored_vectors=stored, collection_count=st.collection.count())

@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("5/minute")
async def ingest(request: Request, file: UploadFile = File(...), _=Security(require_key)):
    _ens_dirs(); _need_env(REQ_PARSE + REQ_GEN)
    data = await file.read()
    if not data: raise HTTPException(400, "Empty file")
    dest = UPLOADS_DIR / Path(file.filename).name
    dest.write_bytes(data); await file.close()
    try:
        return _ingest_pdf(dest)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")

@app.post("/ingest_path", response_model=IngestResponse)
@limiter.limit("5/minute")
def ingest_path(request: Request, pdf_path: str = str(DATA_DIR / "Full.pdf"), _=Security(require_key)):
    _ens_dirs(); _need_env(REQ_PARSE + REQ_GEN)
    p = Path(pdf_path)
    if not p.exists(): raise HTTPException(404, f"PDF not found: {p}")
    try:
        return _ingest_pdf(p)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")

@app.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(request: Request, req: QueryRequest = Body(...), _=Security(require_key)):
    _need_env(REQ_GEN)

    q_raw = (req.question or "").strip()
    if not q_raw: 
        raise HTTPException(400, "Question is empty")
    if looks_clinical_instruction(q_raw):
        return {"answer": f"I can’t help with diagnosis, dosing, or treatment. {DISCLAIMER}"}

    q = redact_pii(q_raw)

    raw = _retriever().query(q, top_k=req.top_k)
    ctx = _dedup(_sanitize(raw))
    if not ctx or max(c.get("score", 0) for c in ctx) < 0.55:
        return {"answer": f"Not enough information to answer reliably. {DISCLAIMER}", "contexts": ctx}

    ans = await _generator().generate_async(question=q, top_k=len(ctx))
    if DISCLAIMER not in ans:
        ans = ans.rstrip() + f"\n\n— {DISCLAIMER}"

    payload = {"answer": ans, "contexts": ctx}
    if req.include_context:
        payload["citations"] = [
            {"id": i+1,
             "page": (m.get("page_label") or m.get("page") or m.get("page_number") or m.get("seq") or i+1),
             "section": (m.get("section_header") or m.get("heading") or m.get("title"))}
            for i, c in enumerate(ctx) for m in [(c.get("meta", {}) or {})]
        ]
    return JSONResponse(content=payload)

# ---
