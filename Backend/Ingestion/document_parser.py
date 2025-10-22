import os
from pathlib import Path
from typing import List, Optional
from llama_index.core import Document
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

def _pymupdf_pages(pdf: Path) -> List[str]:
    if fitz is None: return []
    try:
        d = fitz.open(str(pdf))
        return [(d[i].get_text("text") or "").strip() for i in range(len(d))]
    except Exception:
        return []

def _guess_page(text: str, pages: List[str]) -> Optional[int]:
    probe = " ".join((text or "")[:800].split()).lower()
    if not probe or not pages: return None
    for i, p in enumerate(pages):
        hay = " ".join((p or "").split()).lower()
        if probe and (probe in hay or hay in probe): return i + 1
    # fuzzy overlap
    pw = set(probe.split()[:80]); best = (None, 0.0)
    for i, p in enumerate(pages):
        hw = set((" ".join((p or "").split()).lower()).split())
        score = len(pw & hw) / (len(pw) or 1)
        if score > best[1]: best = (i, score)
    return (best[0] + 1) if best[0] is not None and best[1] >= 0.15 else None

class LlamaParsePDFLoader:
    """Use LlamaParse for text; stamp page numbers via PyMuPDF; fallback to PyMuPDF-only."""
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists(): raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    def _llama(self) -> List[Document]:
        key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not key: return []
        try:
            from llama_parse import LlamaParse
            parser = LlamaParse(
                api_key=key,
                result_type="markdown",
                system_prompt="You are a precise medical document parser.",
                user_prompt="Extract clinically relevant content with headings, lists and tables.",
                max_timeout=600,
            )
            docs = parser.load_data(self.pdf_path) or []
            for d in docs:
                d.metadata = {**(d.metadata or {}), "source": str(self.pdf_path), "type": "text", "parser": "llamaparse"}
            return docs
        except Exception:
            return []

    def _local(self) -> List[Document]:
        pages = _pymupdf_pages(self.pdf_path)
        out: List[Document] = []
        for i, t in enumerate(pages, 1):
            if not t: continue
            out.append(Document(text=t, metadata={"source": str(self.pdf_path), "parser": "pymupdf", "page_label": i, "type": "text"}))
        return out

    def load(self) -> List[Document]:
        docs = self._llama()
        if docs:
            pages = _pymupdf_pages(self.pdf_path)
            if pages:
                for d in docs:
                    md = d.metadata or {}
                    if not (md.get("page_label") or md.get("page") or md.get("page_number")):
                        pg = _guess_page(d.text, pages)
                        if pg: md["page_label"] = pg; d.metadata = md
            return docs
        fallback = self._local()
        if fallback: return fallback
        raise ValueError("No text extracted (LlamaParse + PyMuPDF failed).")
