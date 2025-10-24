from __future__ import annotations
import os, hashlib, logging
from typing import List
import chromadb
from chromadb.config import Settings
from llama_index.core.schema import TextNode


def _norm(text: str) -> str:
    """Normalize whitespace."""
    return " ".join((text or "").split())


def _cid(text: str) -> str:
    """Content hash for deduplication."""
    return hashlib.sha1(_norm(text).encode()).hexdigest()


def _skip(text: str) -> bool:
    """Ignore trivial or short text segments."""
    return len(text) < 160 and len(text.split()) < 25


class MedicalChromaStore:
    """Handles storage and deduplication of embeddings in ChromaDB."""

    def __init__(self, collection: str = "medical_rag", path: str = "./chroma_db"):
        os.makedirs(path, exist_ok=True)
        client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        self.col = client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        logging.info(f"🩺 ChromaDB ready: {collection} @ {path}")

    def store_nodes(self, nodes: List[TextNode], batch: int = 100) -> int:
        """Store unique nodes with embeddings into ChromaDB."""
        if not nodes:
            return 0

        records, seen = [], set()
        for n in nodes:
            emb, text = getattr(n, "embedding", None), _norm(getattr(n, "text", "") or "")
            if not emb or not text or _skip(text):
                continue

            cid = _cid(text)
            _id = f"{n.metadata.get('source', 'unknown')}::{cid}"
            if _id in seen:
                continue
            seen.add(_id)
            meta = {**(n.metadata or {}), "cid": cid}
            records.append((_id, emb, text, meta))

        if not records:
            logging.info("No new nodes to upsert.")
            return 0

        stored = 0
        for i in range(0, len(records), batch):
            ids, embs, docs, metas = zip(*{r[0]: r for r in records[i:i+batch]}.values())
            self.col.upsert(
                ids=list(ids),
                embeddings=list(embs),
                documents=list(docs),
                metadatas=list(metas),
            )
            stored += len(ids)
            logging.info(f"Stored batch {i//batch + 1}: {len(ids)} items")

        logging.info(f"✅ Total stored: {stored} | Collection size: {self.col.count()}")
        return stored
