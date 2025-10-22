import os, logging, hashlib
from typing import List
import chromadb
from chromadb.config import Settings
from llama_index.core.schema import TextNode

def _norm(t: str) -> str: return " ".join((t or "").split())
def _cid(text: str) -> str: return hashlib.sha1(_norm(text).encode()).hexdigest()
def _too_small(t: str) -> bool: return (len(t) < 200) and (len(t.split()) < 30)

class MedicalChromaStore:
    def __init__(self, collection_name="medical_rag", persist_path="./chroma_db"):
        os.makedirs(persist_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_path, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        logging.info(f"🩺 ChromaDB Medical Store: {collection_name} at {persist_path}")

    def _exists(self, cid: str, source: str) -> bool:
        try:
            res = self.collection.get(where={"cid": cid, "source": source}, include=["ids"], limit=1)
            return bool(res and res.get("ids"))
        except Exception:
            return False

    def store_nodes(self, nodes: List[TextNode], batch_size: int = 100) -> int:
        if not nodes: return 0
        recs = []
        for n in nodes:
            if not getattr(n, "embedding", None): continue
            text = _norm(n.text)
            if _too_small(text): continue
            meta = dict(n.metadata or {})
            src  = meta.setdefault("source", meta.get("source", "unknown"))
            cid  = _cid(text); meta["cid"] = cid
            _id  = f"{src}::{cid}"
            if not self._exists(cid, src):
                recs.append((_id, n.embedding, text, meta))
        if not recs:
            logging.info("✓ Nothing new to store"); return 0

        stored = 0
        for i in range(0, len(recs), batch_size):
            b = recs[i:i+batch_size]; ids, embs, docs, metas = zip(*b)
            self.collection.upsert(ids=list(ids), embeddings=list(embs), documents=list(docs), metadatas=list(metas))
            stored += len(b)
            logging.info(f"✓ Stored batch {i//batch_size+1}/{(len(recs)-1)//batch_size+1}: {len(b)}")
        logging.info(f"✅ Stored {stored} new embeddings; 📊 Total: {self.collection.count()}")
        return stored
