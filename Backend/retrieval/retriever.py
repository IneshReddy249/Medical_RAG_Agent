import os, requests, chromadb
from functools import lru_cache
from chromadb.config import Settings
from Backend.Ingestion.embedder import MedicalEmbedder

class MedicalRetriever:
    def __init__(self, db_path="./chroma_db", collection="medical_rag"):
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
        self.col = self.client.get_collection(collection)
        self.embedder = MedicalEmbedder()
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.rerank_url = "https://api.together.xyz/v1/rerank"
        self.rerank_model = "togethercomputer/m2-bert-80M-8k-rerank"

    @lru_cache(maxsize=512)
    def _embed_query(self, query: str):
        return self.embedder._embed_batch([query])[0]

    def retrieve(self, query, top_k=10):
        q_emb = self._embed_query(query)
        res = self.col.query(query_embeddings=[q_emb], n_results=top_k, include=["documents","metadatas","distances"])
        docs, metas, dists = res.get("documents", [[]])[0], res.get("metadatas", [[]])[0], res.get("distances", [[]])[0]
        return [{"text": t, "meta": m, "score": 1 - d} for t, m, d in zip(docs, metas, dists)]

    def rerank(self, query, docs, top_k=5):
        if not self.api_key or not docs: return docs[:top_k]
        try:
            r = requests.post(self.rerank_url, headers={"Authorization": f"Bearer {self.api_key}"},
                              json={"model": self.rerank_model, "query": query,
                                    "documents": [d["text"] for d in docs], "top_n": top_k}, timeout=15)
            r.raise_for_status()
            for item in r.json().get("results", []):
                docs[item["index"]]["score"] = item["relevance_score"]
        except Exception:
            pass
        return sorted(docs, key=lambda x: x["score"], reverse=True)[:top_k]

    def query(self, question, top_k=5):
        docs = self.retrieve(question, top_k * 3)
        if self.api_key: docs = self.rerank(question, docs, top_k)
        return [{"text": d["text"], "score": round(d["score"], 3), "meta": d.get("meta", {})} for d in docs[:top_k]]
