from __future__ import annotations
import os, chromadb
from typing import List, Dict, Any
from chromadb.config import Settings
from together import Together
from Backend.Ingestion.embedder import MedicalEmbedder


class MedicalRetriever:
    """Retrieve and rerank medical documents from ChromaDB."""

    def __init__(self, db_path="./chroma_db", collection="medical_rag", oversample=3):
        self.collection = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        ).get_collection(collection)

        self.embedder = MedicalEmbedder()
        key = os.getenv("TOGETHER_API_KEY", "").strip()
        self.client = Together(api_key=key) if key else None
        self.rerank_model = os.getenv("RERANK_MODEL", "Salesforce/Llama-Rank-v1")
        self.oversample = max(1, int(oversample))

    def _embed_query(self, query: str) -> List[float]:
        """Embed the query using the same embedding model."""
        return self.embedder._embed_batch([query])[0]

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top-K documents by embedding similarity."""
        res = self.collection.query(
            query_embeddings=[self._embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        return [
            {"text": t, "meta": m or {}, "score": 1 - float(d)}
            for t, m, d in zip(docs, metas, dists)
        ]

    def rerank(self, query: str, docs: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Rerank retrieved docs using Together's reranker."""
        if not self.client or not docs:
            return docs[:k]

        result = self.client.rerank.create(
            model=self.rerank_model,
            query=query,
            documents=[d["text"] for d in docs],
            top_n=min(k, len(docs)),
        )

        for item in result.results:
            if 0 <= item.index < len(docs):
                docs[item.index]["score"] = float(item.relevance_score)

        return sorted(docs, key=lambda x: x["score"], reverse=True)[:k]

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Full retrieval + reranking pipeline."""
        raw = self.retrieve(question, top_k * self.oversample)
        return self.rerank(question, raw, top_k)
