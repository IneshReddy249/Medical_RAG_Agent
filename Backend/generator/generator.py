import os, httpx, logging
from Backend.retrieval.retriever import MedicalRetriever

class MedicalGenerator:
    def __init__(self, db_path=None, collection=None):
        self.retriever = MedicalRetriever(db_path=db_path or "./chroma_db", collection=collection or "medical_rag")
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        self.url = "https://api.together.xyz/v1/chat/completions"

    async def generate_async(self, question: str, top_k: int = 5) -> str:
        if not self.api_key: return "⚠️ TOGETHER_API_KEY not set."
        ctx = self.retriever.query(question, top_k=top_k)
        if not ctx or max(c.get("score", 0) for c in ctx) < 0.55:
            return "Not enough information to answer reliably."
        context = "\n\n".join(d["text"] for d in ctx)
        prompt = ("You are a medical assistant. Use ONLY the context.\n"
                  "Cite like [1], [2] mapped to the order of the context chunks.\n\n"
                  f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                r = await client.post(self.url, headers={"Authorization": f"Bearer {self.api_key}"},
                                      json={"model": self.model, "messages":[{"role":"user","content":prompt}],
                                            "max_tokens":400, "temperature":0.2})
                r.raise_for_status()
                ans = r.json()["choices"][0]["message"]["content"].strip()
                if "[" not in ans: ans += "\n\nSources: [1]"
                return ans
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return "Error generating answer."
