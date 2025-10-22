# Backend/Ingestion/embedder.py
import os, time, logging
from together import Together
from transformers import AutoTokenizer
from llama_index.core.schema import TextNode
from typing import List

class MedicalEmbedder:
    """Simple and safe medical embedder using Together API"""

    def __init__(self, model_name="BAAI/bge-large-en-v1.5", batch_size=50):
        key = os.getenv("TOGETHER_API_KEY")
        if not key:
            raise EnvironmentError("Set TOGETHER_API_KEY in environment")
        self.client = Together(api_key=key)
        self.model = model_name
        self.bs = min(max(batch_size, 1), 128)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _truncate(self, text: str, max_tokens=440) -> str:
        tokens = self.tokenizer.tokenize(text)
        return text if len(tokens) <= max_tokens else self.tokenizer.convert_tokens_to_string(tokens[:max_tokens])

    def _normalize(self, v: List[float]) -> List[float]:
        n = (sum(x*x for x in v) ** 0.5) or 1.0
        return [x / n for x in v]

    def _embed_batch(self, texts: List[str], retries=3) -> List[List[float]]:
        for attempt in range(retries + 1):
            try:
                resp = self.client.embeddings.create(model=self.model, input=texts)
                return [self._normalize(d.embedding) for d in resp.data]
            except Exception as e:
                if attempt == retries:
                    raise
                wait = 2 ** attempt
                logging.warning(f"Retry {attempt+1} in {wait}s: {e}")
                time.sleep(wait)

    def embed_nodes(self, nodes: List[TextNode]) -> List[TextNode]:
        if not nodes: return []
        for n in nodes: n.text = self._truncate(n.text or "")
        for i in range(0, len(nodes), self.bs):
            batch = nodes[i:i+self.bs]
            vecs = self._embed_batch([n.text for n in batch])
            for n, v in zip(batch, vecs): n.embedding = v
            time.sleep(0.3)
        logging.info(f"✅ Embedded {len(nodes)} nodes with {self.model}")
        return nodes
