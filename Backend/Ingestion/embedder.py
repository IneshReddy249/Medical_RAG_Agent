from __future__ import annotations
import os, time, logging
from typing import List
from together import Together
from transformers import AutoTokenizer
from llama_index.core.schema import TextNode


class MedicalEmbedder:
    """Generate embeddings for text nodes using Together API."""

    def __init__(self, model_name: str | None = None, batch_size: int = 48):
        key = os.getenv("TOGETHER_API_KEY", "").strip()
        if not key:
            raise EnvironmentError("Missing TOGETHER_API_KEY")

        self.client = Together(api_key=key)
        self.model = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        self.batch_size = min(max(batch_size, 1), 128)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def _truncate(self, text: str, max_tokens: int = 448) -> str:
        """Trim long texts to fit model token limits."""
        tokens = self.tokenizer.tokenize(text or "")
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.convert_tokens_to_string(tokens[:max_tokens])

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_nodes(self, nodes: List[TextNode]) -> List[TextNode]:
        """Embed nodes in batches and attach vector embeddings."""
        if not nodes:
            return []

        # Pre-truncate all texts
        for n in nodes:
            n.text = self._truncate(n.text or "")

        # Batch embed and assign vectors
        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i:i + self.batch_size]
            embeddings = self._embed_batch([n.text for n in batch])
            for n, e in zip(batch, embeddings):
                n.embedding = e
            time.sleep(0.1)  # avoid rate limits

        logging.info(f"✅ Embedded {len(nodes)} nodes using {self.model}")
        return nodes
