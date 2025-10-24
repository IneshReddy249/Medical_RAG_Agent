from __future__ import annotations
import os, logging
from together import Together
from Backend.retrieval.retriever import MedicalRetriever


class MedicalGenerator:
    """Generates context-grounded medical answers using Together API."""

    def __init__(self, db_path: str = "./chroma_db", collection: str = "medical_rag"):
        # Load Together API key
        api_key = os.getenv("TOGETHER_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("❌ TOGETHER_API_KEY not set. Please add it to your .env file.")

        # Initialize Together client and model
        self.client = Together(api_key=api_key)
        self.model = os.getenv("CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct-Turbo")

        # Initialize retriever
        self.retriever = MedicalRetriever(db_path=db_path, collection=collection)

    async def generate_async(self, question: str, top_k: int = 5) -> str:
        """Retrieve context from DB and generate a precise, factual medical answer."""
        try:
            docs = self.retriever.query(question, top_k=top_k)
            if not docs:
                return "⚠️ Not enough relevant information found."

            # Merge top documents into a single context
            context = "\n\n".join(d["text"] for d in docs)

            # Construct focused RAG prompt
            prompt = (
                "You are a medical assistant. Use ONLY the provided context to answer the question.\n"
                "Be factual, concise, and professional. Cite sources like [1], [2].\n"
                "If the context lacks info, state that explicitly.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )

            # Generate response from Together LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2,
            )

            # Extract output safely
            return (
                response.choices[0].message.content.strip()
                if response and response.choices
                else "⚠️ No response from model."
            )

        except Exception as e:
            logging.error(f"[MedicalGenerator Error] {e}")
            return f"❌ Generation failed: {str(e)}"
