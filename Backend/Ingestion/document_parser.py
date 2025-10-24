from __future__ import annotations
import os
from pathlib import Path
from typing import List
from llama_index.core import Document


class LlamaParsePDFLoader:
    """Parse PDFs using LlamaParse — clean and minimal."""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not key:
            raise EnvironmentError("Missing LLAMA_CLOUD_API_KEY")

        from llama_parse import LlamaParse
        self.parser = LlamaParse(
            api_key=key,
            result_type="markdown",
            system_prompt="You are a precise medical document parser.",
            user_prompt="Extract clinically relevant content with headings, lists, and tables.",
            max_timeout=600,
        )

    def load(self) -> List[Document]:
        """Parse and return structured LlamaIndex Documents."""
        docs = self.parser.load_data(self.pdf_path) or []
        for d in docs:
            d.metadata = {
                **(d.metadata or {}),
                "source": str(self.pdf_path),
                "parser": "llamaparse",
                "type": "text",
            }
        return docs
