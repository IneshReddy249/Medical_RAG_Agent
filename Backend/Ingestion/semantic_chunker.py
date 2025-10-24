from __future__ import annotations
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode


class SemanticChunker:
    """Split documents into sentence-based chunks with sequence and page labels."""

    def __init__(self, chunk_size: int = 440, overlap: int = 40):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            paragraph_separator="\n\n",
        )

    def chunk(self, docs: List[Document]) -> List[TextNode]:
        nodes = self.splitter.get_nodes_from_documents(docs)
        for i, node in enumerate(nodes, 1):
            md = node.metadata or {}
            md["seq"] = i
            md["page_label"] = (
                md.get("page_label")
                or md.get("page")
                or md.get("page_number")
            )
            node.metadata = md
        return nodes
