from typing import List
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter

class SemanticChunker:
    def __init__(self, chunk_size: int = 440, chunk_overlap: int = 40):
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, paragraph_separator="\n\n")

    def chunk(self, docs: List[Document]) -> List[TextNode]:
        nodes = self.splitter.get_nodes_from_documents(docs)
        out: List[TextNode] = []
        for i, n in enumerate(nodes, 1):
            md = dict(n.metadata or {})
            md.setdefault("seq", i)
            if "page_label" not in md:
                for k in ("page", "page_number"):
                    if k in md: md["page_label"] = md[k]; break
            n.metadata = md
            out.append(n)
        return out
