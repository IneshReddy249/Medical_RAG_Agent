🧠 Medical RAG Agent — Context-Aware Healthcare Question Answering System
🚀 Overview

The Medical RAG Agent is a Retrieval-Augmented Generation (RAG) system built to deliver accurate, explainable, and source-grounded answers to medical questions.
It combines document retrieval, reranking, and large-language-model reasoning using LlamaIndex, LangChain, Together API, and ChromaDB — ensuring both factual precision and clinical reliability.

🧩 Key Features

✅ Parse and embed medical PDFs with LlamaParse + BGE embeddings
✅ Store document vectors in ChromaDB (persistent local DB)
✅ Retrieve and rerank relevant context using Llama-Rank-v1
✅ Generate fact-grounded answers using Meta-Llama-3.1-70B-Instruct-Turbo
✅ Deploy through FastAPI, with endpoints for ingestion, retrieval, and generation
✅ Scalable and ready for Docker + AWS ECS/RDS + CloudWatch monitoring

⚙️ System Architecture

Backend/
├── Ingestion/
│   ├── document_parser.py     # LlamaParse PDF parser
│   ├── embedder.py            # Embeddings generator (BGE)
│   ├── vector_store.py        # Vector storage using ChromaDB
│
├── retrieval/
│   ├── retriever.py           # Context search + reranking
│
├── generator/
│   ├── generator.py           # LLM-based context-grounded answer generator
│
├── main.py                    # FastAPI entry point (API endpoints)
│
├── data/                      # Medical PDFs and extracted data
│   ├── Full.pdf
│   ├── uploads/
│
├── requirements.txt
└── README.md


🧠 RAG Flow

Ingest Documents — Parse PDFs using LlamaParse
Chunk + Embed — Generate vector embeddings (BGE model)
Store — Save embeddings to ChromaDB
Retrieve — Fetch top-k relevant context chunks
Rerank — Optimize order with Llama-Rank-v1
Generate — Produce medically-accurate answer with Together API (Llama-3.1-70B)


| Layer          | Technology                                |
| :------------- | :---------------------------------------- |
| **LLM**        | Meta-Llama-3.1-70B-Instruct-Turbo         |
| **Embeddings** | BAAI/bge-large-en-v1.5                    |
| **Reranker**   | Salesforce/Llama-Rank-v1                  |
| **Vector DB**  | ChromaDB                                  |
| **Backend**    | FastAPI                                   |
| **Deployment** | Docker, AWS ECS, RDS/pgvector, CloudWatch |
| **Parser**     | LlamaParse                                |
| **Language**   | Python 3.11                               |
