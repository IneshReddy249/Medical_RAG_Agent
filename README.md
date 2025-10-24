🧠 Medical RAG Agent — Context-Aware Healthcare Question Answering System
🚀 Overview

The Medical RAG Agent is a Retrieval-Augmented Generation (RAG) system built to deliver accurate, explainable, and source-grounded answers to medical questions.
It combines document retrieval, reranking, and large-language-model reasoning using LlamaIndex, LangChain, Together API, and ChromaDB — ensuring both factual precision and clinical reliability.

🧩 Key Features

🔍 Contextual Understanding: Parses and embeds medical PDFs using LlamaParse + BGE-large embeddings

🧠 Semantic Retrieval: Retrieves and reranks context via ChromaDB + Llama-Rank-v1

💬 LLM-Grounded Answers: Generates fact-based, source-cited responses using Meta-Llama-3.1-70B-Instruct-Turbo

⚙️ Modular Design: Separate layers for Ingestion, Retrieval, and Generation

🚀 FastAPI Backend: Exposes REST endpoints for ingestion and querying

☁️ Scalable Deployment: Dockerized and compatible with AWS ECS, RDS, S3, and CloudWatch for production use
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


| Step            | Description                                        | Tools Used                                         |
| --------------- | -------------------------------------------------- | -------------------------------------------------- |
| **1. Ingest**   | Parse medical PDFs using LlamaParse and preprocess | `LlamaParseReader`, `PyMuPDF`                      |
| **2. Embed**    | Convert chunks to dense vector embeddings          | `BAAI/bge-large-en-v1.5`                           |
| **3. Store**    | Persist embeddings in a local vector DB            | `ChromaDB`                                         |
| **4. Retrieve** | Retrieve top-k relevant chunks                     | `cosine similarity`                                |
| **5. Rerank**   | Refine context order for accuracy                  | `Llama-Rank-v1`                                    |
| **6. Generate** | Create source-grounded answers via LLM             | `Meta-Llama-3.1-70B-Instruct-Turbo` (Together API) |


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
