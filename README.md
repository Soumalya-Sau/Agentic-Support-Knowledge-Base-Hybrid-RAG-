# 🤖 Support Knowledge Base Agent (Hybrid RAG System)

An intelligent **Support Knowledge Base Agent** built using a **hybrid Retrieval-Augmented Generation (RAG)** pipeline.  
The system allows users to upload documents (PDFs), retrieve relevant information using **keyword + semantic search**, and generate accurate answers using a **Groq LLM**, with clear source citations.

This project is designed to be **stable, modular, and interview-ready**, avoiding experimental tooling while following real-world engineering practices.

---

## 🚀 Setup & Installation

### Prerequisites
- Python **3.10+**
- Virtual environment (`venv`)
- Groq API Key

### Installation

```
# Clone the repository
git clone <your-repo-url>
cd Agentic-Support-Knowledge-Base-System

# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 🔐 Environment Variables
Create a .env file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Only the Groq API key is required.
Embeddings are computed locally using HuggingFace models.

## 🏃 How to Run

### 1️⃣ Start the Tool Server (FastAPI)
Run from the project root:

```
uvicorn mcp_server.server:app --host 0.0.0.0 --port 8000
```

The tool server exposes retrieval endpoints used by the frontend.

### 2️⃣ Start the Streamlit App
Open a new terminal:

```
python -m streamlit run frontend/app.py
```

The app will open at:

```
http://localhost:8501
```

## 📂 Document Ingestion Workflow

1. Upload PDFs using the Streamlit sidebar

2. Click Ingest Documents

3. Documents are:

• Loaded using PyPDFLoader

• Chunked with RecursiveCharacterTextSplitter

• Embedded using a free HuggingFace embedding model

• Stored locally in a FAISS vector index

• Embeddings are computed once and reused for all queries.

## 🔍 RAG Pipeline (Hybrid Retrieval)

### Retrieval Strategy

The system uses hybrid retrieval:

• BM25 (keyword search) — exact term matching

• FAISS (semantic search) — vector similarity

Both signals are combined to improve retrieval quality and reduce hallucinations.

### Retrieval Flow

1. User query → FastAPI tool server

2. Hybrid retrieval over indexed documents

3. Top relevant chunks returned to frontend

4. Context passed to the LLM  

## 🧠 Answer Generation

• LLM Provider: Groq

• Prompt is strictly grounded in retrieved context

• The model is instructed to answer only from provided documents

• The final answer includes the most relevant source citation

Example citation format:

```
Source:
- user_manual.pdf (page 3)
```

## 🧱 System Architecture

```
Streamlit Frontend
   |
   |  HTTP (requests)
   v
FastAPI Tool Server (MCP-style)
   |
   v
Hybrid RAG Backend
(BM25 + FAISS + HuggingFace Embeddings)
   |
   v
Groq LLM
```

## 📁 Project Structure

```
Agentic Support Knowledge Base System/
│
├── backend/
│   ├── ingest.py        # Document loading & chunking
│   ├── embeddings.py   # FAISS vector store handling
│   ├── retriever.py    # Hybrid BM25 + FAISS retrieval
│   ├── llm.py          # Groq LLM initialization
│   └── config.py
│
├── mcp_server/
│   └── server.py       # FastAPI-based tool server
│
├── mcp_client/
│   └── client.py       # HTTP client for tool calls
│
├── frontend/
│   └── app.py          # Streamlit UI
│
├── data/
│   ├── raw/            # Uploaded PDFs
│   └── vectorstore/    # FAISS index
│
├── .env.example
├── requirements.txt
└── .gitignore
```

All directories are valid Python packages (__init__.py included).

## ✨ Features

✅ PDF-based knowledge ingestion
✅ Hybrid retrieval (BM25 + semantic search)
✅ Free local embeddings (no embedding API cost)
✅ FastAPI-based tool server
✅ Streamlit chat UI with history
✅ Source citation for answers
✅ Cached document loading for performance
✅ Clean, modular, interview-ready codebase

## ⚙️ Design Choices & Rationale

• FastAPI instead of experimental MCP transports for stability

• Hybrid retrieval to improve recall and precision

• Local embeddings to avoid API costs

• No agent overengineering — focused on correctness and clarity

• Separation of concerns between UI, tools, and backend logic

## 📦 Dependencies

Key libraries used:

• Streamlit

• FastAPI + Uvicorn

• LangChain

• FAISS

• Rank-BM25

• Sentence-Transformers

• Groq LLM

See requirements.txt for full list.


## 📄 License

This project is licensed under the MIT License.

🙏 Acknowledgements

• LangChain for RAG abstractions

• HuggingFace for embedding models

• Groq for fast LLM inference

• Streamlit for rapid UI development

• FastAPI for reliable backend services
