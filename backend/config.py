import os
from dotenv import load_dotenv
from langchain_core.tracers import LangChainTracer

load_dotenv()

# ===== API Keys =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "support-kb-agent")

# ===== LangSmith Tracer =====
tracer = LangChainTracer(project_name=LANGCHAIN_PROJECT)

# ===== Paths =====
DATA_DIR = "data/raw"
VECTORSTORE_DIR = "data/vectorstore"

# ===== RAG Params =====
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
