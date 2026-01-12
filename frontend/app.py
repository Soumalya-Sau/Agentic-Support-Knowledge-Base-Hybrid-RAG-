import os
import sys
import streamlit as st

# ---------------- PATH FIX (already discussed earlier) ----------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ---------------- BACKEND IMPORTS (UNCHANGED) ----------------
from backend.ingest import load_and_chunk_pdfs
from backend.embeddings import create_vectorstore
from backend.graph import app
from backend.config import tracer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Support KB Agent", layout="wide")

st.title("🤖 Support Knowledge Base Agent")

# ================= SIDEBAR =================
st.sidebar.title("📂 Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Ingest Documents"):
    os.makedirs("data/raw", exist_ok=True)

    for file in uploaded_files:
        with open(f"data/raw/{file.name}", "wb") as f:
            f.write(file.getbuffer())

    with st.spinner("Indexing documents..."):
        docs = load_and_chunk_pdfs("data/raw")
        create_vectorstore(docs)

    st.sidebar.success("Documents ingested successfully")

# ================= CHAT HISTORY (NEW) =================
# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render previous chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ================= CHAT INPUT =================


query = st.chat_input("Ask a question about your documents")

if query:
    # -------- Store & show user message --------
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    st.chat_message("user").write(query)

    # -------- Run LangGraph agent --------
    documents = load_and_chunk_pdfs("data/raw")

    result = app.invoke(
        {
            "question": query,
            "documents": documents
        },
        config={"callbacks": [tracer]}
    )

    # -------- Store & show assistant message --------
    assistant_content = result["answer"]
    if result.get("citations"):
        assistant_content += "\n\n**Source:**\n"
        assistant_content += "\n".join(f"- {c}" for c in result["citations"])

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": assistant_content
    })

    with st.chat_message("assistant"):
        st.write(result["answer"])
        st.markdown("**Source:**")
        for c in result["citations"]:
            st.write(f"- {c}")