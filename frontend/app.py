import os
import sys
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.ingest import load_and_chunk_pdfs
from backend.embeddings import create_vectorstore
from backend.llm import get_llm
from mcp_client.client import MCPToolClient

llm = get_llm()
tool_client = MCPToolClient()

st.set_page_config(page_title="Support KB Agent", layout="wide")
st.title("🤖 Support Knowledge Base Agent")

# -------- Sidebar --------
st.sidebar.title("📂 Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
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

# -------- Chat History --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask a question about your documents")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    results = tool_client.rag_search(query)
    sources = tool_client.best_source(results)

    context = "\n\n".join(r["content"] for r in results)
    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""
    response = llm.invoke(prompt)

    assistant_text = response.content
    if sources:
        assistant_text += "\n\n**Source:**\n" + "\n".join(f"- {s}" for s in sources)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_text}
    )

    with st.chat_message("assistant"):
        st.write(response.content)
        if sources:
            st.markdown("**Source:**")
            for s in sources:
                st.write(f"- {s}")
