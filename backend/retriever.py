from rank_bm25 import BM25Okapi
from backend.config import TOP_K
from backend.embeddings import load_vectorstore

# ✅ Load FAISS ONCE (safe)
VECTORSTORE = None

def get_vectorstore():
    global VECTORSTORE
    if VECTORSTORE is None:
        VECTORSTORE = load_vectorstore()
    return VECTORSTORE


def hybrid_retrieve(query, documents):
    # ----- Keyword retrieval (BM25) -----
    tokenized_docs = [doc.page_content.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_docs = bm25.get_top_n(query.split(), documents, n=TOP_K)

    # ----- Semantic retrieval (FAISS) -----
    vectorstore = get_vectorstore()
    semantic_docs = vectorstore.similarity_search(query, k=TOP_K)

    # ----- Merge & deduplicate -----
    combined = {doc.page_content: doc for doc in bm25_docs + semantic_docs}
    return list(combined.values())
