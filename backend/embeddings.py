from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from backend.config import VECTORSTORE_DIR

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

def create_vectorstore(docs):
    embeddings = get_embeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTORSTORE_DIR)
    return db

def load_vectorstore():
    embeddings = get_embeddings()
    return FAISS.load_local(VECTORSTORE_DIR, embeddings,allow_dangerous_deserialization=True)
