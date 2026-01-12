from langchain_groq import ChatGroq
from backend.config import tracer

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        streaming=True,
        callbacks=[tracer],  
    )

