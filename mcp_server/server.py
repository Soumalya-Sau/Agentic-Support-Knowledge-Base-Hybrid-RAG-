from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from collections import Counter

from backend.ingest import load_and_chunk_pdfs
from backend.retriever import hybrid_retrieve

app = FastAPI(title="Support KB Tool Server")

# -------- Cache documents --------
DOCUMENTS_CACHE = None

# -------- Models --------

class RAGQuery(BaseModel):
    query: str

class RAGResult(BaseModel):
    content: str
    source: str
    page: int | str

class SourceResponse(BaseModel):
    sources: List[str]

# -------- Endpoints --------

@app.post("/rag_search", response_model=List[RAGResult])
def rag_search(payload: RAGQuery):
    global DOCUMENTS_CACHE

    if DOCUMENTS_CACHE is None:
        DOCUMENTS_CACHE = load_and_chunk_pdfs("data/raw")

    results = hybrid_retrieve(payload.query, DOCUMENTS_CACHE)

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A"),
        }
        for doc in results
    ]


@app.post("/best_source", response_model=SourceResponse)
def best_source(results: List[RAGResult]):
    counter = Counter()
    for r in results:
        counter[(r.source, r.page)] += 1

    if not counter:
        return {"sources": []}

    (source, page), _ = counter.most_common(1)[0]
    return {"sources": [f"{source} (page {page})"]}

