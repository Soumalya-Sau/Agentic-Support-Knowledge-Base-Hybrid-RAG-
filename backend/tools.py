from langchain_core.tools import tool
from backend.retriever import hybrid_retrieve
from collections import Counter
from langchain_core.tools import tool

@tool
def rag_search_tool(query: str, documents: list):
    """
    Hybrid retrieval tool (semantic + keyword).
    """
    results = hybrid_retrieve(query, documents)

    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "N/A"),
        }
        for doc in results
    ]

@tool
def metadata_tool(results: list):
    """
    Return the most relevant source(s) based on frequency.
    """
    source_counter = Counter()

    for r in results:
        source_counter[(r["source"], r["page"])] += 1

    # pick the MOST frequent source
    most_common = source_counter.most_common(1)

    if not most_common:
        return []

    source, count = most_common[0][0]

    return [f"{source} (page {count})"]
