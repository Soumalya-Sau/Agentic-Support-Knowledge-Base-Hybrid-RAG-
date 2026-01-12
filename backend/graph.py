from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from backend.llm import get_llm
from backend.tools import rag_search_tool, metadata_tool

class AgentState(TypedDict):
    question: str
    documents: list
    retrieved: List[dict]
    answer: str
    citations: list

llm = get_llm()

def retrieve_node(state: AgentState):
    retrieved = rag_search_tool.invoke({
        "query": state["question"],
        "documents": state["documents"]
    })
    return {"retrieved": retrieved}

def draft_answer_node(state: AgentState):
    context = "\n\n".join(r["content"] for r in state["retrieved"])

    prompt = f"""
You are a support assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{state['question']}
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}

def cite_node(state: AgentState):
    citations = metadata_tool.invoke({
        "results": state["retrieved"]
    })
    return {"citations": citations}

graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("draft", draft_answer_node)
graph.add_node("cite", cite_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "draft")
graph.add_edge("draft", "cite")
graph.add_edge("cite", END)

app = graph.compile()
