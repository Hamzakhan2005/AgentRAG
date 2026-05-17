import logging
from app.services.retriever import retrieve_docs
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

def retriever_node(state: AgentState) -> AgentState:
    query = state.get("rewritten_query") or state["query"]
    logger.info(f"Retriever node called with query: '{query}'")

    try:
        docs = retrieve_docs(query)
        sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
        logger.info(f"Retriever node got {len(docs)} docs from sources: {sources}")
        return {**state, "documents": docs, "sources": sources}

    except Exception as e:
        logger.error(f"Retriever node failed: {e}")
        return {**state, "documents": [], "sources": []}