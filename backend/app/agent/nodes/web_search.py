import logging
from langchain_core.documents import Document
from app.agent.state import AgentState
from app.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)

def web_search_node(state: AgentState) -> AgentState:
    query = state.get("rewritten_query") or state["query"]
    logger.info(f"Web search node called for query: '{query}'")

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query=query, max_results=4)

        docs = []
        for r in results.get("results", []):
            content = r.get("content", "")
            url = r.get("url", "unknown")
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={"source": url}
                ))

        sources = [doc.metadata["source"] for doc in docs]
        logger.info(f"Web search returned {len(docs)} results")
        return {**state, "documents": docs, "sources": sources, "web_search_used": True}

    except Exception as e:
        logger.error(f"Web search node failed: {e}")
        return {**state, "documents": [], "sources": [], "web_search_used": True}