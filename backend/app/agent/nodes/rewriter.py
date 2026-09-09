
import logging
from langchain_core.messages import HumanMessage
from app.services.llm import get_llm
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

REWRITER_PROMPT = """You are a query rewriter. The original query did not retrieve relevant documents.
Rewrite the query to be clearer and more specific in WORDING ONLY - do not invent or assume any facts not present in the original query (no years, numbers, names, or locations that weren't stated). If the original query doesn't mention a year, do NOT add one. If it mentions multiple items (e.g. two dates), keep the rewritten query asking about all of them, not just one.
Reply with ONLY the rewritten query, nothing else.

Original query: {query}"""

def rewriter_node(state: AgentState) -> AgentState:
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    logger.info(f"Rewriter node called, attempt {retry_count + 1}")

    try:
        llm = get_llm()
        messages = [HumanMessage(content=REWRITER_PROMPT.format(query=query))]
        response = llm.invoke(messages)
        rewritten = response.content.strip()
        logger.info(f"Rewritten query: '{rewritten}'")
        return {**state, "rewritten_query": rewritten, "retry_count": retry_count + 1}

    except Exception as e:
        logger.error(f"Rewriter node failed: {e}")
        return {**state, "retry_count": retry_count + 1}