import logging
from langchain_core.messages import HumanMessage
from app.services.llm import get_llm
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

GRADER_PROMPT = """You are a relevance grader. Given a user query and a retrieved document chunk, decide if the chunk is relevant to the query.

Reply with ONLY 'yes' or 'no', nothing else.

Query: {query}
Document: {document}"""

MAX_RETRIES = 2

def grader_node(state: AgentState) -> AgentState:
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    logger.info(f"Grader node called, {len(docs)} docs to grade, retry_count={retry_count}")

    if not docs:
        logger.warning("No documents to grade, will trigger rewrite or fallback")
        return {**state, "documents": []}

    try:
        llm = get_llm()
        relevant_docs = []

        for i, doc in enumerate(docs):
            messages = [HumanMessage(content=GRADER_PROMPT.format(
                query=query,
                document=doc.page_content[:500]
            ))]
            response = llm.invoke(messages)
            verdict = response.content.strip().lower()
            logger.info(f"Grader verdict for chunk {i+1}: {verdict}")

            if verdict == "yes":
                relevant_docs.append(doc)

        logger.info(f"Grader kept {len(relevant_docs)}/{len(docs)} docs as relevant")
        return {**state, "documents": relevant_docs}

    except Exception as e:
        logger.error(f"Grader node failed: {e}, keeping all docs")
        return {**state}