import logging
from langchain_core.messages import HumanMessage
from app.services.llm import get_llm
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

HALLUCINATION_PROMPT = """You are a hallucination checker. Given a generated answer and source documents, decide if the answer is broadly supported by or consistent with the documents.

Be lenient — if the answer is mostly correct and references content from the documents, reply 'yes'.
Only reply 'no' if the answer contains clear fabrications not present anywhere in the documents.

Reply with ONLY 'yes' or 'no'.

Answer: {answer}
Documents: {context}"""

def hallucination_checker_node(state: AgentState) -> AgentState:
    answer = state.get("answer", "")
    docs = state.get("documents", [])
    logger.info("Hallucination checker node called")

    if not docs:
        logger.warning("No docs to check against, skipping hallucination check")
        return {**state}

    try:
        context = "\n---\n".join(doc.page_content for doc in docs)  # full content, not [:300]
        llm = get_llm()
        messages = [HumanMessage(content=HALLUCINATION_PROMPT.format(
            answer=answer,
            context=context
        ))]
        response = llm.invoke(messages)
        verdict = response.content.strip().lower()
        logger.info(f"Hallucination check verdict: {verdict}")

        if verdict == "no":
            logger.warning("Hallucination detected, regenerating once with a stricter prompt")
            strict_messages = [
                HumanMessage(content=(
                    f"Your previous answer may not be fully supported by the source documents below. "
                    f"Re-answer the same question using ONLY facts explicitly present in these documents. "
                    f"If a fact isn't there, say it's not available - do not guess.\n\n"
                    f"Documents:\n{context}\n\nPrevious answer: {answer}"
                ))
            ]
            retry = llm.invoke(strict_messages)
            return {**state, "answer": retry.content}

        return {**state}

    except Exception as e:
        logger.error(f"Hallucination checker failed: {e}, skipping check")
        return {**state}