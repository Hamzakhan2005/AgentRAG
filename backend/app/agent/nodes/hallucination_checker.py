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
        context = "\n".join([doc.page_content[:300] for doc in docs])
        llm = get_llm()
        messages = [HumanMessage(content=HALLUCINATION_PROMPT.format(
            answer=answer,
            context=context
        ))]
        response = llm.invoke(messages)
        verdict = response.content.strip().lower()
        logger.info(f"Hallucination check verdict: {verdict}")

        if verdict == "no":
            logger.warning("Hallucination detected, flagging answer")
            flagged_answer = f"[Warning: answer may not be fully supported by documents]\n\n{answer}"
            return {**state, "answer": flagged_answer}

        return {**state}

    except Exception as e:
        logger.error(f"Hallucination checker failed: {e}, skipping check")
        return {**state}