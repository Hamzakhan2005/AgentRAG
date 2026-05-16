import logging
from langchain_core.messages import HumanMessage, SystemMessage
from app.services.llm import get_llm
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question using the provided context.
If no context is provided, answer from your general knowledge.
If the context does not contain enough information, say so clearly — do not make things up.
Be concise and mention which part of the context supports your answer.
You also have access to the conversation history — use it to answer follow-up questions correctly."""

def format_context(docs) -> str:
    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        context += f"\n[Chunk {i+1} | Source: {source}]\n{doc.page_content}\n"
    return context

def generator_node(state: AgentState) -> AgentState:
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("documents", [])
    chat_history = state.get("chat_history", [])
    logger.info(f"Generator received chat_history length: {len(chat_history)}")
    for i, msg in enumerate(chat_history):
        logger.info(f"Generator history[{i}]: {msg.__class__.__name__}: {msg.content[:80]}")

    try:
        llm = get_llm()

        if not docs:
            logger.info("No docs, answering directly from LLM + history")
            
            # Build explicit history string instead of relying on message objects
            history_text = ""
            for msg in chat_history:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content}\n\n"
            
            prompt = f"""You are a helpful assistant. Use the conversation history below to answer the current question.

        CONVERSATION HISTORY:
        {history_text}
        CURRENT QUESTION: {query}

        Answer based on the conversation history above."""

            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            return {**state, "answer": response.content}

        context = format_context(docs)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *chat_history,
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        response = llm.invoke(messages)
        answer = response.content
        logger.info(f"Generator produced answer of length {len(answer)}")
        return {**state, "answer": answer}

    except Exception as e:
        logger.error(f"Generator node failed: {e}")
        return {**state, "answer": f"Error generating answer: {str(e)}"}