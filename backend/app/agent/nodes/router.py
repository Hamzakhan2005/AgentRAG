import logging
from langchain_core.messages import HumanMessage
from app.services.llm import get_llm
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """You are a query router. Given a user query and optional conversation history, decide how to answer.

Reply with ONLY one of these three words, nothing else:
- rag        → question is about uploaded documents or specific file content
- direct     → question can be answered from conversation history OR is general knowledge
- websearch  → question needs current/real-time information NOT answerable from history or general knowledge

IMPORTANT: If the question is a follow-up that refers to something already discussed in history (like "what company is that", "who said that", "tell me more about that"), ALWAYS reply with "direct" so history is used.

Conversation history (last 4 messages):
{history}

Current query: {query}"""

def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    chat_history = state.get("chat_history", [])
    logger.info(f"Router node called for query: '{query}', history messages: {len(chat_history)}")

    try:
        # Build history snippet for router context
        history_text = ""
        for msg in chat_history[-4:]:
            role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
            history_text += f"{role}: {msg.content[:200]}\n"

        if not history_text:
            history_text = "No previous conversation."

        llm = get_llm()
        messages = [HumanMessage(content=ROUTER_PROMPT.format(
            query=query,
            history=history_text
        ))]
        response = llm.invoke(messages)
        route = response.content.strip().lower()

        if route not in ["rag", "websearch", "direct"]:
            logger.warning(f"Router returned unexpected value '{route}', defaulting to 'rag'")
            route = "rag"

        logger.info(f"Router decision: {route}")
        return {**state, "route": route}

    except Exception as e:
        logger.error(f"Router node failed: {e}, defaulting to 'rag'")
        return {**state, "route": "rag"}