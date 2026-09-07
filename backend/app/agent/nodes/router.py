import logging
from langchain_core.messages import HumanMessage
from app.services.llm import get_llm
from app.services.vectorstore import session_document_count, detect_named_file
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """You are a query router. Given a user query and optional conversation history, decide how to answer.

Reply with ONLY one of these three words, nothing else:
- rag        → question is about uploaded documents or specific file content
- direct     → question can be answered from conversation history OR is general knowledge
- websearch  → question needs current/real-time information NOT answerable from history or general knowledge

IMPORTANT: If the question is a follow-up that refers to something already discussed in history (like "what company is that", "who said that", "tell me more about that"), ALWAYS reply with "direct" so history is used.

{document_guidance}

Conversation history (last 4 messages):
{history}

Current query: {query}"""

DOCS_UPLOADED_GUIDANCE = """IMPORTANT: {doc_count} document(s) have been uploaded this session. Terms in the query that sound generic or ambiguous out of context - like a person's name, a role/title (e.g. "guide", "supervisor", "manager"), an ID/number, a date, or a company/project name - are very likely referring to FIELDS INSIDE THOSE UPLOADED DOCUMENTS, not real-world entities to look up. When in doubt with documents present, prefer "rag" over "websearch". Only choose "websearch" for queries that clearly need current/real-time info unrelated to any document (e.g. "what's today's weather", "latest news on X", "current stock price of Y")."""

NO_DOCS_GUIDANCE = """No documents have been uploaded this session, so "rag" cannot be answered - choose "direct" or "websearch" instead."""

def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    chat_history = state.get("chat_history", [])
    logger.info(f"Router node called for query: '{query}', history messages: {len(chat_history)}")

    try:
        doc_count = session_document_count()

        # Fast path: if the query explicitly names one of the uploaded files
        # (e.g. "in wpr4 pdf", "the Diary_7th document"), that's an unambiguous
        # signal - force rag directly and skip the LLM call entirely.
        # Faster (no extra API round trip) and more reliable than hoping
        # the LLM notices the filename in free text.
        named_file = detect_named_file(query) if doc_count > 0 else None
        if named_file:
            logger.info(f"Router fast-path: query names uploaded file '{named_file}' - forcing 'rag'")
            return {**state, "route": "rag"}

        # Build history snippet for router context
        history_text = ""
        for msg in chat_history[-4:]:
            role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
            history_text += f"{role}: {msg.content[:200]}\n"

        if not history_text:
            history_text = "No previous conversation."

        if doc_count > 0:
            document_guidance = DOCS_UPLOADED_GUIDANCE.format(doc_count=doc_count)
        else:
            document_guidance = NO_DOCS_GUIDANCE

        llm = get_llm()
        messages = [HumanMessage(content=ROUTER_PROMPT.format(
            query=query,
            history=history_text,
            document_guidance=document_guidance,
        ))]
        response = llm.invoke(messages)
        route = response.content.strip().lower()

        if route not in ["rag", "websearch", "direct"]:
            logger.warning(f"Router returned unexpected value '{route}', defaulting to 'rag'")
            route = "rag"

        # Safety net: never route to rag if nothing is actually indexed.
        if route == "rag" and doc_count == 0:
            logger.warning("Router chose 'rag' but no documents are uploaded - falling back to 'direct'")
            route = "direct"

        logger.info(f"Router decision: {route} (session_documents={doc_count})")
        return {**state, "route": route}

    except Exception as e:
        logger.error(f"Router node failed: {e}, defaulting to 'rag'")
        return {**state, "route": "rag"}