import logging
import time
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import agent_graph

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session store: session_id -> chat_history
session_store: dict = {}

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    stream: bool = False

@router.post("/chat")
async def chat(request: ChatRequest):
    logger.info(f"Chat request: query='{request.query}', session_id={request.session_id}")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Create new session if none provided
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Using session_id: {session_id}")

    # Load existing history or start fresh
    chat_history = session_store.get(session_id, [])
    logger.info(f"Loaded {len(chat_history)} messages from session history")

    for i, msg in enumerate(chat_history):
        role = msg.__class__.__name__
        logger.info(f"History[{i}] {role}: {msg.content[:80]}")

    try:
        initial_state = {
            "query": request.query,
            "rewritten_query": None,
            "documents": [],
            "answer": None,
            "sources": [],
            "web_search_used": False,
            "retry_count": 0,
            "route": None,
            "chat_history": chat_history,
        }

        

        logger.info("Invoking agent graph")
        t_start = time.perf_counter()
        result = agent_graph.invoke(initial_state)
        t_end = time.perf_counter()
        logger.info(f"[TIMING] full agent_graph.invoke(): {t_end - t_start:.3f}s (route={result.get('route')})")
        answer = result.get("answer", "No answer generated")

        # Update session history
        chat_history.append(HumanMessage(content=request.query))
        clean_answer = answer.replace("[Warning: answer may not be fully supported by documents]\n\n", "")
        chat_history.append(AIMessage(content=clean_answer))
        session_store[session_id] = chat_history
        logger.info(f"Session {session_id} now has {len(chat_history)} messages")

        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "chunks_used": len(result.get("documents", [])),
            "route_taken": result.get("route"),
            "web_search_used": result.get("web_search_used", False),
            "rewritten_query": result.get("rewritten_query"),
            "session_id": session_id,
        }

    except Exception as e:
        logger.error(f"Agent graph failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    logger.info(f"History requested for session: {session_id}")
    history = session_store.get(session_id, [])
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")

    formatted = []
    for msg in history:
        formatted.append({
            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
            "content": msg.content,
        })

    return {"session_id": session_id, "messages": formatted}

@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    logger.info(f"Clearing session: {session_id}")
    if session_id in session_store:
        del session_store[session_id]
        return {"message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")