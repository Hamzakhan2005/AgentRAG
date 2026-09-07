import functools
import logging
import time
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState

from app.agent.nodes import (
    router_node,
    retriever_node,
    web_search_node,
    grader_node,
    rewriter_node,
    generator_node,
    hallucination_checker_node,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(state):
        t0 = time.perf_counter()
        result = fn(state)
        elapsed = time.perf_counter() - t0
        logger.info(f"[TIMING] node '{name}': {elapsed:.3f}s")
        return result
    return wrapper

def route_after_router(state: AgentState) -> str:
    route = state.get("route", "rag")
    logger.info(f"Routing after router: {route}")
    if route == "websearch":
        return "web_search"
    elif route == "direct":
        return "generator"
    return "retriever"

def route_after_grader(state: AgentState) -> str:
    docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)
    web_search_used = state.get("web_search_used", False)

    if docs:
        logger.info("Grader passed, moving to generator")
        return "generator"

    if web_search_used or retry_count >= MAX_RETRIES:
        logger.warning("Max retries or web search already used, going to generator anyway")
        return "generator"

    logger.info("Grader failed, moving to rewriter")
    return "rewriter"

def route_after_rewriter(state: AgentState) -> str:
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        logger.warning("Max retries reached, falling back to web search")
        return "web_search"
    return "retriever"

def build_graph():
    logger.info("Building LangGraph agent graph")
    graph = StateGraph(AgentState)

    graph.add_node("router", timed("router", router_node))
    graph.add_node("retriever", timed("retriever", retriever_node))
    graph.add_node("web_search", timed("web_search", web_search_node))
    graph.add_node("grader", timed("grader", grader_node))
    graph.add_node("rewriter", timed("rewriter", rewriter_node))
    graph.add_node("generator", timed("generator", generator_node))
    graph.add_node("hallucination_checker", timed("hallucination_checker", hallucination_checker_node))

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", route_after_router, {
        "retriever": "retriever",
        "web_search": "web_search",
        "generator": "generator",
    })

    graph.add_edge("retriever", "grader")
    graph.add_edge("web_search", "grader")

    graph.add_conditional_edges("grader", route_after_grader, {
        "generator": "generator",
        "rewriter": "rewriter",
    })

    graph.add_conditional_edges("rewriter", route_after_rewriter, {
        "retriever": "retriever",
        "web_search": "web_search",
    })

    graph.add_edge("generator", "hallucination_checker")
    graph.add_edge("hallucination_checker", END)

    compiled = graph.compile()
    logger.info("LangGraph agent graph built successfully")
    return compiled

agent_graph = build_graph()