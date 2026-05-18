import logging
from langchain_core.messages import HumanMessage, SystemMessage
from app.services.retriever import retrieve_docs
from app.services.llm import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly — do not make things up.
Always be concise and cite which part of the context your answer is based on."""

def format_context(docs: list) -> str:
    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        context += f"\n[Chunk {i+1} | Source: {source}]\n{doc.page_content}\n"
    return context

def run_rag(query: str) -> dict:
    logger.info(f"Running RAG pipeline for query: '{query}'")
    try:
        docs = retrieve_docs(query)

        if not docs:
            logger.warning("No documents retrieved, returning fallback response")
            return {
                "answer": "I could not find relevant information in the uploaded documents.",
                "sources": [],
                "chunks_used": 0,
            }

        context = format_context(docs)
        logger.info(f"Context built, total chars: {len(context)}")

        llm = get_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]

        logger.info("Sending request to Groq LLM")
        response = llm.invoke(messages)
        answer = response.content
        logger.info(f"LLM response received, answer length: {len(answer)}")

        sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(docs),
        }

    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        raise