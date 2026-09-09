# agent/nodes/grader.py — full replacement
import json
import logging
from collections import defaultdict
from langchain_core.messages import HumanMessage
from app.services.llm import get_llm
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

GRADER_PROMPT = """You are a relevance grader. You will be shown a user query and several candidate documents, each from a different source file (assembled from excerpts of that file).

IMPORTANT: The query may ask about MULTIPLE items, dates, or documents at once (e.g. "compare X and Y", "totals for the 10th and 13th"). A document only needs to be relevant to PART of the query to count as relevant - do not reject a document just because it doesn't cover every part of a multi-part question. If a document plausibly answers even one part/date/item mentioned in the query, mark it relevant.

A document counts as relevant even if a specific detail isn't spelled out in the excerpts shown - the missing detail may be elsewhere in that same document.

Query: {query}

Candidate documents:
{documents_block}

Reply with ONLY a JSON array, no other text, in this exact form:
[{{"source": "<source filename or url exactly as given>", "relevant": true}}, ...]
One entry per candidate document listed above, in the same order."""


def _build_documents_block(by_source: dict) -> str:
    parts = []
    for i, (source, docs) in enumerate(by_source.items(), start=1):
        combined = "\n---\n".join(d.page_content[:800] for d in docs)
        parts.append(f"[Document {i}] Source: {source}\n{combined[:2500]}")
    return "\n\n".join(parts)


def grader_node(state: AgentState) -> AgentState:
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    logger.info(f"Grader node called, {len(docs)} docs to grade, retry_count={retry_count}")

    if not docs:
        logger.warning("No documents to grade, will trigger rewrite or fallback")
        return {**state, "documents": []}

    by_source = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        by_source[source].append(doc)

    # Single doc case doesn't need the batching machinery - keep it simple.
    try:
        llm = get_llm()
        documents_block = _build_documents_block(by_source)
        messages = [HumanMessage(content=GRADER_PROMPT.format(
            query=query,
            documents_block=documents_block,
        ))]
        response = llm.invoke(messages)
        raw = response.content.strip()
        # Strip markdown fences the model sometimes adds despite instructions.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        verdicts = json.loads(raw)
        verdict_map = {v["source"]: v.get("relevant", False) for v in verdicts}

        relevant_docs = []
        for source, source_docs in by_source.items():
            is_relevant = verdict_map.get(source, False)
            logger.info(f"Grader verdict for source '{source}' ({len(source_docs)} chunks): {'yes' if is_relevant else 'no'}")
            if is_relevant:
                relevant_docs.extend(source_docs)

        logger.info(f"Grader kept {len(relevant_docs)}/{len(docs)} docs as relevant, from {len(by_source)} source(s)")
        return {**state, "documents": relevant_docs}

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Grader JSON parse failed ({e}), keeping all docs to avoid false rejection")
        return {**state, "documents": docs}
    except Exception as e:
        logger.error(f"Grader node failed: {e}, keeping all docs")
        return {**state, "documents": docs}