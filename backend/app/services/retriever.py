# app/services/retriever.py — full replacement, this is the version that must be live
import logging
import time
from app.services.vectorstore import (
    get_vectorstore,
    detect_named_file,
    session_document_count,
    is_broad_query,
    get_all_chunks_grouped_by_source,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def retrieve_docs(query: str) -> list:
    logger.info(f"Retrieving docs for query: '{query}'")
    t0 = time.time()

    doc_count = session_document_count()

    if doc_count > 1 and is_broad_query(query):
        t_vs0 = time.time()
        grouped = get_all_chunks_grouped_by_source()
        docs = []
        for source, texts in grouped.items():
            for text in texts[:5]:
                docs.append(Document(page_content=text, metadata={"source": source}))
        logger.info(f"[TIMING] broad-query full fetch: {time.time() - t_vs0:.3f}s")
        logger.info(f"Retrieved {len(docs)} chunks (broad query, all {len(grouped)} sources)")
        logger.info(f"[TIMING] total retrieve_docs(): {time.time() - t0:.3f}s")
        return docs

    t_vs0 = time.time()
    vectorstore = get_vectorstore()
    logger.info(f"[TIMING] get_vectorstore(): {time.time() - t_vs0:.3f}s")

    top_k = 4 if doc_count <= 1 else min(4 + doc_count, 20)
    search_kwargs = {"k": top_k}

    named_file = detect_named_file(query)
    if named_file:
        search_kwargs["filter"] = {"source": named_file}
        logger.info(f"Scoping retrieval to named file: {named_file}")

    t_search0 = time.time()
    docs = vectorstore.similarity_search(query, **search_kwargs)
    logger.info(f"[TIMING] vector search only: {time.time() - t_search0:.3f}s")

    if named_file and not docs:
        logger.warning(f"Scoped search for '{named_file}' returned nothing, retrying unscoped")
        docs = vectorstore.similarity_search(query, k=top_k)

    logger.info(f"Retrieved {len(docs)} chunks")
    logger.info(f"[TIMING] total retrieve_docs(): {time.time() - t0:.3f}s")
    return docs