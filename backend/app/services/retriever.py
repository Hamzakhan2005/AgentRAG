import logging
import time
from app.services.vectorstore import get_vectorstore, detect_named_file

logger = logging.getLogger(__name__)

TOP_K = 4


def retrieve_docs(query: str) -> list:
    logger.info(f"Retrieving docs for query: '{query}'")
    try:
        t0 = time.perf_counter()
        vectorstore = get_vectorstore()
        t1 = time.perf_counter()
        logger.info(f"[TIMING] get_vectorstore() (incl. embedding model load): {t1 - t0:.3f}s")

        source_filter = detect_named_file(query)
        if source_filter:
            logger.info(f"Query names a specific uploaded file: '{source_filter}' - scoping retrieval to it")

        search_kwargs = {"k": TOP_K}
        if source_filter:
            search_kwargs["filter"] = {"source": source_filter}

        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(query)

        # Safety net: if a scoped search came back empty (e.g. filename
        # matched but that doc has no relevant chunks for this query),
        # retry once without the filter rather than returning nothing.
        if not docs and source_filter:
            logger.info(f"Scoped search on '{source_filter}' returned no results - retrying unscoped")
            retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            docs = retriever.invoke(query)

        t2 = time.perf_counter()
        logger.info(f"[TIMING] vector search only: {t2 - t1:.3f}s")
        logger.info(f"[TIMING] total retrieve_docs(): {t2 - t0:.3f}s")

        logger.info(f"Retrieved {len(docs)} chunks")
        for i, doc in enumerate(docs):
            logger.debug(f"Chunk {i+1} source: {doc.metadata.get('source', 'unknown')}, preview: {doc.page_content[:80]}")
        return docs
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise