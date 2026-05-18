import logging
from app.services.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

TOP_K = 4

def retrieve_docs(query: str) -> list:
    logger.info(f"Retrieving docs for query: '{query}'")
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} chunks")
        for i, doc in enumerate(docs):
            logger.debug(f"Chunk {i+1} source: {doc.metadata.get('source', 'unknown')}, preview: {doc.page_content[:80]}")
        return docs
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise