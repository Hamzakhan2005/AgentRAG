# vectorstore.py — full replacement (same public API, adds vectorstore caching)
import logging
import re
import chromadb
from langchain_chroma import Chroma
from app.services.embeddings import get_embedding_model
from app.config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
MAX_DOCUMENTS_PER_SESSION = 15

_uploaded_filenames: list[str] = []

# Singleton - was reconnecting to Chroma (and transitively reloading the
# embedding model, before that was cached too) on every single retrieval.
_vectorstore_instance = None


def session_document_count() -> int:
    return len(_uploaded_filenames)


def session_document_limit_reached() -> bool:
    return len(_uploaded_filenames) >= MAX_DOCUMENTS_PER_SESSION


def register_uploaded_document(filename: str) -> None:
    _uploaded_filenames.append(filename)
    logger.info(
        f"Session document count: {len(_uploaded_filenames)}/{MAX_DOCUMENTS_PER_SESSION}"
    )


def get_session_filenames() -> list[str]:
    return list(_uploaded_filenames)


def reset_session_tracking() -> None:
    _uploaded_filenames.clear()
    logger.info("Session document tracking reset")


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def detect_named_file(query: str) -> str | None:
    filenames = get_session_filenames()
    if not filenames:
        return None

    normalized_query = _normalize(query)
    matches = []

    for filename in filenames:
        stem = filename.rsplit(".", 1)[0]
        norm_stem = _normalize(stem)
        if len(norm_stem) < 3:
            continue
        if norm_stem in normalized_query:
            matches.append(filename)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        logger.info(f"Query ambiguously matches multiple files {matches} - treating as no match")
        return None
    return None


def is_broad_query(query: str) -> bool:
    """Detect queries that reference multiple/all uploaded documents at once
    (e.g. 'compare X and Y', 'total across all invoices', 'for each ticket').
    These need retrieval to guarantee coverage of every document rather than
    relying on pure embedding top-k, which competes documents against each
    other and can silently drop some."""
    q = query.lower()
    broad_markers = [
        "all ", "every ", "each ", "across ", "compare", "both",
        " and ", "combined", "total for", "list all",
    ]
    return any(marker in q for marker in broad_markers)


def get_vectorstore():
    global _vectorstore_instance
    if _vectorstore_instance is not None:
        return _vectorstore_instance

    logger.info(f"Connecting to ChromaDB at: {CHROMA_PERSIST_DIR}")
    try:
        embeddings = get_embedding_model()
        _vectorstore_instance = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info("ChromaDB vectorstore connected successfully")
        return _vectorstore_instance
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise


def add_documents(docs: list):
    logger.info(f"Adding {len(docs)} chunks to vectorstore")
    try:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(docs)
        logger.info("Documents added to vectorstore successfully")
    except Exception as e:
        logger.error(f"Failed to add documents to vectorstore: {e}")
        raise


def get_all_chunks_grouped_by_source() -> dict:
    """Pull every chunk in the collection, grouped by source filename.
    Used for broad/'all documents' queries so we guarantee every uploaded
    file gets at least some representation, instead of letting embedding
    top-k competition silently drop some documents."""
    try:
        vectorstore = get_vectorstore()
        raw = vectorstore._collection.get(include=["documents", "metadatas"])
        grouped = {}
        for text, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
            source = (meta or {}).get("source", "unknown")
            grouped.setdefault(source, []).append(text)
        return grouped
    except Exception as e:
        logger.error(f"Failed to fetch all chunks grouped by source: {e}")
        return {}


def clear_vectorstore():
    global _vectorstore_instance
    logger.info("Clearing all documents from vectorstore")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        client.delete_collection(COLLECTION_NAME)
        logger.info("Vectorstore cleared successfully")
    except Exception as e:
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            logger.info("No existing collection to clear (fresh install) - skipping")
        else:
            logger.error(f"Failed to clear vectorstore: {e}")
    finally:
        _vectorstore_instance = None  # force reconnect next call, since collection changed
        reset_session_tracking()


def get_document_count() -> int:
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
        logger.info(f"Vectorstore document count: {count}")
        return count
    except Exception as e:
        logger.error(f"Failed to get document count: {e}")
        return 0