import logging
import re
import chromadb
from langchain_chroma import Chroma
from app.services.embeddings import get_embedding_model
from app.config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"

# Max number of distinct files that can be indexed in one server run.
# Purely in-memory - resets automatically on every server restart, which
# is exactly when we also want a clean vectorstore (see clear_vectorstore()
# called from main.py's startup hook).
MAX_DOCUMENTS_PER_SESSION = 15

# Tracks filenames uploaded during THIS process's lifetime only.
# No file/db needed - restart already gives us a clean slate for free.
_uploaded_filenames: list[str] = []


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
    """Lowercase and strip everything except letters/digits, so 'WPR 4', 'wpr4',
    'WPR-4.pdf' all normalize to the same comparable token."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def detect_named_file(query: str) -> str | None:
    """If the query clearly names one of the currently-uploaded files
    (by filename, ignoring spacing/punctuation/extension), return that
    filename. Returns None if there's no confident single match - callers
    should fall back to unscoped behavior in that case, which is always safe.
    Shared by the retriever (to scope vector search) and the router (to
    force 'rag' deterministically, skipping an LLM call)."""
    filenames = get_session_filenames()
    if not filenames:
        return None

    normalized_query = _normalize(query)
    matches = []

    for filename in filenames:
        stem = filename.rsplit(".", 1)[0]  # drop extension
        norm_stem = _normalize(stem)
        # Skip stems too short to be a reliable, specific match (avoids
        # false positives like a 1-2 char stem matching common words).
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


def get_vectorstore():
    logger.info(f"Connecting to ChromaDB at: {CHROMA_PERSIST_DIR}")
    try:
        embeddings = get_embedding_model()
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info("ChromaDB vectorstore connected successfully")
        return vectorstore
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

def clear_vectorstore():
    logger.info("Clearing all documents from vectorstore")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        client.delete_collection(COLLECTION_NAME)
        logger.info("Vectorstore cleared successfully")
    except Exception as e:
        # On a fresh install/first run there's no collection to delete yet -
        # that's expected, not a real failure. Only log unexpected errors loudly.
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            logger.info("No existing collection to clear (fresh install) - skipping")
        else:
            logger.error(f"Failed to clear vectorstore: {e}")
    finally:
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