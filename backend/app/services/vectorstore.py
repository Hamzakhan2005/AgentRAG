import logging
import chromadb
from langchain_chroma import Chroma
from app.services.embeddings import get_embedding_model
from app.config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"

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
        logger.error(f"Failed to clear vectorstore: {e}")
        raise

def get_document_count() -> int:
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
        logger.info(f"Vectorstore document count: {count}")
        return count
    except Exception as e:
        logger.error(f"Failed to get document count: {e}")
        return 0