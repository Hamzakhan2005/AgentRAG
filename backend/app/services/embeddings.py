# embeddings.py — full replacement
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import HF_TOKEN

logger = logging.getLogger(__name__)

# Singleton - was reloading the sentence-transformers model from disk/HF
# cache on every retrieval call (~6-8s each). Load once, reuse forever.
_embedding_instance = None


def get_embedding_model():
    global _embedding_instance
    if _embedding_instance is not None:
        return _embedding_instance

    logger.info("Loading HuggingFace embedding model: all-MiniLM-L6-v2")
    try:
        _embedding_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully")
        return _embedding_instance
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise