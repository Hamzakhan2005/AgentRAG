import logging
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import HF_TOKEN

logger = logging.getLogger(__name__)

def get_embedding_model():
    logger.info("Loading HuggingFace embedding model: all-MiniLM-L6-v2")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise