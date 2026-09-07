import logging
from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY

logger = logging.getLogger(__name__)

def get_llm(streaming: bool = False):
    logger.info(f"Initializing Groq LLM, streaming={streaming}")
    try:
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=GROQ_API_KEY,
            temperature=0.2,
            streaming=streaming,
        )
        logger.info("Groq LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {e}")
        raise