import logging
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

logger.info("Initializing in-memory checkpointer for session memory")
memory = MemorySaver()