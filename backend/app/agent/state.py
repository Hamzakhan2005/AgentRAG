from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    query: str                        # original user query
    rewritten_query: Optional[str]    # rewritten query if grader fails
    documents: List[Document]         # retrieved documents
    answer: Optional[str]             # final generated answer
    sources: List[str]                # source filenames
    web_search_used: bool             # did we fall back to web search
    retry_count: int                  # how many times we've retried
    route: Optional[str]              # router decision: rag / websearch / direct
    chat_history: List[BaseMessage] 