from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represent the state of the graph.
    """
    question: str
    chat_history: List[dict]
    documents: List[Document]
    generation: str
    references: list
    web_search: bool
    retry_count: int  # number of generation attempts; caps at MAX_RETRIES in graph.py