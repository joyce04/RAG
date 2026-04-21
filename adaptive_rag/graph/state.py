from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represent the state of the graph.
    """
    question: str
    documents: List[Document]
    generation: str
    web_search: bool
    retry_count: int  # number of generation attempts; caps at MAX_RETRIES in graph.py