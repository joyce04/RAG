from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represent the state of the graph.
    """
    question: str
    documents: List[str]
    generation: str
    web_search: bool