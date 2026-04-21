import logging
from typing import Any, Dict

from graph.state import GraphState
from data.ingest import get_retriever

logger = logging.getLogger(__name__)

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents
    """

    logger.info("[Retrieve] Retrieving documents")

    question = state['question']
    documents = get_retriever().invoke(question)

    return {'question': question, 'documents': documents}