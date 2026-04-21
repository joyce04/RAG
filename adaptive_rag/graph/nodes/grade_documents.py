import logging
from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

logger = logging.getLogger(__name__)

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    determine whether the retrieved documents are relevant to the question
    """
    logger.info("grade_documents")
    
    question = state["question"]
    documents = state['documents']

    inputs = [{"question": question, "document": d.page_content} for d in documents]
    scores = retrieval_grader.batch(inputs)

    filtered_docs = []
    for d, score in zip(documents, scores):
        if score.binary_score.lower() == 'yes':
            logger.info("document is relevant")
            filtered_docs.append(d)
        else:
            logger.info("document is not relevant")

    web_search = len(filtered_docs) == 0

    return {"documents": filtered_docs, "question": question, "web_search": web_search}
        