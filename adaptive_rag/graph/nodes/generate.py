import logging
from typing import Any, Dict

from graph.chains.generator import generation_chain
from graph.state import GraphState

logger = logging.getLogger(__name__)

def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate answer using the vectorstore
    """
    logger.info("[Generate] Generating answer from retrieved documents")
    
    question = state['question']
    documents = state['documents']

    generation = generation_chain.invoke({
        'context': documents,
        'question': question
    })

    retry_count = state.get('retry_count', 0) + 1

    return {
        'question': question,
        'documents': documents,
        'generation': generation,
        'retry_count': retry_count,
    }