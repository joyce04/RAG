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
    chat_history = state.get('chat_history', [])

    generation_obj = generation_chain.invoke({
        'context': documents,
        'question': question,
        'chat_history': chat_history
    })

    retry_count = state.get('retry_count', 0) + 1

    return {
        'question': question,
        'documents': documents,
        'generation': generation_obj.answer if hasattr(generation_obj, 'answer') else str(generation_obj),
        'references': [dict(ref.dict(), source=__import__('os').path.basename(ref.source)) for ref in generation_obj.references] if hasattr(generation_obj, 'references') else [],
        'retry_count': retry_count,
    }