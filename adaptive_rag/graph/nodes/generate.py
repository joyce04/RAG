from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: StateGraph) -> Dict[str, Any]:
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

    return {'question':question, 
    'documents':documents, 
    'generation': generation}