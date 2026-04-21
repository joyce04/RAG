import logging
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

logger = logging.getLogger(__name__)

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Web search based on the rephrased question
    """
    logger.info("[Web Search] Web searching")

    question = state['question']
    
    if 'documents' in state:
        documents = state['documents']
    else:
        documents = None
    
    tavily_results = web_search_tool.invoke({'query': question})['results']

    joined_tavily_result = "\n".join([
        tavily_result['content'] for tavily_result in tavily_results
    ])

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {'question': question, 'documents': documents}

if __name__ == "__main__":
    web_search(state={"question": "2024년 1월 1일 이후에 발생한 공정거래법 관련 판례를 알려줘", "documents": None})