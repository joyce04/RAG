import os
import logging
import unicodedata
from typing import Any, Dict, List

from langchain_core.documents import Document

from graph.chains.generator import generation_chain
from graph.state import GraphState

logger = logging.getLogger(__name__)


def _nfc(s: str) -> str:
    return unicodedata.normalize('NFC', s)


def _format_context(documents: List[Document], index_map: dict) -> str:
    """
    Format documents with [문서 N] indices.
    Populates index_map: {N: {"source": filename, "page": page_num}}
    so the LLM only needs to cite the index number.
    """
    parts = []
    for i, doc in enumerate(documents):
        idx = i + 1
        raw_source = doc.metadata.get('source', '') if isinstance(doc, Document) else ''
        filename = _nfc(os.path.basename(raw_source)) if raw_source else ''
        page = doc.metadata.get('page', 1) if isinstance(doc, Document) else 1
        index_map[idx] = {"source": filename, "page": page}
        content = doc.page_content if isinstance(doc, Document) else str(doc)
        header = f"[문서 {idx}]" if filename else f"[문서 {idx}]"
        parts.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(parts)


def generate(state: GraphState) -> Dict[str, Any]:
    logger.info("[Generate] Generating answer from retrieved documents")

    question = state['question']
    documents = state['documents']
    chat_history = state.get('chat_history', [])

    index_map: dict = {}
    formatted_context = _format_context(documents, index_map)

    generation_obj = generation_chain.invoke({
        'context': formatted_context,
        'question': question,
        'chat_history': chat_history
    })

    retry_count = state.get('retry_count', 0) + 1

    raw_refs = generation_obj.references if hasattr(generation_obj, 'references') else []
    seen: set = set()
    unique_refs = []
    for ref in raw_refs:
        idx = getattr(ref, 'source_index', None)
        doc_info = index_map.get(idx)
        if not doc_info or not doc_info.get('source'):
            logger.warning("Invalid source_index %s — skipping", idx)
            continue
        key = (doc_info['source'], doc_info['page'])
        if key not in seen:
            seen.add(key)
            unique_refs.append({
                "source": doc_info['source'],
                "page": doc_info['page'],
                "snippet": ref.snippet,
            })

    return {
        'question': question,
        'documents': documents,
        'generation': generation_obj.answer if hasattr(generation_obj, 'answer') else str(generation_obj),
        'references': unique_refs,
        'retry_count': retry_count,
    }
