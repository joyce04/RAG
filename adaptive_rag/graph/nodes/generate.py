import os
import difflib
import logging
import unicodedata
from typing import Any, Dict, List

from langchain_core.documents import Document

from graph.chains.generator import generation_chain
from graph.state import GraphState

logger = logging.getLogger(__name__)


def _nfc(s: str) -> str:
    return unicodedata.normalize('NFC', s)


def _format_context(documents: List[Document]) -> str:
    """Format documents with their actual filename so the LLM can cite them correctly."""
    parts = []
    for doc in documents:
        raw_source = doc.metadata.get('source', '') if isinstance(doc, Document) else ''
        if not raw_source:
            parts.append(doc.page_content if isinstance(doc, Document) else str(doc))
            continue
        # Show NFC filename so LLM can copy it exactly
        filename = _nfc(os.path.basename(raw_source))
        parts.append(f"[출처: {filename}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_source_map(documents: List[Document]) -> dict:
    """Return {nfc_filename: original_filename} for all retrieved docs."""
    mapping = {}
    for doc in documents:
        if not isinstance(doc, Document):
            continue
        raw_source = doc.metadata.get('source', '')
        if not raw_source:
            continue
        original = os.path.basename(raw_source)
        mapping[_nfc(original)] = original
    return mapping


def _resolve_filename(llm_filename: str, source_map: dict) -> str | None:
    """Map an LLM-generated filename (NFC) to the actual original filename."""
    name = _nfc(os.path.basename(llm_filename))
    # Exact match after NFC normalization
    if name in source_map:
        return source_map[name]
    # Stem-based partial match
    stem = os.path.splitext(name)[0]
    for nfc_key, original in source_map.items():
        nfc_stem = os.path.splitext(nfc_key)[0]
        if stem in nfc_stem or nfc_stem in stem:
            return original
    # Fuzzy match as last resort
    matches = difflib.get_close_matches(name, list(source_map.keys()), n=1, cutoff=0.4)
    if matches:
        return source_map[matches[0]]
    return None


def generate(state: GraphState) -> Dict[str, Any]:
    logger.info("[Generate] Generating answer from retrieved documents")

    question = state['question']
    documents = state['documents']
    chat_history = state.get('chat_history', [])

    formatted_context = _format_context(documents)
    source_map = _build_source_map(documents)

    generation_obj = generation_chain.invoke({
        'context': formatted_context,
        'question': question,
        'chat_history': chat_history
    })

    retry_count = state.get('retry_count', 0) + 1

    raw_refs = generation_obj.references if hasattr(generation_obj, 'references') else []
    seen = set()
    unique_refs = []
    for ref in raw_refs:
        filename = _resolve_filename(ref.source, source_map)
        if not filename:
            logger.warning("Could not resolve reference source '%s' — skipping", ref.source)
            continue
        key = (filename, ref.page)
        if key not in seen:
            seen.add(key)
            unique_refs.append({"source": filename, "page": ref.page, "snippet": ref.snippet})

    return {
        'question': question,
        'documents': documents,
        'generation': generation_obj.answer if hasattr(generation_obj, 'answer') else str(generation_obj),
        'references': unique_refs,
        'retry_count': retry_count,
    }
