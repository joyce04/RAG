"""
llm.py
------
Single place to configure and instantiate all LLM clients.

All models run locally via Ollama. `get_llms()` is cached so the
connections are created only once per process.

Model roles:
  planner       - structured JSON planning (llama3.1:8b)
  drafter       - free-form text generation (qwen2:7b)
  sql_coder     - SQL generation against DuckDB (qwen2:7b)
  director      - high-level evaluation / orchestration qwen2.5:14b #(llama3:70b)
  embedding_model - dense embeddings for FAISS (nomic-embed-text)

Alternative models (swap by changing the string below):
  planner:        llama3.2:3b, mistral:7b
  drafter:        qwen2.5:7b, mistral:7b
  sql_coder:      sqlcoder:7b, deepseek-coder:6.7b
  director:       llama3.1:70b, qwen2.5:72b
  embedding_model: mxbai-embed-large (higher quality), all-minilm (fastest)
"""

from functools import lru_cache

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings


@lru_cache(maxsize=1)  # construct once; subsequent calls return the cached dict
def get_llms() -> dict:
    """Return a dict of all configured LLM / embedding clients."""
    return {
        "planner":         ChatOllama(model="llama3.1:8b", temperature=0.0, format="json"),
        "drafter":         ChatOllama(model="qwen2:7b",              temperature=0.2),
        "sql_coder":       ChatOllama(model="qwen2:7b",              temperature=0.0),
        "director":        ChatOllama(model="qwen2.5:14b",             temperature=0.0, format="json"), #"llama3:70b"
        "embedding_model": OllamaEmbeddings(model="nomic-embed-text"),
    }
