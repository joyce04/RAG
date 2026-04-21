import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm(temperature: float = 0) -> ChatOpenAI:
    """Return a configured OpenRouter LLM instance."""
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        temperature=temperature,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

# Pre-instantiated default LLM for chains
default_llm = get_llm(temperature=0)
