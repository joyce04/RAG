"""
teamsop.py
----------
Defines `TeamSOP` — the evolvable configuration object that controls
how every agent in the Team graph behaves.

The SOP is passed through graph state on every run. The SOP Architect
(graph/architect.py) mutates it between evolution cycles to improve
multi-dimensional evaluation scores.
"""

from pydantic import BaseModel, Field


class TeamSOP(BaseModel):
    """
    Standard Operating Procedure for the clinical trial Team.

    Each field is a tunable hyper-parameter that the self-improvement
    loop can modify to address a diagnosed weakness.
    """

    planner_prompt: str = Field(
        description="System prompt for the Planner agent. Controls how the "
                    "trial concept is decomposed into specialist sub-tasks."
    )

    researcher_retriever_k: int = Field(
        default=3,
        description="Number of PubMed documents the Medical Researcher retrieves. "
                    "Higher k = more context but slower and noisier."
    )

    synthesizer_prompt: str = Field(
        description="System prompt for the Criteria Synthesizer. Controls tone, "
                    "structure, and strictness of the final output document."
    )

    # Constrained to models that are available locally via Ollama
    synthesizer_model: str = Field(
        default="qwen2:7b",
        description="Ollama model name for the Synthesizer. "
                    "Must be a model already pulled via `ollama pull`."
    )

    use_sql_analyst: bool = Field(
        default=True,
        description="Whether to run the Patient Cohort Analyst (SQL agent). "
                    "Disable to skip DuckDB queries and speed up the run."
    )

    use_ethics_specialist: bool = Field(
        default=True,
        description="Whether to include the Ethics Specialist agent. "
                    "Disable to skip ethics retrieval."
    )
