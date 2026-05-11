"""
drugsop.py
----------
Defines `DrugSOP` — the evolvable configuration object for the
Treatment & Drug Relationship RAG system.

Replaces TeamSOP. All fields are tunable by the SOP Architect between
evolution cycles.
"""

from pydantic import BaseModel, Field


class DrugSOP(BaseModel):
    """Standard Operating Procedure for the Drug Relationship Team."""

    planner_prompt: str = Field(
        description="System prompt for the Planner. Controls how the drug "
                    "concept is decomposed into specialist sub-tasks."
    )

    synthesizer_prompt: str = Field(
        description="System prompt for the Synthesizer. Controls the format "
                    "and depth of the Drug Relationship Report."
    )

    pharmacology_retriever_k: int = Field(
        default=5,
        description="Number of DrugBank/KEGG documents the Pharmacology "
                    "Specialist retrieves."
    )

    ddi_retriever_k: int = Field(
        default=8,
        description="Number of DDI corpus documents the Interaction Specialist "
                    "retrieves. Higher because interactions are numerous."
    )

    evidence_retriever_k: int = Field(
        default=5,
        description="Number of PubMed documents the Clinical Evidence "
                    "Specialist retrieves."
    )

    synthesizer_model: str = Field(
        default="qwen2:7b",
        description="Ollama model for the Synthesizer. Must be already pulled."
    )

    use_pathway_analyst: bool = Field(
        default=True,
        description="Whether to run the Pathway Analyst (KEGG retrieval)."
    )

    use_mimic_analyst: bool = Field(
        default=True,
        description="Whether to run the MIMIC Prescribing Analyst (PRESCRIPTIONS queries)."
    )

    use_interaction_db_analyst: bool = Field(
        default=True,
        description="Whether to run the Interaction DB Analyst (DrugBank/ChEMBL DuckDB)."
    )

    min_interaction_severity: str = Field(
        default="moderate",
        description="Minimum severity to surface in Interaction Map. "
                    "One of: minor, moderate, major, contraindicated."
    )

    top_coprescription_k: int = Field(
        default=10,
        description="How many top co-prescriptions to surface from MIMIC-III."
    )
