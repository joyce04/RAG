"""
diagnostician.py
----------------
The Performance Diagnostician: analyses the five-dimensional evaluation
vector and identifies the single biggest weakness in the current SOP.

Output is a `Diagnosis` object consumed by the SOP Architect to guide
the next round of mutations.
"""

from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph.evaluator import _invoke_structured

from graph.evaluator import EvaluationResult


class Diagnosis(BaseModel):
    """
    Structured output produced by the Diagnostician LLM.

    primary_weakness    : which of the five dimensions is weakest
    root_cause_analysis : explanation referencing the specific scores
    recommendation      : strategic suggestion for the SOP Architect
    """
    primary_weakness:    Literal['rigor', 'compliance', 'ethics', 'feasibility', 'simplicity']
    root_cause_analysis: str = Field(
        description="Detailed analysis of why the weakness occurred, referencing specific scores."
    )
    recommendation:      str = Field(
        description="High-level recommendation for how to modify the SOP to address the weakness."
    )


def performance_diagnostician(eval_result: EvaluationResult, llms: dict) -> Diagnosis:
    """
    Use the director LLM (most capable local model) to analyse the
    5D evaluation report and return a structured Diagnosis.

    `llms` is passed explicitly — no global state.
    """
    print("--- EXECUTING PERFORMANCE DIAGNOSTICIAN ---")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a world-class management consultant specialising in process "
            "optimisation. Analyse a performance scorecard and identify the single "
            "biggest weakness. Provide a root cause analysis and a strategic "
            "recommendation for fixing it.",
        ),
        (
            "human",
            "Please analyse the following performance evaluation report:\n\n{report}",
        ),
    ])

    # Serialise the evaluation result to JSON so the LLM sees all scores.
    # _invoke_structured handles JSON parsing + field-name enforcement.
    return _invoke_structured(llms['director'], prompt, Diagnosis,
                              {"report": eval_result.model_dump_json()})
