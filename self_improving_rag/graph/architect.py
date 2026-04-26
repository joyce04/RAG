"""
architect.py
------------
The SOP Architect: the mutation engine of the self-improvement loop.

Given a Diagnosis of the current SOP's weakness, it generates 2–3
new TeamSOP variants ('mutations') that attempt to address that weakness.
The mutations are diverse: different prompts, toggled agents, tuned k, or
alternative synthesizer models.
"""

from typing import List

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from graph.teamsop import TeamSOP
from graph.diagnostician import Diagnosis


class EvolvedSOPs(BaseModel):
    """Container returned by the SOP Architect — a list of mutated SOPs."""
    mutations: List[TeamSOP]


def sop_architect(diagnosis: Diagnosis, current_sop: TeamSOP, llms: dict) -> EvolvedSOPs:
    """
    Generate 2–3 new TeamSOP mutations that target the diagnosed weakness.

    The director LLM is given:
      - the full TeamSOP JSON schema (so it knows what fields exist)
      - the current SOP's values
      - the Diagnosis (primary weakness + recommendation)

    It returns a list of valid TeamSOP objects under the 'mutations' key.

    `llms` is passed explicitly — no global state.
    """
    print("--- EXECUTING SOP ARCHITECT ---")

    architect_llm = llms['director'].with_structured_output(EvolvedSOPs)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You are an AI process architect. Your job is to modify a process "
            f"configuration (an SOP) to fix a diagnosed performance problem.\n\n"
            f"The SOP schema is:\n{TeamSOP.model_json_schema()}\n\n"
            f"Return a JSON object with a single key 'mutations' containing a list "
            f"of 2–3 new, valid SOP objects. Propose diverse and creative mutations: "
            f"you may change prompts, toggle agents, adjust retrieval k, or change "
            f"the synthesizer model. Only modify fields relevant to the diagnosis.",
        ),
        (
            "human",
            "Current SOP:\n{current_sop}\n\n"
            "Performance diagnosis:\n{diagnosis}\n\n"
            "Generate 2–3 improved SOPs.",
        ),
    ])

    return (prompt | architect_llm).invoke({
        "current_sop": current_sop.model_dump_json(),
        "diagnosis":   diagnosis.model_dump_json(),
    })
