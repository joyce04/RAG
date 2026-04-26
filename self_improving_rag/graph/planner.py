"""
planner.py
----------
The Planner agent: receives the high-level trial concept and breaks it
into a structured JSON plan that assigns sub-tasks to each specialist.

Uses a factory pattern (`make_planner_node`) so the LLM dict is closed
over rather than accessed as a global variable.
"""

import json
from typing import Callable

from graph.states import TeamState


def make_planner_node(llms: dict) -> Callable[[TeamState], dict]:
    """
    Factory that binds `llms` and returns a LangGraph-compatible node
    function with signature `(state: TeamState) -> dict`.

    The returned node:
      1. Reads the planner prompt from the active SOP.
      2. Calls the `planner` LLM in JSON mode (llama3.1:8b-instruct).
      3. Writes the resulting plan dict back into state.
    """
    planner_llm = llms['planner']

    def planner_agent(state: TeamState) -> dict:
        sop = state['sop']

        # Combine the SOP's system prompt with the user's trial concept
        prompt = f"{sop.planner_prompt}\n\nTrial Concept: '{state['initial_request']}'"
        print(f"[Planner] Generating plan...")

        response = planner_llm.invoke(prompt)

        # The planner LLM runs in JSON mode; response.content is a JSON string
        plan = json.loads(response.content) if isinstance(response.content, str) else response
        print(f"[Planner] Plan generated:\n{json.dumps(plan, indent=2)}")

        return {**state, "plan": plan}

    return planner_agent
