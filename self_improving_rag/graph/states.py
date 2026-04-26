"""
states.py
---------
Shared data models used as the LangGraph state and inter-node contracts.

  TeamState   - the mutable dict passed through every node in the graph
  AgentOutput  - a single specialist agent's findings, stored in TeamState
"""

from typing import List, Dict, Any, Optional

from pydantic import BaseModel            # pydantic v2 (project pins pydantic==2.x)
from typing_extensions import TypedDict

from graph.teamsop import TeamSOP


class AgentOutput(BaseModel):
    """Holds the result of one specialist agent's retrieval + reasoning."""

    agent_name: str
    findings: Any   # str in practice; Any to accommodate future structured outputs


class TeamState(TypedDict):
    """
    The shared mutable state that flows through every LangGraph node.

    Fields
    ------
    initial_request : the raw trial concept entered by the user
    sop             : the TeamSOP config governing this run
    plan            : JSON plan produced by the Planner node
    agent_outputs   : list of AgentOutput objects, one per specialist
    final_criteria  : the synthesized inclusion/exclusion criteria text
    """

    initial_request: str
    sop:             TeamSOP
    plan:            Optional[Dict[str, Any]]
    agent_outputs:   List[AgentOutput]
    final_criteria:  Optional[str]
