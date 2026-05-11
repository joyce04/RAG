"""
states.py
---------
Shared data models for the LangGraph state and inter-node contracts.

  TeamState   - the mutable dict passed through every node in the graph
  AgentOutput - a single specialist agent's findings, stored in TeamState
"""

from typing import List, Optional, Any

from pydantic import BaseModel
from typing_extensions import TypedDict

from graph.drugsop import DrugSOP


class AgentOutput(BaseModel):
    """Holds the result of one specialist agent's retrieval + reasoning."""
    agent_name: str
    findings: Any


class TeamState(TypedDict):
    """
    The shared mutable state that flows through every LangGraph node.

    Fields
    ------
    initial_request          : the drug name / treatment concept entered by the user
    sop                      : the DrugSOP config governing this run
    plan                     : JSON plan produced by the Planner node
    agent_outputs            : list of AgentOutput objects, one per specialist
    drug_relationship_report : the synthesised Drug Relationship Report
    """
    initial_request:          str
    sop:                      DrugSOP
    plan:                     Optional[dict]
    agent_outputs:            List[AgentOutput]
    drug_relationship_report: Optional[str]
