"""
graph.py
--------
Assembles the LangGraph Team workflow and exposes `build_team_graph`.

Graph topology (linear):
  planner → execute_specialists → synthesizer → END

Specialist routing:
  "Pharmacology*"    → drugbank_retriever (FAISS)
  "Interaction Spec" → ddi_retriever (FAISS)
  "Clinical*"        → pubmed_retriever (FAISS)
  "Pathway*"         → kegg_retriever (FAISS) — toggle: use_pathway_analyst
  "MIMIC*"           → MIMIC DuckDB PRESCRIPTIONS — toggle: use_mimic_analyst
  "Interaction DB*"  → DrugBank/DDI DuckDB — toggle: use_interaction_db_analyst
"""

from langgraph.graph import StateGraph, END

from graph.states import TeamState
from graph.planner import make_planner_node
from graph.retriever import make_retrieval_agent
from graph.analyst import make_analyst
from graph.synthesizer import criteria_synthesizer


def build_team_graph(llms: dict, knowledge_stores: dict):
    """
    Build and compile the Team LangGraph workflow.

    Parameters
    ----------
    llms             : dict from get_llms()
    knowledge_stores : dict from create_retrievers()

    Returns a compiled LangGraph runnable.
    """
    planner_node    = make_planner_node(llms)
    retrieval_agent = make_retrieval_agent(knowledge_stores)
    analyst         = make_analyst(knowledge_stores, llms)

    def specialist_execution_node(state: TeamState) -> dict:
        """Route each planner task to the correct specialist agent."""
        plan_tasks = state['plan'].get('plan', [])
        outputs = []

        for task in plan_tasks:
            agent_name = task.get('agent', '')
            task_desc  = task.get('task_description', '')

            if "Pharmacology" in agent_name:
                output = retrieval_agent(task_desc, state, "drugbank_retriever", "Pharmacology Specialist")

            elif "Interaction Specialist" in agent_name or (
                "Interaction" in agent_name and "DB" not in agent_name
            ):
                output = retrieval_agent(task_desc, state, "ddi_retriever", "Interaction Specialist")

            elif "Clinical" in agent_name:
                output = retrieval_agent(task_desc, state, "pubmed_retriever", "Clinical Evidence Specialist")

            elif "Pathway" in agent_name:
                if not state['sop'].use_pathway_analyst:
                    continue
                output = retrieval_agent(task_desc, state, "kegg_retriever", "Pathway Analyst")

            elif "MIMIC" in agent_name:
                if not state['sop'].use_mimic_analyst:
                    continue
                output = analyst(task_desc, state, analyst_type="mimic")

            elif "Interaction DB" in agent_name:
                if not state['sop'].use_interaction_db_analyst:
                    continue
                output = analyst(task_desc, state, analyst_type="interaction_db")

            else:
                print(f"[Specialists] Unrecognised agent '{agent_name}' — skipping.")
                continue

            outputs.append(output)

        return {**state, "agent_outputs": outputs}

    workflow = StateGraph(TeamState)

    workflow.add_node("planner",             planner_node)
    workflow.add_node("execute_specialists", specialist_execution_node)
    workflow.add_node("synthesizer",         criteria_synthesizer)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner",             "execute_specialists")
    workflow.add_edge("execute_specialists", "synthesizer")
    workflow.add_edge("synthesizer",          END)

    return workflow.compile()
