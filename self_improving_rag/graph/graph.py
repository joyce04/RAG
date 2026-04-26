"""
graph.py
--------
Assembles the LangGraph Team workflow and exposes `build_team_graph`.

Graph topology (linear):
  planner → execute_specialists → synthesizer → END

Node descriptions:
  planner             : decomposes the trial concept into a structured plan
  execute_specialists : dispatches sub-tasks to the appropriate agents
                        (retrieval agents + optional SQL analyst)
  synthesizer         : synthesises all agent findings into final criteria

Dependencies (`llms` and `knowledge_stores`) are injected via factory
functions so no module-level globals are needed.
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
    llms             : dict returned by `get_llms()` — all LLM clients
    knowledge_stores : dict returned by `create_retrievers()` — FAISS
                       retrievers and the DuckDB path

    Returns
    -------
    A compiled LangGraph runnable (call `.invoke(state)` to run it).
    """

    # ------------------------------------------------------------------
    # Bind dependencies into each node via factory closures
    # ------------------------------------------------------------------
    planner_node    = make_planner_node(llms)
    retrieval_agent = make_retrieval_agent(knowledge_stores)
    analyst         = make_analyst(knowledge_stores, llms)

    # ------------------------------------------------------------------
    # Specialist dispatcher node
    # This node reads the planner's task list and routes each task to
    # the correct specialist agent based on the agent name in the plan.
    # ------------------------------------------------------------------
    def specialist_execution_node(state: TeamState) -> dict:
        """
        Execute all specialist sub-tasks produced by the planner.

        Routing logic:
          "Regulatory*"  → Retrieval agent on fda_retriever
          "Medical*"     → Retrieval agent on pubmed_retriever (k from SOP)
          "Ethics*"      → Retrieval agent on ethics_retriever (if enabled in SOP)
          "Cohort*"      → SQL analyst against DuckDB (if enabled in SOP)

        Any agent name not matched is silently skipped (e.g. when an
        agent is disabled in the SOP).
        """
        plan_tasks = state['plan'].get('plan', [])
        outputs = []

        for task in plan_tasks:
            agent_name = task.get('agent', '')
            task_desc  = task.get('task_description', '')

            if "Regulatory" in agent_name:
                output = retrieval_agent(task_desc, state, "fda_retriever", "Regulatory Specialist")

            elif "Medical" in agent_name:
                output = retrieval_agent(task_desc, state, "pubmed_retriever", "Medical Researcher")

            elif "Ethics" in agent_name:
                # Honour the SOP toggle — skip if disabled
                if not state['sop'].use_ethics_specialist:
                    continue
                output = retrieval_agent(task_desc, state, "ethics_retriever", "Ethics Specialist")

            elif "Cohort" in agent_name:
                # use_sql_analyst toggle is handled inside the analyst itself
                output = analyst(task_desc, state)

            else:
                print(f"[Specialists] Unrecognised agent '{agent_name}' — skipping.")
                continue

            outputs.append(output)

        return {**state, "agent_outputs": outputs}

    # ------------------------------------------------------------------
    # Wire the graph
    # ------------------------------------------------------------------
    workflow = StateGraph(TeamState)

    workflow.add_node("planner",             planner_node)
    workflow.add_node("execute_specialists", specialist_execution_node)
    workflow.add_node("synthesizer",         criteria_synthesizer)

    # Linear pipeline: planner → specialists → synthesizer → done
    workflow.set_entry_point("planner")
    workflow.add_edge("planner",             "execute_specialists")
    workflow.add_edge("execute_specialists", "synthesizer")
    workflow.add_edge("synthesizer",          END)

    return workflow.compile()
