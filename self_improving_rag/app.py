"""
app.py
------
Streamlit chat UI for the Self-Improving RAG — Treatment & Drug Relationship System.

Tabs
----
  Chat      — multi-turn input, per-node progress, score cards, agent findings
  Gene Pool — leaderboard, Pareto frontier chart, evolution trigger

Run
---
  uv run streamlit run app.py
"""

import queue
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

from graph.drugsop import DrugSOP
from graph.graph import build_team_graph
from main import (
    initialize_pipeline,
    identify_pareto_front,
    _BASELINE_PLANNER_PROMPT,
    _BASELINE_SYNTHESIZER_PROMPT,
)
from ui.session import init_session_state, save_gene_pool, load_gene_pool
from ui.runner import run_pipeline_in_thread, run_evolution_in_thread
from ui.components import (
    render_evaluation_scores,
    render_agent_findings,
    render_leaderboard,
    render_sop_controls,
)
from ui.charts import pareto_figure

load_dotenv()

st.set_page_config(
    page_title="Drug Relationship RAG",
    page_icon="💊",
    layout="wide",
)

init_session_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💊 Drug Relationship RAG")
    st.markdown("---")

    if not st.session_state.pipeline_ready:
        if st.button("▶ Initialize Pipeline", type="primary", use_container_width=True):
            with st.status("Initialising pipeline…", expanded=True) as status:
                try:
                    st.write("Downloading corpora & building databases…")
                    llms, knowledge_stores = initialize_pipeline()
                    st.session_state.llms             = llms
                    st.session_state.knowledge_stores = knowledge_stores
                    st.session_state.team_graph       = build_team_graph(llms, knowledge_stores)
                    st.session_state.pipeline_ready   = True

                    saved_pool = load_gene_pool()
                    if saved_pool and saved_pool.pool:
                        st.session_state.gene_pool = saved_pool
                        st.write(f"Loaded {len(saved_pool.pool)} gene pool entries from disk.")

                    st.session_state.active_sop = DrugSOP(
                        planner_prompt=_BASELINE_PLANNER_PROMPT,
                        synthesizer_prompt=_BASELINE_SYNTHESIZER_PROMPT,
                    )
                    status.update(label="Pipeline ready!", state="complete")
                except Exception as e:
                    status.update(label=f"Error: {e}", state="error")
                    st.exception(e)

        st.info("Click **Initialize Pipeline** to download data, build FAISS stores, and connect to Ollama.")
    else:
        st.success("Pipeline ready")
        st.markdown("---")

        new_sop = render_sop_controls(st.session_state.active_sop)
        if new_sop != st.session_state.active_sop:
            st.session_state.active_sop = new_sop

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Pool", use_container_width=True):
                save_gene_pool(st.session_state.gene_pool)
                st.toast("Gene pool saved.")
        with col2:
            if st.button("📂 Load Pool", use_container_width=True):
                loaded = load_gene_pool()
                if loaded:
                    st.session_state.gene_pool = loaded
                    st.toast(f"Loaded {len(loaded.pool)} entries.")
                else:
                    st.toast("No saved pool found.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_pool = st.tabs(["💬 Chat", "🧬 Gene Pool"])


# ===========================================================================
# Tab 1: Chat
# ===========================================================================

with tab_chat:
    st.header("Drug / Treatment Concept → Relationship Report")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                st.markdown("**Drug Relationship Report**")
                st.markdown(msg["content"])
                if msg.get("eval_result"):
                    st.markdown("**Evaluation Scores**")
                    render_evaluation_scores(msg["eval_result"])
                if msg.get("agent_outputs"):
                    st.markdown("**Agent Findings**")
                    render_agent_findings(msg["agent_outputs"])

    user_input = st.chat_input(
        "Enter a drug name or treatment concept…",
        disabled=not st.session_state.pipeline_ready,
    )

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        status_q: queue.Queue = queue.Queue()
        run_pipeline_in_thread(
            request=user_input,
            sop=st.session_state.active_sop,
            team_graph=st.session_state.team_graph,
            gene_pool=st.session_state.gene_pool,
            llms=st.session_state.llms,
            status_q=status_q,
            save_fn=save_gene_pool,
        )

        with st.chat_message("assistant"):
            with st.status("Running pipeline…", expanded=True) as run_status:
                final_state = None
                eval_result = None
                while True:
                    try:
                        msg_type, payload = status_q.get(timeout=300)
                    except queue.Empty:
                        run_status.update(label="Timed out.", state="error")
                        break

                    if msg_type == "PROGRESS":
                        st.write(payload)
                    elif msg_type == "RESULT":
                        final_state, eval_result = payload
                        run_status.update(label="Pipeline complete!", state="complete")
                        break
                    elif msg_type == "ERROR":
                        run_status.update(label="Pipeline error.", state="error")
                        st.error(payload)
                        break

            if final_state and eval_result:
                report_text = final_state.get("drug_relationship_report", "")
                st.markdown("**Drug Relationship Report**")
                st.markdown(report_text)

                st.markdown("**Evaluation Scores**")
                render_evaluation_scores(eval_result)

                agent_outputs = final_state.get("agent_outputs", [])
                if agent_outputs:
                    st.markdown("**Agent Findings**")
                    render_agent_findings(agent_outputs)

                st.session_state.chat_history.append({
                    "role":          "assistant",
                    "content":       report_text,
                    "eval_result":   eval_result,
                    "agent_outputs": agent_outputs,
                })


# ===========================================================================
# Tab 2: Gene Pool
# ===========================================================================

with tab_pool:
    st.header("SOP Gene Pool — Leaderboard")

    gene_pool    = st.session_state.gene_pool
    pareto_sops  = identify_pareto_front(gene_pool) if gene_pool.pool else []
    pareto_versions = {e["version"] for e in pareto_sops}

    render_leaderboard(gene_pool, pareto_versions)

    if pareto_sops:
        st.subheader("Pareto Frontier")
        fig = pareto_figure(pareto_sops)
        if fig:
            st.pyplot(fig)
        else:
            st.info("Need at least 2 Pareto-optimal SOPs to render the frontier chart.")

    st.markdown("---")
    st.subheader("Run Evolution Cycle")
    evo_request = st.text_area(
        "Drug concept for evolution run",
        value=(
            "Analyse Warfarin: map its pharmacological profile, rank its top drug-drug "
            "interactions by severity, summarise clinical evidence for its use in atrial "
            "fibrillation and VTE, and show how frequently it is co-prescribed with "
            "Amiodarone, Aspirin, and Digoxin in real ICU patients from MIMIC-III."
        ),
        height=100,
    )

    if st.button(
        "🔬 Run Evolution Cycle",
        disabled=not st.session_state.pipeline_ready or not gene_pool.pool,
        type="primary",
    ):
        evo_q: queue.Queue = queue.Queue()
        run_evolution_in_thread(
            team_graph=st.session_state.team_graph,
            gene_pool=st.session_state.gene_pool,
            llms=st.session_state.llms,
            trial_request=evo_request,
            status_q=evo_q,
            save_fn=save_gene_pool,
        )

        with st.status("Running evolution cycle…", expanded=True) as evo_status:
            while True:
                try:
                    msg_type, payload = evo_q.get(timeout=600)
                except queue.Empty:
                    evo_status.update(label="Timed out.", state="error")
                    break

                if msg_type == "PROGRESS":
                    st.write(payload)
                elif msg_type == "RESULT":
                    evo_status.update(
                        label=f"Evolution complete — added {len(payload)} new SOP(s) to pool.",
                        state="complete",
                    )
                    break
                elif msg_type == "ERROR":
                    evo_status.update(label="Evolution error.", state="error")
                    st.error(payload)
                    break

        st.rerun()


