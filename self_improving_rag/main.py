"""
main.py
-------
Entry point for the Self-Improving RAG — Treatment & Drug Relationship system.

High-level flow:
  1. Download data  — DrugBank vocab, DDI corpus, PubMed abstracts, KEGG entries
  2. Build DBs      — MIMIC-III DuckDB (PRESCRIPTIONS), DrugBank/DDI DuckDB
  3. Build stores   — embed corpora into FAISS vector stores
  4. Baseline run   — run the Team graph with the seed DrugSOP
  5. Evaluate       — score the output on 5 dimensions
  6. Evolve         — diagnose weakness → mutate SOP → re-evaluate
  7. Analyse        — compute the Pareto front and visualise it
"""

import os
import json
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv

from llm import get_llms
from data.download_raw_data import (
    data_paths,
    prep_paths,
    download_pubmed_articles,
    download_drugbank_vocab,
    download_ddi_corpus,
    download_kegg_drug_entries,
)
from data.process_mimic import load_real_mimic_data
from data.process_drug_db import load_drug_databases
from data.process_unstructured import create_retrievers
from graph.drugsop import DrugSOP
from graph.graph import build_team_graph
from graph.evaluator import run_full_evaluation
from graph.diagnostician import performance_diagnostician
from graph.architect import sop_architect
from graph.sop_pool import SOPGenePool
from display import visualize_frontier

load_dotenv()

_BASELINE_PLANNER_PROMPT = (
    "You are a master pharmacology analyst. Your task is to receive a drug name or "
    "treatment concept and decompose it into a structured plan with specific sub-tasks "
    "for a team of specialists: a Pharmacology Specialist (mechanism of action, targets, "
    "drug class, PK properties), an Interaction Specialist (drug-drug interactions, "
    "severity, clinical management), a Clinical Evidence Specialist (RCT outcomes, "
    "meta-analyses, treatment guidelines by indication), a Pathway Analyst (biological "
    "pathways and target networks), a MIMIC Prescribing Analyst (real-world co-prescription "
    "frequency from MIMIC-III PRESCRIPTIONS), and an Interaction DB Analyst (known "
    "interaction count from DrugBank database). "
    "Output a JSON object with a single key 'plan' containing a list of tasks. "
    "Each task must have 'agent', 'task_description', and 'dependencies' keys."
)

_BASELINE_SYNTHESIZER_PROMPT = (
    "You are an expert clinical pharmacologist and medical writer. Your task is to "
    "synthesise the structured findings from all specialist teams into a formal "
    "'Drug Relationship Report'. Structure your output into exactly five sections:\n\n"
    "1. Drug Profile — mechanism of action, pharmacological class, molecular targets, "
    "half-life, clearance route\n"
    "2. Related Drugs — same class (similar MOA), competing mechanisms, synergistic "
    "combinations, biosimilars/generics\n"
    "3. Interaction Map — drug-drug interactions ranked by severity "
    "(Contraindicated → Major → Moderate → Minor), with mechanism for each\n"
    "4. Clinical Evidence by Indication — for each indication: evidence level "
    "(RCT/meta-analysis/observational), treatment line (first/second/adjunct), key trial names\n"
    "5. Real-World Prescribing Patterns (MIMIC-III) — use the MIMIC Prescribing Analyst "
    "output verbatim: admission count, top co-prescriptions with frequencies, and "
    "notable concurrent DDI pairs observed in real patients.\n\n"
    "Be concise, precise, and cite specific numbers wherever the specialist data provides them."
)


# ---------------------------------------------------------------------------
# Pipeline initialisation (shared by main() and Streamlit app)
# ---------------------------------------------------------------------------

def initialize_pipeline() -> tuple[dict, dict]:
    """
    Run Steps 1–3: download data, build DBs, build FAISS stores.

    Returns (llms, knowledge_stores) ready for build_team_graph().
    """
    print("=== Self-Improving RAG — Drug Relationship System — Initialising ===\n")

    prep_paths()
    llms = get_llms()
    print("LLM clients configured:")
    for name, client in llms.items():
        if hasattr(client, 'model'):
            print(f"  {name}: {client.model}")

    # Step 1: download corpora
    pubmed_query = "(drug interactions) AND (pharmacology) AND (clinical pharmacokinetics)"
    n_pubmed = download_pubmed_articles(pubmed_query)
    print(f"PubMed: {n_pubmed} article(s).")

    n_drugbank = download_drugbank_vocab()
    print(f"DrugBank vocab: {n_drugbank} file(s).")

    n_ddi = download_ddi_corpus()
    print(f"DDI corpus: {n_ddi} file(s).")

    n_kegg = download_kegg_drug_entries()
    print(f"KEGG entries: {n_kegg} drug(s).")

    # Step 2: build structured DBs
    mimic_db_path = load_real_mimic_data()
    if mimic_db_path:
        print(f"MIMIC-III database: {mimic_db_path}")
    else:
        print("MIMIC-III not available — MIMIC Prescribing Analyst will skip.")

    drugbank_db_path = load_drug_databases()
    if drugbank_db_path:
        print(f"DrugBank/DDI database: {drugbank_db_path}")
    else:
        print("DrugBank/DDI database not available — Interaction DB Analyst will skip.")

    # Step 3: embed and build FAISS stores
    knowledge_stores = create_retrievers(
        llms['embedding_model'],
        mimic_db_path,
        drugbank_db_path,
    )
    for name, store in knowledge_stores.items():
        print(f"  {name}: {store}")

    return llms, knowledge_stores


# ---------------------------------------------------------------------------
# Evolution cycle
# ---------------------------------------------------------------------------

def run_evolution_cycle(
    team_graph,
    gene_pool: SOPGenePool,
    llms: dict,
    trial_request: str,
) -> None:
    """
    One full cycle of diagnosis → mutation → evaluation.
    """
    print("\n" + "=" * 25 + " STARTING NEW EVOLUTION CYCLE " + "=" * 25)

    current_best   = gene_pool.get_latest_entry()
    parent_sop     = current_best['sop']
    parent_eval    = current_best['evaluation']
    parent_version = current_best['version']
    print(f"Improving upon SOP v{parent_version}...")

    diagnosis = performance_diagnostician(parent_eval, llms)
    print(
        f"Diagnosis: primary weakness = '{diagnosis.primary_weakness}'. "
        f"Recommendation: {diagnosis.recommendation}"
    )

    new_sop_candidates = sop_architect(diagnosis, parent_sop, llms)
    print(f"Generated {len(new_sop_candidates.mutations)} SOP candidate(s).")

    for i, candidate_sop in enumerate(new_sop_candidates.mutations):
        print(f"\n--- Testing SOP candidate {i + 1}/{len(new_sop_candidates.mutations)} ---")

        team_input = {"initial_request": trial_request, "sop": candidate_sop}
        final_state = team_graph.invoke(team_input)

        eval_result = run_full_evaluation(final_state, llms)
        gene_pool.add(sop=candidate_sop, eval_result=eval_result, parent_version=parent_version)

    print("\n" + "=" * 25 + " EVOLUTION CYCLE COMPLETE " + "=" * 26)


# ---------------------------------------------------------------------------
# Pareto-front analysis
# ---------------------------------------------------------------------------

def identify_pareto_front(gene_pool: SOPGenePool) -> List[Dict[str, Any]]:
    """Return non-dominated SOP entries from the gene pool."""
    pareto_front = []
    pool_entries = gene_pool.pool

    for i, candidate in enumerate(pool_entries):
        cand_scores = np.array([
            s['score'] for s in candidate['evaluation'].model_dump().values()
        ])
        is_dominated = False
        for j, other in enumerate(pool_entries):
            if i == j:
                continue
            other_scores = np.array([
                s['score'] for s in other['evaluation'].model_dump().values()
            ])
            if np.all(other_scores >= cand_scores) and np.any(other_scores > cand_scores):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(candidate)

    return pareto_front


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Self-Improving RAG — Treatment & Drug Relationship System ===\n")

    llms, knowledge_stores = initialize_pipeline()

    # Baseline SOP
    baseline_sop = DrugSOP(
        planner_prompt=_BASELINE_PLANNER_PROMPT,
        synthesizer_prompt=_BASELINE_SYNTHESIZER_PROMPT,
        pharmacology_retriever_k=5,
        ddi_retriever_k=8,
        evidence_retriever_k=5,
        synthesizer_model="qwen2:7b",
        use_pathway_analyst=True,
        use_mimic_analyst=True,
        use_interaction_db_analyst=True,
        min_interaction_severity="moderate",
        top_coprescription_k=10,
    )
    print("\nBaseline DrugSOP:")
    print(json.dumps(baseline_sop.model_dump(), indent=4))

    team_graph = build_team_graph(llms, knowledge_stores)

    test_request = (
        "Analyse Warfarin: map its pharmacological profile, rank its top drug-drug "
        "interactions by severity, summarise clinical evidence for its use in atrial "
        "fibrillation and VTE, and show how frequently it is co-prescribed with "
        "Amiodarone, Aspirin, and Digoxin in real ICU patients from MIMIC-III."
    )
    print("\nRunning Team graph with baseline DrugSOP...")
    final_result = team_graph.invoke({"initial_request": test_request, "sop": baseline_sop})

    print("\nDrug Relationship Report (Baseline SOP):")
    print("-" * 40)
    print(final_result.get('drug_relationship_report', ''))

    # Evaluate
    baseline_eval = run_full_evaluation(final_result, llms)
    print("\nBaseline Evaluation:")
    print(json.dumps(baseline_eval.model_dump(), indent=4))

    # Seed gene pool and evolve
    gene_pool = SOPGenePool()
    gene_pool.add(sop=baseline_sop, eval_result=baseline_eval)

    run_evolution_cycle(team_graph, gene_pool, llms, test_request)

    # Leaderboard
    print("\nSOP Gene Pool — Full Leaderboard:")
    print("-" * 60)
    for entry in gene_pool.pool:
        v  = entry['version']
        p  = entry['parent']
        ev = entry['evaluation']
        a, i, e, g, ac = (
            ev.accuracy.score, ev.interactions.score, ev.evidence.score,
            ev.grounding.score, ev.actionability.score,
        )
        parent_str = "(Baseline)" if p is None else f"(Child of v{p})"
        print(
            f"SOP v{v:<2} {parent_str:<14}: "
            f"Accuracy={a:.2f}  Interactions={i:.2f}  Evidence={e:.2f}  "
            f"Grounding={g:.2f}  Actionability={ac:.2f}"
        )

    # Pareto front
    pareto_sops = identify_pareto_front(gene_pool)
    print("\nPareto-Optimal SOPs:")
    print("-" * 60)
    for entry in pareto_sops:
        v  = entry['version']
        ev = entry['evaluation']
        a, i, e, g, ac = (
            ev.accuracy.score, ev.interactions.score, ev.evidence.score,
            ev.grounding.score, ev.actionability.score,
        )
        print(
            f"SOP v{v}: "
            f"Accuracy={a:.2f}  Interactions={i:.2f}  Evidence={e:.2f}  "
            f"Grounding={g:.2f}  Actionability={ac:.2f}"
        )

    visualize_frontier(pareto_sops)


if __name__ == "__main__":
    main()
