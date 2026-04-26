"""
main.py
-------
Entry point for the self-improving RAG system.

High-level flow:
  1. Download data  — PubMed abstracts, FDA guideline PDF, ethics summary
  2. Build DB       — load MIMIC-III CSVs into DuckDB
  3. Build stores   — embed corpora into FAISS vector stores
  4. Baseline run   — run the Team graph with the seed SOP
  5. Evaluate       — score the output on 5 dimensions
  6. Evolve         — diagnose weakness → mutate SOP → re-evaluate
  7. Analyse        — compute the Pareto front and visualise it
"""

import os
import json
from typing import List, Dict, Any

import duckdb
import numpy as np
from dotenv import load_dotenv

from llm import get_llms
from data.download_raw_data import (
    data_paths,
    download_pubmed_articles,
    download_extract_from_pdf,
    prep_ethics_guidelines,
    prep_paths,
)
from data.process_mimic import load_real_mimic_data
from data.process_unstructured import create_retrievers
from graph.teamsop import TeamSOP
from graph.graph import build_team_graph
from graph.evaluator import run_full_evaluation
from graph.diagnostician import performance_diagnostician
from graph.architect import sop_architect
from graph.sop_pool import SOPGenePool
from display import visualize_frontier

load_dotenv()


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
    Run one full cycle of diagnosis → mutation → evaluation.

    Steps
    -----
    1. Pull the latest SOP from the gene pool as the parent.
    2. Diagnose its primary weakness using the director LLM.
    3. Generate 2–3 mutated SOP candidates via the SOP Architect.
    4. Run the full Team graph for each candidate.
    5. Evaluate each run and add the results to the gene pool.
    """
    print("\n" + "=" * 25 + " STARTING NEW EVOLUTION CYCLE " + "=" * 25)

    current_best = gene_pool.get_latest_entry()
    parent_sop     = current_best['sop']
    parent_eval    = current_best['evaluation']
    parent_version = current_best['version']
    print(f"Improving upon SOP v{parent_version}...")

    # Step 1: identify the weakest dimension
    diagnosis = performance_diagnostician(parent_eval, llms)
    print(
        f"Diagnosis: primary weakness = '{diagnosis.primary_weakness}'. "
        f"Recommendation: {diagnosis.recommendation}"
    )

    # Step 2: generate diverse SOP mutations targeting the weakness
    new_sop_candidates = sop_architect(diagnosis, parent_sop, llms)
    print(f"Generated {len(new_sop_candidates.mutations)} SOP candidate(s).")

    # Step 3: evaluate each candidate by running the full graph
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
    """
    Return the subset of SOPs in the gene pool that are non-dominated.

    A SOP is dominated if another SOP is at least as good on every
    dimension AND strictly better on at least one dimension.
    Non-dominated SOPs form the Pareto front — they represent optimal
    trade-offs across the five evaluation dimensions.
    """
    pareto_front = []
    pool_entries = gene_pool.pool

    for i, candidate in enumerate(pool_entries):
        # Build a numpy array of scores for quick vector comparison
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

            # 'other' dominates 'candidate' if better-or-equal on all dims
            # and strictly better on at least one
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
    print("=== Self-Improving RAG — Clinical Trial Team ===\n")

    # ------------------------------------------------------------------
    # Step 0: initialise
    # ------------------------------------------------------------------
    prep_paths()   # create data directories if missing
    llms = get_llms()
    print("LLM clients configured:")
    for name, client in llms.items():
        if hasattr(client, 'model'):
            print(f"  {name}: {client.model}")

    # ------------------------------------------------------------------
    # Step 1: download / prepare data
    # ------------------------------------------------------------------
    pubmed_query = "(SGLT2 inhibitor) AND (type 2 diabetes) AND (renal impairment)"
    num_downloads = download_pubmed_articles(pubmed_query)
    print(f"Downloaded {num_downloads} PubMed article(s).")

    fda_url      = "https://www.fda.gov/media/71185/download"
    fda_pdf_path = os.path.join(data_paths['fda'], 'fda_guideline.pdf')
    download_extract_from_pdf(fda_url, fda_pdf_path)

    prep_ethics_guidelines()

    # ------------------------------------------------------------------
    # Step 2: build DuckDB from MIMIC-III CSVs
    # ------------------------------------------------------------------
    db_path = load_real_mimic_data()
    if db_path:
        print(f"\nMIMIC-III database: {db_path}")
        con = duckdb.connect(db_path)
        print(f"Tables: {con.execute('SHOW TABLES').df()['name'].tolist()}")
        print("\nSample — patients:")
        print(con.execute("SELECT * FROM patients LIMIT 5").df())
        print("\nSample — diagnoses_icd:")
        print(con.execute("SELECT * FROM diagnoses_icd LIMIT 5").df())
        con.close()
    else:
        print("MIMIC-III data not available — SQL analyst will be skipped.")
        db_path = None

    # ------------------------------------------------------------------
    # Step 3: embed corpora and build FAISS stores + retrievers
    # ------------------------------------------------------------------
    knowledge_stores = create_retrievers(llms['embedding_model'], db_path)
    for name, store in knowledge_stores.items():
        print(f"  {name}: {store}")

    # ------------------------------------------------------------------
    # Step 4: define the baseline SOP and run the first Team graph
    # ------------------------------------------------------------------
    baseline_sop = TeamSOP(
        planner_prompt=(
            "You are a master planner for clinical trial design. Your task is to "
            "receive a high-level trial concept and break it down into a structured "
            "plan with specific sub-tasks for a team of specialists: a Regulatory "
            "Specialist, a Medical Researcher, an Ethics Specialist, and a Patient "
            "Cohort Analyst. Output a JSON object with a single key 'plan' containing "
            "a list of tasks. Each task must have 'agent', 'task_description', and "
            "'dependencies' keys."
        ),
        synthesizer_prompt=(
            "You are an expert medical writer. Your task is to synthesise the "
            "structured findings from all specialist teams into a formal 'Inclusion "
            "and Exclusion Criteria' document. Be concise, precise, and adhere "
            "strictly to the information provided. Structure your output into two "
            "sections: 'Inclusion Criteria' and 'Exclusion Criteria'."
        ),
        researcher_retriever_k=3,
        synthesizer_model="qwen2:7b",
        use_sql_analyst=True,
        use_ethics_specialist=True,
    )
    print("\nBaseline SOP:")
    print(json.dumps(baseline_sop.model_dump(), indent=4))

    # Build the graph (injects llms + knowledge_stores via closures)
    team_graph = build_team_graph(llms, knowledge_stores)

    test_request = (
        "Draft inclusion/exclusion criteria for a Phase II trial of 'Sotagliflozin', "
        "a novel SGLT2 inhibitor, for adults with uncontrolled Type 2 Diabetes "
        "(HbA1c > 8.0%) and moderate chronic kidney disease (CKD Stage 3)."
    )
    print("\nRunning Team graph with baseline SOP...")
    final_result = team_graph.invoke({"initial_request": test_request, "sop": baseline_sop})

    print("\nFinal Criteria (Baseline SOP):")
    print("-" * 40)
    print(final_result['final_criteria'])

    # ------------------------------------------------------------------
    # Step 5: evaluate the baseline run
    # ------------------------------------------------------------------
    baseline_eval = run_full_evaluation(final_result, llms)
    print("\nBaseline Evaluation:")
    print(json.dumps(baseline_eval.model_dump(), indent=4))

    # ------------------------------------------------------------------
    # Step 6: seed the gene pool and run one evolution cycle
    # ------------------------------------------------------------------
    gene_pool = SOPGenePool()
    gene_pool.add(sop=baseline_sop, eval_result=baseline_eval)

    run_evolution_cycle(team_graph, gene_pool, llms, test_request)

    # ------------------------------------------------------------------
    # Step 7: print the full leaderboard
    # ------------------------------------------------------------------
    print("\nSOP Gene Pool — Full Leaderboard:")
    print("-" * 60)
    for entry in gene_pool.pool:
        v   = entry['version']
        p   = entry['parent']
        ev  = entry['evaluation']
        r, c, e, f, s = (
            ev.rigor.score, ev.compliance.score, ev.ethics.score,
            ev.feasibility.score, ev.simplicity.score,
        )
        parent_str = "(Baseline)" if p is None else f"(Child of v{p})"
        print(
            f"SOP v{v:<2} {parent_str:<14}: "
            f"Rigor={r:.2f}  Compliance={c:.2f}  Ethics={e:.2f}  "
            f"Feasibility={f:.2f}  Simplicity={s:.2f}"
        )

    # ------------------------------------------------------------------
    # Step 8: compute Pareto front and visualise
    # ------------------------------------------------------------------
    pareto_sops = identify_pareto_front(gene_pool)
    print("\nPareto-Optimal SOPs:")
    print("-" * 60)
    for entry in pareto_sops:
        v  = entry['version']
        ev = entry['evaluation']
        r, c, e, f, s = (
            ev.rigor.score, ev.compliance.score, ev.ethics.score,
            ev.feasibility.score, ev.simplicity.score,
        )
        print(
            f"SOP v{v}: "
            f"Rigor={r:.2f}  Compliance={c:.2f}  Ethics={e:.2f}  "
            f"Feasibility={f:.2f}  Simplicity={s:.2f}"
        )

    visualize_frontier(pareto_sops)


if __name__ == "__main__":
    main()
