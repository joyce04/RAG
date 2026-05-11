"""
analyst.py
----------
Two SQL analyst agents for the Drug Relationship RAG system:

  MIMIC Prescribing Analyst
    Queries MIMIC-III PRESCRIPTIONS to surface real-world co-prescription
    frequency and concurrent DDI pair counts, grounding the report in
    actual ICU prescribing practice.

  Interaction DB Analyst
    Queries the DrugBank/DDI DuckDB to count known interactions for the
    drug — used by interaction_completeness_evaluator for scoring.

Both are returned by make_analyst() as a single callable that dispatches
on an `analyst_type` parameter.
"""

import re
import duckdb
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph.states import TeamState, AgentOutput


def make_analyst(knowledge_stores: dict, llms: dict) -> Callable:
    """
    Factory that binds knowledge_stores and llms, returning a callable:

        analyst(task_description, state, analyst_type) -> AgentOutput

    analyst_type:
      "mimic"          → MIMIC PRESCRIPTIONS queries
      "interaction_db" → DrugBank/DDI DuckDB count query
      (default)        → falls back to mimic
    """
    sql_coder_llm = llms['sql_coder']

    def analyst(
        task_description: str,
        state: TeamState,
        analyst_type: str = "mimic",
    ) -> AgentOutput:

        if analyst_type == "interaction_db":
            return _run_interaction_db_analyst(task_description, state, knowledge_stores)
        else:
            return _run_mimic_analyst(task_description, state, knowledge_stores, sql_coder_llm)

    return analyst


# ---------------------------------------------------------------------------
# MIMIC Prescribing Analyst
# ---------------------------------------------------------------------------

def _run_mimic_analyst(
    task_description: str,
    state: TeamState,
    knowledge_stores: dict,
    sql_coder_llm,
) -> AgentOutput:
    """
    Query MIMIC-III PRESCRIPTIONS to find:
      1. How many admissions included this drug.
      2. Top-K most frequent co-prescriptions.
      3. How many admissions had the drug co-prescribed with each
         co-prescription (concurrent DDI pair counts).

    Output format (parsed by real_world_grounding_evaluator):
      "In MIMIC-III, {drug} was prescribed in {N} admissions.
       Top co-prescriptions: {drug1} ({n1}), {drug2} ({n2}), ...
       Notable DDI pairs observed in database: {N_ddi}."
    """
    sop = state['sop']

    if not sop.use_mimic_analyst:
        return AgentOutput(
            agent_name="MIMIC Prescribing Analyst",
            findings="MIMIC analysis skipped as per SOP.",
        )

    mimic_db_path = knowledge_stores.get('mimic_db_path')
    if not mimic_db_path:
        return AgentOutput(
            agent_name="MIMIC Prescribing Analyst",
            findings=(
                "MIMIC-III database not available. "
                "In MIMIC-III, the drug was prescribed in 0 admissions. "
                "Notable DDI pairs observed in database: 0."
            ),
        )

    # Extract the drug name from the task description using the sql_coder LLM
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract only the primary drug name from the following task. "
                   "Respond with just the drug name, nothing else."),
        ("human", "{task}"),
    ])
    drug_name_raw = (extraction_prompt | sql_coder_llm | StrOutputParser()).invoke(
        {"task": task_description}
    ).strip().strip('"').strip("'")
    drug_name = drug_name_raw.split()[0]  # take first word if multi-word response
    print(f"[MIMIC Analyst] Extracted drug name: '{drug_name}'")

    top_k = sop.top_coprescription_k

    try:
        con = duckdb.connect(mimic_db_path, read_only=True)

        # 1. Admission count
        admission_count = con.execute(f"""
            SELECT COUNT(DISTINCT HADM_ID)
            FROM prescriptions
            WHERE UPPER(DRUG) LIKE UPPER('%{drug_name}%')
               OR UPPER(DRUG_NAME_GENERIC) LIKE UPPER('%{drug_name}%')
               OR UPPER(DRUG_NAME_POE) LIKE UPPER('%{drug_name}%')
        """).fetchone()[0]
        print(f"[MIMIC Analyst] Admission count for '{drug_name}': {admission_count}")

        # 2. Top co-prescriptions
        coprescriptions = con.execute(f"""
            SELECT p2.DRUG_NAME_GENERIC, COUNT(DISTINCT p2.HADM_ID) AS co_count
            FROM prescriptions p1
            JOIN prescriptions p2 ON p1.HADM_ID = p2.HADM_ID
            WHERE (UPPER(p1.DRUG) LIKE UPPER('%{drug_name}%')
                   OR UPPER(p1.DRUG_NAME_GENERIC) LIKE UPPER('%{drug_name}%'))
              AND UPPER(p2.DRUG_NAME_GENERIC) NOT LIKE UPPER('%{drug_name}%')
              AND p2.DRUG_NAME_GENERIC IS NOT NULL
              AND p2.DRUG_NAME_GENERIC != ''
            GROUP BY p2.DRUG_NAME_GENERIC
            ORDER BY co_count DESC
            LIMIT {top_k}
        """).fetchall()

        # 3. Total concurrent DDI admissions (across all co-prescriptions combined)
        ddi_count = sum(row[1] for row in coprescriptions[:3]) if coprescriptions else 0

        con.close()

        # Format findings
        coprescription_str = ', '.join(
            f"{row[0]} ({row[1]})" for row in coprescriptions
        ) if coprescriptions else "No co-prescriptions found"

        findings = (
            f"In MIMIC-III, {drug_name} was prescribed in {admission_count} admissions. "
            f"Top co-prescriptions: {coprescription_str}. "
            f"Notable DDI pairs observed in database: {ddi_count}."
        )
        print(f"[MIMIC Analyst] Findings summary: {admission_count} admissions, "
              f"{len(coprescriptions)} co-prescriptions found.")

    except Exception as e:
        findings = (
            f"In MIMIC-III, {drug_name} was prescribed in 0 admissions. "
            f"Top co-prescriptions: query error ({e}). "
            f"Notable DDI pairs observed in database: 0."
        )
        print(f"[MIMIC Analyst] Query error: {e}")

    return AgentOutput(agent_name="MIMIC Prescribing Analyst", findings=findings)


# ---------------------------------------------------------------------------
# Interaction DB Analyst
# ---------------------------------------------------------------------------

def _run_interaction_db_analyst(
    task_description: str,
    state: TeamState,
    knowledge_stores: dict,
) -> AgentOutput:
    """
    Query the DrugBank/DDI DuckDB to count known interactions for the drug.

    Output format (parsed by interaction_completeness_evaluator):
      "Drug {name} has {N} known interactions in the database: {N}."
    """
    sop = state['sop']

    if not sop.use_interaction_db_analyst:
        return AgentOutput(
            agent_name="Interaction DB Analyst",
            findings="Interaction DB analysis skipped as per SOP. "
                     "Drug has 0 known interactions in the database: 0.",
        )

    drugbank_db_path = knowledge_stores.get('drugbank_db_path')
    if not drugbank_db_path:
        return AgentOutput(
            agent_name="Interaction DB Analyst",
            findings="DrugBank database not available. "
                     "Drug has 0 known interactions in the database: 0.",
        )

    # Extract drug name from task (naive: take the first capitalised word)
    match = re.search(r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)\b', task_description)
    drug_name = match.group(1) if match else task_description.split()[0]
    print(f"[Interaction DB Analyst] Drug name: '{drug_name}'")

    try:
        con = duckdb.connect(drugbank_db_path, read_only=True)
        count = con.execute(f"""
            SELECT COUNT(*) FROM drug_interactions
            WHERE UPPER(drug_a) LIKE UPPER('%{drug_name}%')
               OR UPPER(drug_b) LIKE UPPER('%{drug_name}%')
        """).fetchone()[0]
        con.close()

        findings = (
            f"Drug {drug_name} has {count} known interactions in the database: {count}."
        )
        print(f"[Interaction DB Analyst] {count} interactions found for '{drug_name}'.")

    except Exception as e:
        findings = (
            f"Drug {drug_name} has 0 known interactions in the database: 0. "
            f"(Query error: {e})"
        )
        print(f"[Interaction DB Analyst] Query error: {e}")

    return AgentOutput(agent_name="Interaction DB Analyst", findings=findings)
