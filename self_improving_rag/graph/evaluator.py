"""
evaluator.py
------------
Multi-dimensional evaluation of the generated Drug Relationship Report.

Five evaluators run after each Team graph execution:

  1. Pharmacological Accuracy  (LLM-as-judge, director model)
     — Are MOA, targets, and class claims grounded in DrugBank/KEGG context?

  2. Interaction Completeness  (programmatic)
     — DDI pairs cited in report vs. known count from Interaction DB Analyst.

  3. Evidence Depth            (LLM-as-judge, director model)
     — Is each indication backed by a named trial from the PubMed context?

  4. Real-World Grounding      (programmatic)
     — Does the report cite MIMIC-III patient counts? Score = min(1.0, N/500).

  5. Clinical Actionability    (LLM-as-judge, director model)
     — Would a clinician act on this? Rewards specific numbers and sourced claims.
"""

import re

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph.states import TeamState, AgentOutput


def _invoke_structured(llm, prompt, schema, inputs: dict):
    """
    Call `llm` via `prompt`, enforce JSON field names, validate into `schema`.

    ChatOllama does not implement with_structured_output, so we handle
    JSON parsing explicitly, appending a field-name instruction to every prompt.
    """
    props = schema.model_json_schema().get('properties', {})
    fields_str = ", ".join(f'"{k}"' for k in props.keys())
    format_instruction = (
        "system",
        f"You MUST respond with a valid JSON object. "
        f"Required keys: {fields_str}. No other keys, no markdown, no explanation.",
    )
    augmented_prompt = ChatPromptTemplate.from_messages(
        list(prompt.messages) + [format_instruction]
    )
    chain = augmented_prompt | llm | StrOutputParser()
    return schema.model_validate_json(chain.invoke(inputs))


# ---------------------------------------------------------------------------
# Shared data model
# ---------------------------------------------------------------------------

class GradedScore(BaseModel):
    """A normalised score (0.0–1.0) with a human-readable justification."""
    score:     float = Field(description="Score from 0.0 (worst) to 1.0 (best).")
    reasoning: str   = Field(description="Brief justification for the score.")


# ---------------------------------------------------------------------------
# LLM-as-judge evaluators
# ---------------------------------------------------------------------------

def pharmacological_accuracy_evaluator(
    report: str, drugbank_context: str, llms: dict
) -> GradedScore:
    """Score how well MOA, targets, and class claims match the DrugBank/KEGG context."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert clinical pharmacologist. Evaluate a Drug Relationship "
            "Report for pharmacological accuracy. Check that mechanism of action, "
            "molecular targets, and drug class claims are supported by the provided "
            "DrugBank/KEGG context. 1.0 = fully accurate and well-grounded.",
        ),
        (
            "human",
            "**Drug Relationship Report:**\n{report}\n\n"
            "**DrugBank / KEGG Context:**\n{context}",
        ),
    ])
    return _invoke_structured(llms['director'], prompt, GradedScore,
                              {"report": report, "context": drugbank_context})


def evidence_depth_evaluator(
    report: str, pubmed_context: str, llms: dict
) -> GradedScore:
    """Score whether each indication is backed by a named trial from PubMed context."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert clinical researcher. Evaluate a Drug Relationship "
            "Report for evidence depth. Check that each indication or treatment "
            "claim cites a specific trial name, meta-analysis, or guideline "
            "from the provided PubMed context. 1.0 = all claims evidence-backed.",
        ),
        (
            "human",
            "**Drug Relationship Report:**\n{report}\n\n"
            "**PubMed Context:**\n{context}",
        ),
    ])
    return _invoke_structured(llms['director'], prompt, GradedScore,
                              {"report": report, "context": pubmed_context})


def clinical_actionability_evaluator(
    report: str, llms: dict
) -> GradedScore:
    """
    Score whether a clinician or formulary committee could act on this report.
    Rewards specific numbers (MIMIC counts, trial names), penalises generic
    unsourced warnings.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a senior clinical pharmacist reviewing a Drug Relationship "
            "Report for practical utility. Score how actionable this report is "
            "for a clinician or formulary committee. Reward: specific patient "
            "counts from real databases, named trials, severity-ranked interaction "
            "tables, concrete management recommendations. Penalise: vague warnings, "
            "unsourced claims, generic disclaimers. 1.0 = immediately actionable.",
        ),
        (
            "human",
            "**Drug Relationship Report:**\n{report}",
        ),
    ])
    return _invoke_structured(llms['director'], prompt, GradedScore,
                              {"report": report})


# ---------------------------------------------------------------------------
# Programmatic evaluators
# ---------------------------------------------------------------------------

def interaction_completeness_evaluator(
    report: str, interaction_db_output: AgentOutput
) -> GradedScore:
    """
    Score the fraction of known DDIs captured in the report.

    Parses the known DDI count from the Interaction DB Analyst's findings:
      "Drug X has N known interactions in the database: N."

    Then counts DDI pair mentions in the report (lines containing '+' or
    'interact'). Score = min(1.0, mentioned / known).
    """
    IDEAL_KNOWN = 10.0
    findings_text = interaction_db_output.findings if interaction_db_output else ""

    try:
        known_count = int(findings_text.split("in the database: ")[1].replace('.', '').strip())
    except (IndexError, ValueError):
        known_count = 0

    # Count interaction mentions in the report
    mentioned = sum(
        1 for line in report.splitlines()
        if ('+' in line or 'interact' in line.lower() or 'contraindicated' in line.lower())
        and len(line.strip()) > 10
    )

    if known_count == 0:
        score = min(1.0, mentioned / IDEAL_KNOWN)
        reasoning = (
            f"No interaction count available from DB. "
            f"Report mentions ~{mentioned} interaction-related lines. "
            f"Normalised against default target of {int(IDEAL_KNOWN)}."
        )
    else:
        score = min(1.0, mentioned / max(known_count, 1))
        reasoning = (
            f"Report contains ~{mentioned} interaction mentions; "
            f"{known_count} known interactions in database. "
            f"Score = min(1.0, {mentioned}/{known_count})."
        )

    return GradedScore(score=score, reasoning=reasoning)


def real_world_grounding_evaluator(mimic_analyst_output: AgentOutput) -> GradedScore:
    """
    Score how well the report is grounded in real MIMIC-III prescribing data.

    Parses the admission count from the MIMIC Prescribing Analyst's findings:
      "In MIMIC-III, {drug} was prescribed in {N} admissions."

    Score = min(1.0, N / 500). Reflects that 500+ admissions in MIMIC
    provides a strongly grounded real-world example.
    """
    IDEAL_COUNT = 500.0
    findings_text = mimic_analyst_output.findings if mimic_analyst_output else ""

    if not findings_text or "not available" in findings_text or "skipped" in findings_text:
        return GradedScore(
            score=0.0,
            reasoning="MIMIC Prescribing Analyst did not run or data unavailable.",
        )

    try:
        # "In MIMIC-III, {drug} was prescribed in {N} admissions."
        match = re.search(r'prescribed in (\d+) admissions', findings_text)
        admission_count = int(match.group(1)) if match else 0
    except (AttributeError, ValueError):
        return GradedScore(
            score=0.0,
            reasoning="Could not parse admission count from MIMIC analyst output.",
        )

    score = min(1.0, admission_count / IDEAL_COUNT)
    reasoning = (
        f"Drug found in {admission_count} MIMIC-III admissions. "
        f"Score normalised against {int(IDEAL_COUNT)} (threshold for strong grounding)."
    )
    return GradedScore(score=score, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Aggregate result model + orchestration
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """Container for all five evaluation scores from a single Team run."""
    accuracy:      GradedScore
    interactions:  GradedScore
    evidence:      GradedScore
    grounding:     GradedScore
    actionability: GradedScore


def run_full_evaluation(team_final_state: TeamState, llms: dict) -> EvaluationResult:
    """
    Run the full five-dimensional evaluation against the Team graph output.
    `llms` is passed explicitly — no global state.
    """
    print("--- RUNNING FULL EVALUATION GAUNTLET ---")

    report         = team_final_state.get('drug_relationship_report', '')
    agent_outputs  = team_final_state.get('agent_outputs', [])

    # Extract each specialist's findings for evaluation context
    drugbank_context = next(
        (o.findings for o in agent_outputs if o.agent_name == "Pharmacology Specialist"), ""
    )
    kegg_context = next(
        (o.findings for o in agent_outputs if o.agent_name == "Pathway Analyst"), ""
    )
    combined_pharm_context = f"{drugbank_context}\n\n{kegg_context}".strip()

    pubmed_context = next(
        (o.findings for o in agent_outputs if o.agent_name == "Clinical Evidence Specialist"), ""
    )
    mimic_output = next(
        (o for o in agent_outputs if o.agent_name == "MIMIC Prescribing Analyst"), None
    )
    interaction_db_output = next(
        (o for o in agent_outputs if o.agent_name == "Interaction DB Analyst"), None
    )

    print("[Eval] Pharmacological Accuracy...")
    accuracy = pharmacological_accuracy_evaluator(report, combined_pharm_context, llms)

    print("[Eval] Interaction Completeness...")
    interactions = interaction_completeness_evaluator(report, interaction_db_output)

    print("[Eval] Evidence Depth...")
    evidence = evidence_depth_evaluator(report, pubmed_context, llms)

    print("[Eval] Real-World Grounding (MIMIC)...")
    grounding = real_world_grounding_evaluator(mimic_output)

    print("[Eval] Clinical Actionability...")
    actionability = clinical_actionability_evaluator(report, llms)

    print("--- EVALUATION GAUNTLET COMPLETE ---")
    return EvaluationResult(
        accuracy=accuracy,
        interactions=interactions,
        evidence=evidence,
        grounding=grounding,
        actionability=actionability,
    )
