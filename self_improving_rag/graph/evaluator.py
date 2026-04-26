"""
evaluator.py
------------
Multi-dimensional evaluation of generated trial criteria.

Five evaluators are run after each Team graph execution:

  1. Scientific Rigor     (LLM-as-judge, director model)
     — Is the criteria grounded in the retrieved PubMed literature?

  2. Regulatory Compliance (LLM-as-judge, director model)
     — Does it comply with the retrieved FDA guidelines?

  3. Ethical Soundness     (LLM-as-judge, director model)
     — Does it respect the Belmont Report principles?

  4. Recruitment Feasibility (programmatic)
     — How many real MIMIC-III patients would qualify? Normalised to 150.

  5. Operational Simplicity  (programmatic)
     — Does it avoid expensive screening procedures?

Each evaluator returns a GradedScore(score: float, reasoning: str).
`run_full_evaluation` runs all five and returns an EvaluationResult.
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from graph.states import TeamState, AgentOutput


# ---------------------------------------------------------------------------
# Shared data model
# ---------------------------------------------------------------------------

class GradedScore(BaseModel):
    """A normalised score (0.0–1.0) with a human-readable justification."""
    score:     float = Field(description="Score from 0.0 (worst) to 1.0 (best).")
    reasoning: str   = Field(description="Brief justification for the score.")


# ---------------------------------------------------------------------------
# LLM-as-judge evaluators (use the director — the most capable local model)
# ---------------------------------------------------------------------------

def scientific_rigor_evaluator(
    generated_criteria: str, pubmed_context: str, llms: dict
) -> GradedScore:
    """
    Score how well the criteria align with the retrieved scientific literature.
    1.0 = perfectly grounded; 0.0 = contradicts or ignores the literature.
    """
    evaluator_llm = llms['director'].with_structured_output(GradedScore)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert clinical scientist. Evaluate a set of clinical trial "
            "criteria based on the provided scientific literature. "
            "1.0 = perfectly aligned with the literature. 0.0 = contradicts it.",
        ),
        (
            "human",
            "**Generated Criteria:**\n{criteria}\n\n"
            "**Supporting Scientific Context:**\n{context}",
        ),
    ])
    return (prompt | evaluator_llm).invoke({"criteria": generated_criteria, "context": pubmed_context})


def regulatory_compliance_evaluator(
    generated_criteria: str, fda_context: str, llms: dict
) -> GradedScore:
    """Score adherence to the retrieved FDA guidelines. 1.0 = full compliance."""
    evaluator_llm = llms['director'].with_structured_output(GradedScore)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert regulatory affairs specialist. Evaluate if a set of "
            "clinical trial criteria adheres to the provided FDA guidelines. "
            "1.0 = full compliance.",
        ),
        (
            "human",
            "**Generated Criteria:**\n{criteria}\n\n"
            "**Applicable FDA Guidelines:**\n{context}",
        ),
    ])
    return (prompt | evaluator_llm).invoke({"criteria": generated_criteria, "context": fda_context})


def ethical_soundness_evaluator(
    generated_criteria: str, ethics_context: str, llms: dict
) -> GradedScore:
    """
    Score ethical quality based on Belmont Report principles.
    1.0 = strong respect for autonomy, beneficence, and justice.
    """
    evaluator_llm = llms['director'].with_structured_output(GradedScore)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert on clinical trial ethics. Evaluate if a set of criteria "
            "adheres to the Belmont Report principles (respect for persons, beneficence, "
            "justice). 1.0 = exemplary ethical practice.",
        ),
        (
            "human",
            "**Generated Criteria:**\n{criteria}\n\n"
            "**Ethical Principles:**\n{context}",
        ),
    ])
    return (prompt | evaluator_llm).invoke({"criteria": generated_criteria, "context": ethics_context})


# ---------------------------------------------------------------------------
# Programmatic evaluators (no LLM needed)
# ---------------------------------------------------------------------------

def feasibility_evaluator(cohort_analyst_output: AgentOutput) -> GradedScore:
    """
    Parse the patient count from the Cohort Analyst's findings and
    normalise it against the ideal Phase II target of 150 patients.

    score = min(1.0, patient_count / 150)
    """
    IDEAL_COUNT = 150.0
    findings_text = cohort_analyst_output.findings

    try:
        # The analyst always ends its findings with "database: <N>."
        count_str = findings_text.split("database: ")[1].replace('.', '').strip()
        patient_count = int(count_str)
    except (IndexError, ValueError):
        return GradedScore(
            score=0.0,
            reasoning="Could not parse patient count from analyst output.",
        )

    score = min(1.0, patient_count / IDEAL_COUNT)
    reasoning = (
        f"Estimated {patient_count} eligible patients. "
        f"Score normalised against ideal target of {int(IDEAL_COUNT)}."
    )
    return GradedScore(score=score, reasoning=reasoning)


def simplicity_evaluator(generated_criteria: str) -> GradedScore:
    """
    Penalise criteria that require expensive or complex screening procedures.
    Each expensive test found reduces the score by 0.5 (floor at 0.0).
    """
    EXPENSIVE_TESTS = [
        "mri", "genetic sequencing", "pet scan",
        "biopsy", "echocardiogram", "endoscopy",
    ]
    test_count = sum(1 for t in EXPENSIVE_TESTS if t in generated_criteria.lower())
    score = max(0.0, 1.0 - test_count * 0.5)
    reasoning = f"Found {test_count} expensive/complex screening procedure(s) in the criteria."
    return GradedScore(score=score, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Aggregate result model + orchestration function
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """Container for all five evaluation scores from a single Team run."""
    rigor:       GradedScore
    compliance:  GradedScore
    ethics:      GradedScore
    feasibility: GradedScore
    simplicity:  GradedScore


def run_full_evaluation(team_final_state: TeamState, llms: dict) -> EvaluationResult:
    """
    Run the full five-dimensional evaluation gauntlet against the
    most recent Team graph output.

    `llms` is passed explicitly so this function has no hidden globals.
    """
    print("--- RUNNING FULL EVALUATION GAUNTLET ---")

    final_criteria = team_final_state['final_criteria']
    agent_outputs  = team_final_state['agent_outputs']

    # Extract each specialist's findings to use as evaluation context
    pubmed_context  = next((o.findings for o in agent_outputs if o.agent_name == "Medical Researcher"),  "")
    fda_context     = next((o.findings for o in agent_outputs if o.agent_name == "Regulatory Specialist"), "")
    ethics_context  = next((o.findings for o in agent_outputs if o.agent_name == "Ethics Specialist"),    "")
    analyst_output  = next((o for o in agent_outputs          if o.agent_name == "Patient Cohort Analyst"), None)

    print("[Eval] Scientific Rigor...")
    rigor = scientific_rigor_evaluator(final_criteria, pubmed_context, llms)

    print("[Eval] Regulatory Compliance...")
    compliance = regulatory_compliance_evaluator(final_criteria, fda_context, llms)

    print("[Eval] Ethical Soundness...")
    ethics = ethical_soundness_evaluator(final_criteria, ethics_context, llms)

    print("[Eval] Recruitment Feasibility...")
    feasibility = (
        feasibility_evaluator(analyst_output)
        if analyst_output
        else GradedScore(score=0.0, reasoning="Analyst did not run.")
    )

    print("[Eval] Operational Simplicity...")
    simplicity = simplicity_evaluator(final_criteria)

    print("--- EVALUATION GAUNTLET COMPLETE ---")
    return EvaluationResult(
        rigor=rigor,
        compliance=compliance,
        ethics=ethics,
        feasibility=feasibility,
        simplicity=simplicity,
    )
