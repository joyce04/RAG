"""
Adaptive RAG Workflow Graph
===========================
Builds a LangGraph state-machine that implements the Adaptive RAG pattern:

  1. ROUTE   → decide: vectorstore retrieval vs. web search
  2. RETRIEVE → fetch documents from ChromaDB
  3. GRADE    → filter irrelevant documents; fallback to web search if needed
  4. GENERATE → synthesize an answer from relevant documents
  5. VALIDATE → check hallucination & answer relevance; retry or redirect

Flow diagram:
  ┌──────────┐
  │  Router  │
  └────┬─────┘
       ├──► [RETRIEVE] ──► [GRADE_DOCUMENTS] ──┬──► [GENERATE] ──┬──► END (useful)
       │                                       │                 ├──► [GENERATE] (hallucinated, retry)
       └──► [WEBSEARCH] ──► [GENERATE] ────────┘                 └──► [WEBSEARCH] (not useful)
"""

import logging

from dotenv import load_dotenv

# ── Step 0: Load environment variables BEFORE importing chains ──────────────
load_dotenv()

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH, GRADE_USEFUL, GRADE_NOT_USEFUL, GRADE_NOT_SUPPORTED
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

logger = logging.getLogger(__name__)

# Router datasource constant (must match the value from RouteQuery schema)
DATASOURCE_VECTORSTORE = "vectorstore"

MAX_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════════════
# Edge Decision Functions
# ═══════════════════════════════════════════════════════════════════════════

def route_question(state: GraphState) -> str:
    """
    Route the user's question to the appropriate retrieval source.

    Decision logic:
      - Ask the LLM-based router to classify the question.
      - If the topic requires up-to-date info → WEBSEARCH
      - If it can be answered from the vectorstore → RETRIEVE

    Args:
        state: Current graph state

    Returns:
        next node to call: WEBSEARCH or RETRIEVE
    """
    question = state["question"]

    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource == WEBSEARCH:
        logger.info("[Route decision]: → web search")
        return WEBSEARCH
    elif source.datasource == DATASOURCE_VECTORSTORE:
        logger.info("[Route decision]: → vectorstore retrieval")
        return RETRIEVE
    else:
        # Defensive fallback — should not happen with a well-constrained router
        logger.warning(
            "Unknown datasource '%s', defaulting to retrieval", source.datasource
        )
        return RETRIEVE


def decide_to_generate(state: GraphState) -> str:
    """
    POST-GRADING — Decide whether to generate an answer or fall back to web search.

    Called after the GRADE_DOCUMENTS node has filtered retrieved documents.
    The grading node sets `state['web_search']` to True if too many
    documents were deemed irrelevant.

    Args:
        state: Current graph state; checks the 'web_search' flag.

    Returns:
        WEBSEARCH if documents were insufficient, GENERATE otherwise.
    """
    if state["web_search"]:
        logger.info("[Graded documents insufficient] → falling back to web search")
        return WEBSEARCH
    else:
        logger.info("[Graded documents sufficient] → proceeding to generate")
        return GENERATE


def grade_generation(state: GraphState) -> str:
    """
    POST-GENERATION — Validate the generated answer for quality.

    Two-stage check:
      1. Hallucination check: Is the answer grounded in the retrieved documents?
         - If NOT grounded → return GRADE_NOT_SUPPORTED (will retry generation)
      2. Answer relevance check: Does the answer actually address the question?
         - If relevant      → return GRADE_USEFUL (→ END, success)
         - If NOT relevant  → return GRADE_NOT_USEFUL (→ WEBSEARCH for more info)

    Args:
        state: Current graph state with 'question', 'documents', and 'generation'.

    Returns:
        One of: GRADE_USEFUL, GRADE_NOT_USEFUL, GRADE_NOT_SUPPORTED.
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)

    if retry_count >= MAX_RETRIES:
        logger.warning("Max retries (%d) reached — accepting current generation", MAX_RETRIES)
        return GRADE_USEFUL

    # ── Stage 1: Hallucination check ────────────────────────────────────
    # Verify the generation is grounded in the source documents.
    hallucination_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_score.binary_score:
        logger.info("✓ Generation is grounded in documents")

        # ── Stage 2: Answer relevance check ─────────────────────────────
        # Verify the generation actually answers the user's question.
        answer_score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )

        if answer_score.binary_score:
            logger.info("✓ Generation addresses the question → useful")
            return GRADE_USEFUL
        else:
            logger.info("✗ Generation does NOT address the question → web search")
            return GRADE_NOT_USEFUL
    else:
        logger.info("✗ Generation is NOT grounded in documents → retry")
        return GRADE_NOT_SUPPORTED


# ═══════════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════════


def build_graph(save_visualization: bool = False) -> StateGraph:
    """
    Construct and compile the Adaptive RAG workflow graph.

    Wrapping construction in a function avoids module-level side effects
    and enables testing with different configurations.

    Args:
        save_visualization: If True, export the graph as 'graph.png'.

    Returns:
        Compiled LangGraph application ready for `.invoke()` or `.stream()`.
    """

    workflow = StateGraph(GraphState)

    # ── Step 1: Register all nodes ──────────────────────────────────────
    # Each node is a function that takes GraphState and returns updates.
    workflow.add_node(RETRIEVE, retrieve)                # Fetch docs from vectorstore
    workflow.add_node(GRADE_DOCUMENTS, grade_documents)  # Filter irrelevant docs
    workflow.add_node(GENERATE, generate)                # Synthesize answer from docs
    workflow.add_node(WEBSEARCH, web_search)              # Fetch info from the web

    # ── Step 2: Set the entry point ─────────────────────────────────────
    # The first node is chosen dynamically by `route_question`.
    workflow.set_conditional_entry_point(
        route_question,
        path_map={
            WEBSEARCH: WEBSEARCH,  # If router says "web" → go to WEBSEARCH node
            RETRIEVE: RETRIEVE,    # If router says "vectorstore" → go to RETRIEVE node
        },
    )

    # ── Step 3: Define edges (transitions between nodes) ────────────────

    # RETRIEVE always flows into GRADE_DOCUMENTS
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

    # After grading, decide: generate from docs or fall back to web search.
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_to_generate,
        path_map={
            WEBSEARCH: WEBSEARCH,  # Docs were poor → web search
            GENERATE: GENERATE,    # Docs are good  → generate answer
        },
    )

    # WEBSEARCH always flows into GENERATE
    workflow.add_edge(WEBSEARCH, GENERATE)

    # After generation, validate the answer quality.
    # This is where self-reflection / corrective RAG happens.
    workflow.add_conditional_edges(
        GENERATE,
        grade_generation,
        path_map={
            GRADE_NOT_SUPPORTED: GENERATE,  # Hallucinated → retry generation
            GRADE_USEFUL: END,              # Good answer  → finish
            GRADE_NOT_USEFUL: WEBSEARCH,    # Off-topic    → try web search
        },
    )
    # The conditional edges above already handle the END transition via the GRADE_USEFUL path

    # ── Step 4: Compile the graph ───────────────────────────────────────
    app = workflow.compile()

    # ── Optional: Export visualization ──────────────────────────────────
    if save_visualization:
        try:
            app.get_graph().draw_mermaid_png(output_file_path="graph.png")
            logger.info("Graph visualization saved to graph.png")
        except Exception as e:
            logger.warning("Could not save graph visualization: %s", e)

    return app


# ── Module-level app instance ──────────────────────────────────────────────
# Created once when the module is first imported (e.g., by main.py).
app = build_graph()
