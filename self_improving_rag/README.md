# Self-Improving RAG — Clinical Trial Team

A multi-agent RAG system that autonomously improves its own Standard Operating Procedure (SOP) by diagnosing weak evaluation scores and evolving better configurations through a genetic-style loop.

Applied to clinical trial design: given a drug concept, a team of specialist agents drafts inclusion/exclusion criteria — then the system scores, diagnoses, and mutates its own workflow to improve scientific rigor, regulatory compliance, ethics, feasibility, and operational simplicity simultaneously.

---

## What it does

1. **Retrieval-augmented drafting** — four specialist agents retrieve from domain-specific corpora (PubMed, FDA guidelines, Belmont Report) and query a real patient database (MIMIC-III via DuckDB) to draft evidence-based clinical trial criteria.
2. **Multi-dimensional evaluation** — output is scored on five axes: scientific rigor, regulatory compliance, ethical soundness, recruitment feasibility, and operational simplicity.
3. **Autonomous self-improvement** — a Performance Diagnostician identifies the weakest dimension; a SOP Architect mutates the configuration (prompts, retrieval depth, model choice, agent toggles); new variants are evaluated and added to a Gene Pool.
4. **Pareto-front analysis** — non-dominated SOP configurations are identified and visualised, surfacing the optimal trade-offs in the five-dimensional score space.

---

## Architecture

```
main.py
├── data/
│   ├── download_raw_data.py     # PubMed (Entrez), FDA PDF, ethics text; defines data_paths
│   ├── process_unstructured.py  # FAISS vector stores for PubMed / FDA / ethics corpora
│   └── process_mimic.py         # MIMIC-III CSVs → DuckDB (patients, diagnoses, labs)
│
├── llm.py                       # All Ollama LLM clients (cached singleton)
│
└── graph/
    ├── teamsop.py              # TeamSOP — the evolvable config object
    ├── states.py                # TeamState TypedDict + AgentOutput model
    ├── graph.py                 # build_team_graph(llms, knowledge_stores) factory
    ├── planner.py               # Planner node — decomposes trial concept into tasks
    ├── retriever.py             # Retrieval agent — FAISS lookup per specialist role
    ├── analyst.py               # Patient Cohort Analyst — SQL generation + DuckDB query
    ├── synthesizer.py           # Criteria Synthesizer — final document generation
    ├── evaluator.py             # 5-dimensional evaluation (LLM-as-judge + programmatic)
    ├── diagnostician.py         # Diagnoses primary weakness in evaluation scores
    ├── architect.py             # Generates mutated SOP candidates
    └── sop_pool.py              # Gene pool — stores all SOP versions + evaluations
```

### LangGraph workflow (linear pipeline)

```
planner → execute_specialists → synthesizer → END
```

| Node | Model | Role |
|---|---|---|
| Planner | `llama3.1:8` (JSON mode) | Decomposes trial concept into specialist sub-tasks |
| Retrieval agents | — | FAISS retrieval per specialist (Regulatory / Medical / Ethics) |
| Patient Cohort Analyst | `qwen2:7b` | Generates + executes DuckDB SQL to count eligible patients |
| Synthesizer | `qwen2:7b` (configurable) | Writes formal Inclusion/Exclusion Criteria document |
| Evaluator | `qwen2.5:14b` #`llama3:70b` (LLM-as-judge) | Scores output on 5 dimensions |
| Diagnostician | `llama3:70b` | Identifies the weakest dimension |
| SOP Architect | `llama3:70b` | Proposes 2–3 mutated SOP configurations |

### Knowledge stores

| Store | Source | Backend |
|---|---|---|
| PubMed abstracts | NCBI Entrez (Biopython) | FAISS + `nomic-embed-text` |
| FDA guidelines | PDF download + extraction | FAISS + `nomic-embed-text` |
| Ethics guidelines | Belmont Report summary | FAISS + `nomic-embed-text` |
| Patient database | MIMIC-III CSVs | DuckDB |

### TeamSOP — evolvable parameters

| Field | Default | Effect |
|---|---|---|
| `planner_prompt` | (long) | Controls how the trial concept is broken into sub-tasks |
| `synthesizer_prompt` | (long) | Controls tone and structure of the output document |
| `researcher_retriever_k` | `3` | PubMed retrieval depth |
| `synthesizer_model` | `qwen2:7b` | Ollama model for the final synthesis step |
| `use_sql_analyst` | `True` | Toggle the DuckDB patient count query |
| `use_ethics_specialist` | `True` | Toggle the Ethics Specialist agent |

---

## Evaluation dimensions

| Dimension | Method | Signal |
|---|---|---|
| Scientific Rigor | LLM-as-judge (`llama3:70b`) | Grounded in PubMed literature? |
| Regulatory Compliance | LLM-as-judge (`llama3:70b`) | Follows FDA guidelines? |
| Ethical Soundness | LLM-as-judge (`llama3:70b`) | Respects Belmont Report principles? |
| Recruitment Feasibility | Programmatic | Estimated patient count / 150 (Phase II target) |
| Operational Simplicity | Programmatic | Penalises expensive screening procedures |

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com/) running locally with these models pulled:
  ```bash
  ollama pull llama3.1:8b
  ollama pull qwen2:7b
  ollama pull qwen2.5:14b #llama3:70b
  ollama pull nomic-embed-text
  ```
- MIMIC-III CSV files (gzipped) placed in `data/mimic/`:
  `PATIENTS.csv.gz`, `DIAGNOSES_ICD.csv.gz`, `PROCEDURES_ICD.csv.gz`,
  `PRESCRIPTIONS.csv.gz`, `LABEVENTS.csv.gz`

### Install

```bash
# From the project root (self_improving_rag/)
uv sync
```

### Configure

```bash
cp .env.example .env
# Fill in:
#   ENTREZ_EMAIL     — required for NCBI/PubMed API calls
#   LANGCHAIN_API_KEY — optional, enables LangSmith tracing
#   LANGCHAIN_PROJECT — optional, groups runs in LangSmith
```

### Run

```bash
uv run python main.py
```

This will:
1. Download PubMed articles and the FDA guideline PDF
2. Build the MIMIC-III DuckDB database
3. Embed all corpora into FAISS stores
4. Run the baseline Team graph
5. Evaluate, evolve, and print the Pareto leaderboard
6. Display the Pareto-front visualisation

---

## Project notes (original)

> langsmith > langfuse
> update ethics content
>
> - raw inputs [pubmed articles, FDA guidelines, Ethics summary] -> data processing/indexing -> FAISS vector stores
> - MIMIC-III clinical data -> structured data pipeline -> DuckDB
> Both should be consolidated knowledge stores
