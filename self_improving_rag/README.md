# Self-Improving RAG — Drug Relationship System

![RAG Drug Relationship System](img/demo.gif)


A multi-agent RAG system that autonomously improves its own Standard Operating Procedure (SOP) by diagnosing weak evaluation scores and evolving better configurations through a genetic-style loop.

Applied to drug and treatment analysis: given a drug name or treatment concept, a team of six specialist agents retrieves pharmacology data, maps drug-drug interactions, surfaces clinical evidence, and grounds findings in real-world MIMIC-III prescribing patterns — then the system scores, diagnoses, and mutates its own workflow to produce better reports over time.

---

## What it does

1. **Retrieval-augmented drug analysis** — six specialist agents retrieve from domain-specific corpora (DrugBank, DDI corpus, PubMed, KEGG pathways) and query real patient data (MIMIC-III PRESCRIPTIONS via DuckDB) to build an evidence-based Drug Relationship Report.
2. **Multi-dimensional evaluation** — output is scored on five axes: pharmacological accuracy, interaction completeness, evidence depth, real-world grounding (MIMIC), and clinical actionability.
3. **Autonomous self-improvement** — a Performance Diagnostician identifies the weakest dimension; a SOP Architect mutates the configuration (prompts, retrieval depth, model choice, agent toggles, severity filters); new variants are evaluated and added to a Gene Pool.
4. **Pareto-front analysis** — non-dominated SOP configurations are identified and visualised, surfacing the optimal trade-offs in the five-dimensional score space.

---

## Architecture

```
main.py
├── data/
│   ├── download_raw_data.py     # PubMed, DrugBank vocab, DDI corpus, KEGG (REST); defines data_paths
│   ├── process_unstructured.py  # FAISS vector stores for DrugBank / DDI / PubMed / KEGG corpora
│   ├── process_mimic.py         # MIMIC-III PRESCRIPTIONS CSV → DuckDB
│   └── process_drug_db.py       # DrugBank vocab + DDI corpus → DuckDB (drug_vocab, drug_interactions)
│
├── llm.py                       # All Ollama LLM clients (cached singleton)
│
└── graph/
    ├── drugsop.py               # DrugSOP — the evolvable config object (11 fields)
    ├── teamsop.py               # Shim: re-exports DrugSOP as TeamSOP
    ├── states.py                # TeamState TypedDict + AgentOutput model
    ├── graph.py                 # build_team_graph(llms, knowledge_stores) factory
    ├── planner.py               # Planner node — decomposes drug query into specialist tasks
    ├── retriever.py             # Retrieval agent — FAISS lookup per specialist role
    ├── analyst.py               # SQL analysts — MIMIC PRESCRIPTIONS + DrugBank DDI queries
    ├── synthesizer.py           # Drug Relationship Report synthesis
    ├── evaluator.py             # 5-dimensional evaluation (LLM-as-judge + programmatic)
    ├── diagnostician.py         # Diagnoses primary weakness in evaluation scores
    ├── architect.py             # Generates mutated DrugSOP candidates
    └── sop_pool.py              # Gene pool — stores all SOP versions + evaluations
```

### LangGraph workflow (linear pipeline)

```
planner → execute_specialists → synthesizer → END
```

| Node | Model | Role |
|---|---|---|
| Planner | `llama3.1:8b` (JSON mode) | Decomposes drug query into six specialist sub-tasks |
| Pharmacology Specialist | — | FAISS retrieval over DrugBank vocab + KEGG pathways |
| Interaction Specialist | — | FAISS retrieval over DDI corpus |
| Clinical Evidence Specialist | — | FAISS retrieval over PubMed abstracts |
| Pathway Analyst | — | FAISS retrieval over KEGG drug entries (togglable) |
| MIMIC Prescribing Analyst | `qwen2:7b` | Queries MIMIC-III PRESCRIPTIONS for co-prescription patterns |
| Interaction DB Analyst | `qwen2:7b` | Counts known DDIs from DrugBank DuckDB |
| Synthesizer | `qwen2:7b` (configurable) | Writes 5-section Drug Relationship Report |
| Evaluator | `qwen2.5:14b` + programmatic | Scores output on 5 dimensions |
| Diagnostician | `qwen2.5:14b` | Identifies the weakest dimension |
| SOP Architect | `qwen2.5:14b` | Proposes 2–3 mutated DrugSOP configurations |

### Knowledge stores

| Store | Source | Backend |
|---|---|---|
| DrugBank vocab | Hardcoded / downloadable vocab file | FAISS + `nomic-embed-text` |
| DDI corpus | Hardcoded / downloadable DDI text | FAISS + `nomic-embed-text` |
| PubMed abstracts | NCBI Entrez (Biopython) | FAISS + `nomic-embed-text` |
| KEGG pathways | KEGG REST API (free, cached) | FAISS + `nomic-embed-text` |
| MIMIC-III prescriptions | MIMIC-III `PRESCRIPTIONS.csv.gz` | DuckDB |
| DrugBank interactions | DDI corpus parsed to tabular | DuckDB |

### DrugSOP — evolvable parameters

| Field | Default | Effect |
|---|---|---|
| `planner_prompt` | (long) | Controls how the drug query is broken into specialist sub-tasks |
| `synthesizer_prompt` | (long) | Controls tone and structure of the Drug Relationship Report |
| `pharmacology_retriever_k` | `5` | DrugBank/KEGG documents retrieved by Pharmacology Specialist |
| `ddi_retriever_k` | `8` | DDI corpus documents retrieved by Interaction Specialist |
| `evidence_retriever_k` | `5` | PubMed documents retrieved by Clinical Evidence Specialist |
| `synthesizer_model` | `qwen2:7b` | Ollama model for the final synthesis step |
| `use_pathway_analyst` | `True` | Toggle the Pathway Analyst (KEGG retrieval) |
| `use_mimic_analyst` | `True` | Toggle the MIMIC Prescribing Analyst |
| `use_interaction_db_analyst` | `True` | Toggle the Interaction DB Analyst (DrugBank DuckDB) |
| `min_interaction_severity` | `"moderate"` | Minimum DDI severity to include in the Interaction Map |
| `top_coprescription_k` | `10` | Top-K co-prescriptions surfaced from MIMIC-III |

---

## Evaluation dimensions

| Dimension | Method | Signal |
|---|---|---|
| Pharmacological Accuracy | LLM-as-judge (`qwen2.5:14b`) | Grounded in DrugBank / KEGG literature? |
| Interaction Completeness | Programmatic | Known DDI count from DrugBank DB / 50 |
| Evidence Depth | LLM-as-judge (`qwen2.5:14b`) | Grounded in PubMed clinical literature? |
| Real-World Grounding | Programmatic | MIMIC-III admission count / 500 |
| Clinical Actionability | LLM-as-judge (`qwen2.5:14b`) | Does the report support clinical decisions? |

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com/) running locally with these models pulled:
  ```bash
  ollama pull llama3.1:8b
  ollama pull qwen2:7b
  ollama pull qwen2.5:14b
  ollama pull nomic-embed-text
  ```
- MIMIC-III CSV files (gzipped) placed in `data/mimic/` (optional but recommended):
  `PRESCRIPTIONS.csv.gz` (minimum), plus `PATIENTS.csv.gz`, `DIAGNOSES_ICD.csv.gz`,
  `PROCEDURES_ICD.csv.gz`, `LABEVENTS.csv.gz`

### Install

```bash
# From the project root (self_improving_rag/)
uv sync
```

### Configure

```bash
cp .env.example .env
# Fill in:
#   ENTREZ_EMAIL      — required for NCBI/PubMed API calls
#   LANGCHAIN_API_KEY — optional, enables LangSmith tracing
#   LANGCHAIN_PROJECT — optional, groups runs in LangSmith
```

### Run — CLI (batch mode)

```bash
uv run python main.py
```

This will:
1. Download PubMed articles, DrugBank vocab, DDI corpus, and KEGG drug entries
2. Build the MIMIC-III DuckDB database and DrugBank DuckDB
3. Embed all corpora into FAISS stores
4. Run the baseline Team graph with Warfarin as the test drug
5. Evaluate the report on 5 dimensions
6. Run one evolution cycle (diagnose → mutate → re-evaluate)
7. Print the leaderboard and display the Pareto-front visualisation

### Run — Streamlit chat UI

```bash
uv run streamlit run app.py
```

Opens a browser tab at `http://localhost:8501` with:

| Tab | What you can do |
|-----|----------------|
| **Chat** | Type any drug name or treatment concept → watch per-node progress → see the Drug Relationship Report + 5-dimensional scores + specialist findings |
| **Gene Pool** | View the leaderboard of all SOP versions (Pareto rows highlighted green), inspect the Pareto frontier chart, and trigger evolution cycles |
| **SOP Inspector** | Browse the full JSON of the currently active DrugSOP |

**Workflow:**
1. Click **Initialize Pipeline** in the sidebar (downloads data, builds FAISS stores, connects to Ollama — one-time per session)
2. Optionally adjust retrieval k sliders, severity filter, and agent toggles in the sidebar
3. Type a drug name in the Chat tab and press Enter
4. Switch to **Gene Pool** → click **Run Evolution Cycle** to generate improved SOP variants
5. Use **Save Pool / Load Pool** to persist results across browser refreshes

**Sample inputs for demo:**

```
Warfarin
Furosemide
Metformin
Aspirin
Amiodarone
```

Or try a detailed query:

```
Analyse Warfarin: map its pharmacological profile, rank its top drug-drug
interactions by severity, summarise clinical evidence for its use in atrial
fibrillation and VTE, and show how frequently it is co-prescribed with
Amiodarone, Aspirin, and Digoxin in real ICU patients from MIMIC-III.
```
