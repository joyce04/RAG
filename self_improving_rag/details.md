# Self-Improving RAG — Drug Relationship System

This document explains the full system end-to-end: what every file does, how the
pieces connect, and why each design decision was made.

---

## Table of Contents

1. [What the System Does (30-second version)](#1-what-the-system-does)
2. [Repository Layout](#2-repository-layout)
3. [How to Run](#3-how-to-run)
4. [Data Layer](#4-data-layer)
5. [LLM Configuration](#5-llm-configuration)
6. [The Team Graph (LangGraph)](#6-the-team-graph-langgraph)
7. [The Self-Improvement Loop](#7-the-self-improvement-loop)
8. [Evaluation System](#8-evaluation-system)
9. [The Streamlit UI](#9-the-streamlit-ui)
10. [Data Flow Diagram](#10-data-flow-diagram)
11. [Key Design Patterns](#11-key-design-patterns)
12. [Known Issues and Mitigations](#12-known-issues-and-mitigations)
13. [Where to Make Common Changes](#13-where-to-make-common-changes)

---

## 1. What the System Does

Given a drug name or treatment concept (e.g. "Warfarin"), the system:

1. Dispatches six specialist AI agents (Pharmacology Specialist, Interaction
   Specialist, Clinical Evidence Specialist, Pathway Analyst, MIMIC Prescribing
   Analyst, Interaction DB Analyst) to retrieve domain-specific evidence.
2. Synthesizes their findings into a formal five-section **Drug Relationship Report**
   (Drug Profile, Related Drugs, Interaction Map, Clinical Evidence, Real-World
   Prescribing Patterns).
3. Scores the output on five dimensions (Accuracy, Interactions, Evidence,
   Grounding, Actionability).
4. Diagnoses the weakest dimension and mutates its own configuration (the "SOP")
   to improve it on the next run.
5. Repeats, accumulating a gene pool of SOP variants and computing the Pareto-optimal
   trade-off frontier across all five dimensions.

The system is fully local — all LLMs run via Ollama, no external LLM API required.

---

## 2. Repository Layout

```
self_improving_rag/
│
├── app.py                      # Streamlit UI entry point
├── main.py                     # CLI entry point (batch mode)
├── display.py                  # Matplotlib Pareto visualisation
├── llm.py                      # LLM client configuration (all Ollama)
│
├── graph/                      # The LangGraph agent system
│   ├── states.py               # TeamState TypedDict + AgentOutput model
│   ├── drugsop.py              # DrugSOP — the evolvable config object (11 fields)
│   ├── teamsop.py              # Shim: re-exports DrugSOP as TeamSOP
│   ├── graph.py                # build_team_graph() — assembles the pipeline
│   ├── planner.py              # Planner node — task decomposition
│   ├── retriever.py            # FAISS retrieval agent (4 specialist roles)
│   ├── analyst.py              # SQL analysts — MIMIC PRESCRIPTIONS + DrugBank DDI
│   ├── synthesizer.py          # Final Drug Relationship Report synthesis
│   ├── evaluator.py            # 5-dimensional scoring (LLM + programmatic)
│   ├── diagnostician.py        # Identifies the weakest evaluation dimension
│   ├── architect.py            # Generates mutated DrugSOP candidates
│   └── sop_pool.py             # In-memory + JSON gene pool store
│
├── data/
│   ├── download_raw_data.py    # Fetches PubMed, DrugBank vocab, DDI corpus, KEGG
│   ├── process_unstructured.py # Builds FAISS stores from text files
│   ├── process_mimic.py        # Loads MIMIC-III PRESCRIPTIONS into DuckDB
│   ├── process_drug_db.py      # Loads DrugBank vocab + DDI corpus into DuckDB
│   ├── pubmed/                 # .txt files, one per PubMed article
│   ├── drugbank/               # drugbank_vocab.txt (downloaded or fallback)
│   ├── ddi/                    # ddi_corpus.txt (downloaded or fallback)
│   ├── kegg/                   # kegg_<drug>.txt files (REST API, cached)
│   ├── mimic/                  # MIMIC-III .csv.gz files (user-provided)
│   └── gene_pool.json          # Persisted gene pool (written by Streamlit UI)
│
├── ui/                         # Streamlit helper modules
│   ├── __init__.py
│   ├── session.py              # session_state init + JSON persistence
│   ├── runner.py               # Thread-pool wrappers for long-running ops
│   ├── components.py           # Reusable st.* widget functions
│   └── charts.py               # Returns matplotlib Figure for st.pyplot()
│
├── pyproject.toml              # uv-managed dependencies
└── .env                        # ENTREZ_EMAIL, LANGCHAIN_API_KEY (not committed)
```

---

## 3. How to Run

### Prerequisites

```bash
# 1. Install Ollama and pull the four required models
ollama pull llama3.1:8b        # planner
ollama pull qwen2:7b           # drafter + sql_coder + synthesizer (default)
ollama pull qwen2.5:14b        # director (evaluator + diagnostician + architect)
ollama pull nomic-embed-text   # embeddings for FAISS

# 2. Install Python dependencies
uv sync

# 3. Copy and fill .env
cp .env.example .env
# Set ENTREZ_EMAIL to any valid email (required by NCBI for PubMed API)
```

MIMIC-III CSVs are optional but strongly recommended — they power the MIMIC
Prescribing Analyst which provides real-world co-prescription grounding. Place
gzipped CSVs in `data/mimic/`:
`PATIENTS.csv.gz`, `PRESCRIPTIONS.csv.gz` (minimum required).
Full set: `DIAGNOSES_ICD.csv.gz`, `PROCEDURES_ICD.csv.gz`, `LABEVENTS.csv.gz`.

The pipeline auto-downloads DrugBank vocab, DDI corpus, KEGG entries, and PubMed
abstracts — no manual data prep needed beyond MIMIC-III.

### CLI (batch, headless)

```bash
uv run python main.py
```

Runs one baseline pass with Warfarin as the test drug + one evolution cycle,
prints the leaderboard, and shows the Pareto chart with `plt.show()`.

### Streamlit UI (interactive)

```bash
uv run streamlit run app.py
```

Opens `http://localhost:8501`. Click **Initialize Pipeline** once per session,
then type drug names or treatment concepts in the Chat tab.

---

## 4. Data Layer

### `data/download_raw_data.py`

**`data_paths` dict** — single source of truth for all directory locations.
Anchored to `Path(__file__).parent` so paths resolve correctly regardless of cwd.

```
data_paths = {
    'base':        .../data/
    'pubmed':      .../data/pubmed/
    'drugbank':    .../data/drugbank/
    'ddi':         .../data/ddi/
    'kegg':        .../data/kegg/
    'mimic':       .../data/mimic/
    'chembl_db':   .../data/chembl.db
    'drugbank_db': .../data/drugs.db
}
```

**`prep_paths()`** — creates all directories if missing. Called first in
`initialize_pipeline()`.

**`download_pubmed_articles(query, max_articles=20)`** — queries NCBI Entrez
for up to 20 articles matching the query, saves each as `data/pubmed/<PMID>.txt`
containing `Title:` and `Abstract:`. Uses `ENTREZ_EMAIL` from `.env`.
Default query: `"(drug interactions) AND (pharmacology) AND (clinical pharmacokinetics)"`.

**`download_drugbank_vocab()`** — writes `data/drugbank/drugbank_vocab.txt` with
drug name / class / description entries in the format
`"Drug: X | Class: Y | Description: ..."`. Falls back to a hardcoded stub via
`_write_fallback_drugbank_vocab()` if any error occurs.

**`download_ddi_corpus()`** — writes `data/ddi/ddi_corpus.txt` with drug-drug
interaction entries in the format
`"DrugA + DrugB / Severity: X / Mechanism: ..."`. Falls back via
`_write_fallback_ddi_corpus()`.

**`download_kegg_drug_entries(drugs=None)`** — calls the free KEGG REST API
(`https://rest.kegg.jp/find/drug/<name>` then
`https://rest.kegg.jp/get/<id>`) for each drug in the default list. Saves as
`data/kegg/kegg_<drug>.txt`. Skip-if-exists caches results; 0.2 s rate-limit
delay between requests. Falls back via `_write_fallback_kegg_entry()`.

---

### `data/process_drug_db.py`

**`load_drug_databases()`** — builds `data/drugs.db` DuckDB with two tables:

- `drug_vocab` — parsed from `drugbank_vocab.txt`: columns `drug_name`, `drug_class`, `description`
- `drug_interactions` — parsed from `ddi_corpus.txt`: columns `drug_a`, `drug_b`, `severity`, `mechanism`

Returns the DB path string, or `None` if neither source file exists.

The Interaction DB Analyst queries `drug_interactions` to count known DDIs for a drug.

---

### `data/process_unstructured.py`

**`create_retrievers(embedding_model, mimic_db_path, drugbank_db_path)`** —
builds four FAISS stores and returns:

```python
{
    'drugbank_retriever': FAISS retriever (k=5),   # DrugBank vocab text
    'ddi_retriever':      FAISS retriever (k=8),   # DDI corpus text
    'pubmed_retriever':   FAISS retriever (k=5),   # PubMed abstracts
    'kegg_retriever':     FAISS retriever (k=3),   # KEGG drug entries
    'mimic_db_path':      str path to DuckDB,      # or None
    'drugbank_db_path':   str path to drugs.db,    # or None
}
```

Each store loads all `.txt` files in its folder, splits into 1000-character
chunks (100-char overlap), and embeds with `nomic-embed-text`. Returns `None` for
any store whose source folder has no files — callers must handle `None`.

---

### `data/process_mimic.py`

Loads MIMIC-III gzipped CSV files into a DuckDB database at
`data/mimic/mimic.duckdb`. Returns the DB path string on success, `None` if
the CSV files are missing.

Key table: `prescriptions` — loaded with columns `SUBJECT_ID`, `HADM_ID`,
`DRUG`, `DRUG_NAME_POE`, `DRUG_NAME_GENERIC`, `DRUG_TYPE`. The `HADM_ID` column
is critical for the co-prescription JOIN queries used by the MIMIC Prescribing
Analyst.

---

## 5. LLM Configuration

### `llm.py`

All models run locally via Ollama through LangChain's `ChatOllama` and
`OllamaEmbeddings`. The function is wrapped with `@lru_cache(maxsize=1)` so
clients are constructed only once per process.

| Key | Model | Temperature | Format | Role |
|-----|-------|------------|--------|------|
| `planner` | `llama3.1:8b` | 0.0 | JSON | Task decomposition |
| `drafter` | `qwen2:7b` | 0.2 | — | Free-form drafting |
| `sql_coder` | `qwen2:7b` | 0.0 | — | DuckDB SQL generation |
| `director` | `qwen2.5:14b` | 0.0 | JSON | Evaluation, diagnosis, mutation |
| `embedding_model` | `nomic-embed-text` | — | — | FAISS vector embeddings |

`get_llms()` returns a plain `dict`. Every downstream function receives this
dict as a parameter — no module-level globals anywhere in `graph/`.

To swap a model: change the string in `llm.py` and restart the process (or
call `get_llms.cache_clear()` in a REPL).

---

## 6. The Team Graph (LangGraph)

### State model — `graph/states.py`

```python
class TeamState(TypedDict):
    initial_request:          str          # user's drug/treatment query (never mutated)
    sop:                      DrugSOP      # active configuration (read-only per run)
    plan:                     dict         # Planner output — list of specialist tasks
    agent_outputs:            List[AgentOutput]  # one per specialist that ran
    drug_relationship_report: str          # synthesised Drug Relationship Report
```

`AgentOutput` is a simple Pydantic model: `agent_name: str`, `findings: Any`.

The state flows through every node. Each node returns a dict that is merged
back into the state by LangGraph (only keys present in the return dict are updated).

---

### Evolvable config — `graph/drugsop.py`

`DrugSOP` is a Pydantic `BaseModel` with eleven fields:

| Field | Type | Default | What it controls |
|-------|------|---------|-----------------|
| `planner_prompt` | str | (long) | How the planner decomposes the drug query |
| `synthesizer_prompt` | str | (long) | Tone and structure of the Drug Relationship Report |
| `pharmacology_retriever_k` | int | 5 | DrugBank/KEGG documents for Pharmacology Specialist |
| `ddi_retriever_k` | int | 8 | DDI corpus documents for Interaction Specialist |
| `evidence_retriever_k` | int | 5 | PubMed documents for Clinical Evidence Specialist |
| `synthesizer_model` | str | `qwen2:7b` | Which Ollama model writes the final report |
| `use_pathway_analyst` | bool | True | Whether the Pathway Analyst (KEGG) runs |
| `use_mimic_analyst` | bool | True | Whether the MIMIC Prescribing Analyst runs |
| `use_interaction_db_analyst` | bool | True | Whether the Interaction DB Analyst (DrugBank DuckDB) runs |
| `min_interaction_severity` | str | `"moderate"` | Filter DDI results by minimum severity |
| `top_coprescription_k` | int | 10 | Top-K co-prescriptions surfaced from MIMIC-III |

`graph/teamsop.py` is a one-line shim: `from graph.drugsop import DrugSOP; TeamSOP = DrugSOP`.
This preserves any external code that still imports `TeamSOP`.

The SOP is passed into every graph run via `state['sop']`. The self-improvement
loop mutates these fields between runs; the graph itself is never recompiled.

---

### Graph assembly — `graph/graph.py`

`build_team_graph(llms, knowledge_stores)` is a factory. It:
1. Calls the node factories (`make_planner_node`, `make_retrieval_agent`,
   `make_analyst`) with injected dependencies.
2. Defines `specialist_execution_node` inline (a closure over the factories).
3. Wires a linear `StateGraph`: `planner → execute_specialists → synthesizer → END`.
4. Returns the compiled graph (a standard LangChain `Runnable`).

The compiled graph is stored in `st.session_state.team_graph` in the UI and
reused across all runs in a session.

**Routing table in `specialist_execution_node`:**

| Agent name prefix | Routes to |
|---|---|
| `"Pharmacology"` | `retrieval_agent(…, "drugbank_retriever", k=sop.pharmacology_retriever_k)` |
| `"Interaction Specialist"` | `retrieval_agent(…, "ddi_retriever", k=sop.ddi_retriever_k)` |
| `"Clinical"` | `retrieval_agent(…, "pubmed_retriever", k=sop.evidence_retriever_k)` |
| `"Pathway"` | `retrieval_agent(…, "kegg_retriever")` — skipped if `sop.use_pathway_analyst=False` |
| `"MIMIC"` | `analyst(task_desc, state, analyst_type="mimic")` — skipped if `sop.use_mimic_analyst=False` |
| `"Interaction DB"` | `analyst(task_desc, state, analyst_type="interaction_db")` — skipped if `sop.use_interaction_db_analyst=False` |

---

### Node 1: Planner — `graph/planner.py`

**Factory:** `make_planner_node(llms)` — closes over `llms['planner']`
(`llama3.1:8b` in JSON mode).

**What it does:** Combines `sop.planner_prompt` with the user's
`initial_request` and sends them to the planner LLM. The LLM returns a JSON
object like:

```json
{
  "plan": [
    {"agent": "Pharmacology Specialist",    "task_description": "...", "dependencies": []},
    {"agent": "Interaction Specialist",     "task_description": "...", "dependencies": []},
    {"agent": "Clinical Evidence Specialist","task_description": "...", "dependencies": []},
    {"agent": "Pathway Analyst",            "task_description": "...", "dependencies": []},
    {"agent": "MIMIC Prescribing Analyst",  "task_description": "...", "dependencies": []},
    {"agent": "Interaction DB Analyst",     "task_description": "...", "dependencies": []}
  ]
}
```

This plan is parsed from JSON and stored in `state['plan']`.

---

### Retrieval Agent — `graph/retriever.py`

**Factory:** `make_retrieval_agent(knowledge_stores)` — closes over the FAISS
retriever dict.

**What it does:**
1. Looks up `knowledge_stores[retriever_name]` — returns a graceful
   `AgentOutput` with `"not available"` message if it is `None`.
2. Overrides `retriever.search_kwargs['k']` with the per-agent k value from the
   SOP (e.g. `sop.pharmacology_retriever_k`, `sop.ddi_retriever_k`). This is how
   the evolution loop tunes retrieval depth.
3. Calls `retriever.invoke(task_description)` — FAISS similarity search.
4. Concatenates results into a single findings string with source metadata.

---

### SQL Analyst — `graph/analyst.py`

**Factory:** `make_analyst(knowledge_stores, llms)` — closes over
`llms['sql_coder']`, `knowledge_stores['mimic_db_path']`, and
`knowledge_stores['drugbank_db_path']`.

The analyst dispatches on `analyst_type`:

#### MIMIC Prescribing Analyst (`analyst_type="mimic"`)

1. Returns early if `sop.use_mimic_analyst=False` or `mimic_db_path` is `None`.
2. Extracts the drug name from the task description using `sql_coder` LLM.
3. Runs three fixed queries against the `prescriptions` table:
   - Admission count: `COUNT(DISTINCT HADM_ID)` where `DRUG ILIKE '%<name>%'`
   - Top-K co-prescriptions: JOIN on `HADM_ID`, grouped by `DRUG_NAME_GENERIC`,
     ordered by frequency, limited to `sop.top_coprescription_k`
   - DDI pair count: count admissions where both the target drug and any of a
     known DDI list appear
4. Formats findings as:
   `"In MIMIC-III, {drug} was prescribed in {N} admissions. Top co-prescriptions: ... Notable DDI pairs observed in database: {N}."`
   The `"prescribed in N admissions"` phrase is parsed by `real_world_grounding_evaluator`.

#### Interaction DB Analyst (`analyst_type="interaction_db"`)

1. Returns early if `sop.use_interaction_db_analyst=False` or `drugbank_db_path` is `None`.
2. Counts rows in `drug_interactions` where `drug_a ILIKE '%<name>%'` or `drug_b ILIKE '%<name>%'`.
3. Formats findings as:
   `"Drug {name} has {N} known interactions in the database: {N}."`
   The `"in the database: N"` phrase is parsed by `interaction_completeness_evaluator`.

**SQL extraction:** strips markdown fences, then uses
`re.search(r'((?:WITH|SELECT)\b.+)', ...)` to discard any prose the LLM
prepended before the actual SQL keyword.

---

### Synthesizer — `graph/synthesizer.py`

A plain LangGraph node function (no factory — reads everything from `state`).

Instantiates `ChatOllama(model=sop.synthesizer_model)` at call time so that SOP
mutations changing the model name take effect immediately. Concatenates all
`agent_outputs` into a labelled context block, then calls the model with
`sop.synthesizer_prompt` prepended. Result goes into
`state['drug_relationship_report']`.

The default synthesizer prompt requests exactly five sections:
1. Drug Profile
2. Related Drugs
3. Interaction Map
4. Clinical Evidence by Indication
5. Real-World Prescribing Patterns (MIMIC-III)

---

## 7. The Self-Improvement Loop

This loop lives in `main.py` (`run_evolution_cycle`) and is mirrored in
`ui/runner.py` (`run_evolution_in_thread`) for the UI. Three components drive it:

### Diagnostician — `graph/diagnostician.py`

**Input:** `EvaluationResult` (five `GradedScore` objects).

**What it does:** Serialises the evaluation to JSON with `model_dump_json()` and
sends it to the `director` LLM with a management-consultant persona. Uses
`_invoke_structured()` (from `evaluator.py`) to parse the response into a
`Diagnosis` Pydantic model:

```python
class Diagnosis(BaseModel):
    primary_weakness: Literal['accuracy', 'interactions', 'evidence', 'grounding', 'actionability']
    root_cause_analysis: str
    recommendation: str
```

The `Literal` type constrains the LLM to one of the five valid dimension names.

---

### Architect — `graph/architect.py`

**Input:** `Diagnosis` + current `DrugSOP`.

**What it does:** Sends the full `DrugSOP` JSON schema, the current SOP's
values, and the Diagnosis to the `director` LLM. Asks for 2–3 new `DrugSOP`
objects as a JSON array under the key `"mutations"`. Parses the response into
an `EvolvedSOPs` model:

```python
class EvolvedSOPs(BaseModel):
    mutations: List[DrugSOP]
```

Mutations can change any of the eleven SOP fields — prompts, k values, model,
agent toggles, severity filter, or top-K co-prescriptions — but only fields
relevant to the diagnosed weakness should change.

---

### Gene Pool — `graph/sop_pool.py`

`SOPGenePool` is an append-only in-memory store:

```python
pool: List[{
    "version":    int,              # auto-incremented from 1
    "sop":        DrugSOP,
    "evaluation": EvaluationResult,
    "parent":     int | None,       # None for baseline (v1)
}]
```

`add(sop, eval_result, parent_version=None)` appends and prints a log line.
`get_latest_entry()` returns the last entry — used as the parent for the next
mutation cycle.

**Persistence** (Streamlit only, via `ui/session.py`): `save_gene_pool()` and
`load_gene_pool()` serialize/deserialize to `data/gene_pool.json` using
`DrugSOP.model_dump()` and `EvaluationResult.model_dump()`. On load,
`version_counter` is restored from `max(entry["version"])`.

---

### Pareto Analysis — `main.py:identify_pareto_front()`

Iterates all pool entries and marks an entry as non-dominated if no other entry
is at least as good on every dimension and strictly better on at least one.
Uses `numpy` vector comparison:

```python
if np.all(other_scores >= cand_scores) and np.any(other_scores > cand_scores):
    is_dominated = True
```

Returns a list of non-dominated entries. Used by `display.py` and the Gene Pool
tab in the UI.

---

## 8. Evaluation System

### `graph/evaluator.py`

All evaluation happens after the Team graph completes. Inputs are
`team_final_state` (the last LangGraph state) and `llms`.

#### `_invoke_structured(llm, prompt, schema, inputs)`

Shared helper used across evaluator, diagnostician, and architect. Because
`ChatOllama` does not implement `.with_structured_output()`, this helper:
1. Reads field names from `schema.model_json_schema()`.
2. Appends a system message: `"You MUST respond with a valid JSON object. Required keys: ..."`.
3. Invokes the chain and calls `schema.model_validate_json()` on the string output.

This is the critical piece that makes structured LLM output reliable with local
Ollama models.

#### `GradedScore`

```python
class GradedScore(BaseModel):
    score:     float   # 0.0 – 1.0
    reasoning: str
```

Returned by every evaluator. Displayed in the UI as metric cards.

#### Five evaluators

| Function | Method | Source context | Parse target |
|----------|--------|----------------|-------------|
| `pharmacological_accuracy_evaluator` | LLM-as-judge (`director`) | DrugBank + KEGG retriever findings | — |
| `interaction_completeness_evaluator` | Programmatic | Interaction DB Analyst output | `"in the database: N"` → score = min(1.0, N/50) |
| `evidence_depth_evaluator` | LLM-as-judge (`director`) | PubMed retriever findings | — |
| `real_world_grounding_evaluator` | Programmatic | MIMIC Prescribing Analyst output | `"prescribed in N admissions"` → score = min(1.0, N/500) |
| `clinical_actionability_evaluator` | LLM-as-judge (`director`) | Full Drug Relationship Report | — |

`run_full_evaluation(team_final_state, llms)` calls all five sequentially and
returns an `EvaluationResult` containing all five `GradedScore` objects
(fields: `accuracy`, `interactions`, `evidence`, `grounding`, `actionability`).

#### Important: parsing contracts

`interaction_completeness_evaluator` parses with:
```python
n = int(findings.split("in the database: ")[1].split(".")[0].strip())
```

`real_world_grounding_evaluator` parses with:
```python
n = int(findings.split("prescribed in ")[1].split(" admissions")[0].strip())
```

If you modify the output format of `analyst.py`, update these parsers too.

---

## 9. The Streamlit UI

### Entry point — `app.py`

Sets `sys.path` to include the project root (necessary because Streamlit's
working directory may not be the project root). Loads `.env`, calls
`init_session_state()`, then renders the sidebar and three tabs.

#### Sidebar

- **Initialize Pipeline** button — calls `initialize_pipeline()` from `main.py`,
  stores `llms`, `knowledge_stores`, and `team_graph` in `st.session_state`.
  Auto-loads `data/gene_pool.json` if it exists. Sets `active_sop` to a
  baseline `DrugSOP`.
- **SOP controls** — three k sliders (`pharmacology_retriever_k`,
  `ddi_retriever_k`, `evidence_retriever_k`, `top_coprescription_k`), a
  `min_interaction_severity` selectbox, and three agent toggles
  (`use_mimic_analyst`, `use_pathway_analyst`, `use_interaction_db_analyst`).
  Returns a new `DrugSOP` on every widget change via `render_sop_controls()`.
- **Save / Load Pool** — explicit persistence buttons.

#### Tab 1 — Chat

Replays `st.session_state.chat_history` on every rerun. On new input:
1. Appends the user message immediately and re-renders it.
2. Calls `run_pipeline_in_thread()` which submits the work to a
   `ThreadPoolExecutor` and puts progress messages on a `queue.Queue`.
3. Inside `st.status()`, polls the queue in a `while True` loop until a
   `"RESULT"` or `"ERROR"` message arrives (timeout: 300 s).
4. Renders Drug Relationship Report text, five metric columns, and per-agent expanders.
5. Appends the full assistant message (including `eval_result` and
   `agent_outputs`) to `chat_history` for replay on future reruns.

#### Tab 2 — Gene Pool

Calls `identify_pareto_front()`, builds a `pd.DataFrame`, highlights Pareto rows
in green using pandas `.style.apply()`, and renders the Pareto frontier chart
via `ui/charts.py:pareto_figure()` → `st.pyplot(fig)`.

**Run Evolution Cycle** button works the same way as the Chat tab — submits to
a thread pool, polls a queue inside `st.status()`, calls `st.rerun()` when done
so the leaderboard refreshes. Default trial drug: Warfarin (the same complex
multi-interaction drug used in `main.py`).

#### Tab 3 — SOP Inspector

`st.json(st.session_state.active_sop.model_dump())` — fully interactive JSON
viewer showing all eleven DrugSOP fields.

---

### Threading model — `ui/runner.py`

Long-running calls (`team_graph.stream()`, `run_full_evaluation()`,
`performance_diagnostician()`, `sop_architect()`) block for 60–300 seconds.
Streamlit's main thread cannot block, so these run in a `ThreadPoolExecutor`
(max 2 workers, module-level singleton).

Communication uses `queue.Queue`:
- `("PROGRESS", message_str)` — displayed inside `st.status()`
- `("RESULT", payload)` — final output
- `("ERROR", traceback_str)` — displayed as `st.error()`

The graph is called with `.stream()` instead of `.invoke()`, which yields one
dict per completed node. This gives free per-node progress messages:

```python
for chunk in team_graph.stream({"initial_request": request, "sop": sop}):
    node_name = list(chunk.keys())[0]
    final_state = list(chunk.values())[0]
    status_q.put(("PROGRESS", f"Completed: {node_name}"))
```

---

### Session persistence — `ui/session.py`

`GENE_POOL_PATH = .../data/gene_pool.json`

`save_gene_pool(pool)` — called inside `run_pipeline_in_thread` and
`run_evolution_in_thread` after every `pool.add()`. Also callable manually
via the sidebar button.

`load_gene_pool()` — reconstructs `SOPGenePool` from JSON by instantiating
`DrugSOP(**d)` and manually building `EvaluationResult` from nested dicts
using field names `accuracy`, `interactions`, `evidence`, `grounding`,
`actionability`. Restores `version_counter` from `max(entry["version"])`.

---

## 10. Data Flow Diagram

```
User types drug name or treatment concept
        │
        ▼
  [Planner — llama3.1:8b]
  Reads: sop.planner_prompt + initial_request
  Writes: state['plan'] — list of specialist tasks
        │
        ▼
  [Specialist Dispatcher — inline in graph.py]
  Routes each task by agent name prefix:
  ┌───────────────────────┐  ┌──────────────────────────┐
  │ Pharmacology Specialist│  │ Interaction Specialist    │
  │ → drugbank_retriever  │  │ → ddi_retriever          │
  │ k=sop.pharm_k         │  │ k=sop.ddi_k              │
  └───────────────────────┘  └──────────────────────────┘
  ┌───────────────────────┐  ┌──────────────────────────┐
  │ Clinical Evidence     │  │ Pathway Analyst           │
  │ → pubmed_retriever    │  │ → kegg_retriever          │
  │ k=sop.evidence_k      │  │ (if use_pathway_analyst)  │
  └───────────────────────┘  └──────────────────────────┘
  ┌───────────────────────┐  ┌──────────────────────────┐
  │ MIMIC Prescribing     │  │ Interaction DB Analyst    │
  │ → DuckDB PRESCRIPTIONS│  │ → DuckDB drug_interactions│
  │ (if use_mimic_analyst)│  │ (if use_interaction_db)   │
  └───────────────────────┘  └──────────────────────────┘
  All outputs → state['agent_outputs']
        │
        ▼
  [Synthesizer — qwen2:7b (configurable)]
  Reads: sop.synthesizer_prompt + all agent_outputs
  Writes: state['drug_relationship_report']
  (5 sections: Profile / Related Drugs / Interactions / Evidence / MIMIC)
        │
        ▼
  [Evaluator — qwen2.5:14b + programmatic]
  5 scores → EvaluationResult
  (accuracy / interactions / evidence / grounding / actionability)
        │
        ▼
  [Gene Pool] — add(sop, eval_result)
        │
   (evolution cycle only)
        │
        ▼
  [Diagnostician — qwen2.5:14b]
  EvaluationResult → Diagnosis (weakest dimension)
        │
        ▼
  [Architect — qwen2.5:14b]
  Diagnosis + DrugSOP → 2–3 mutated DrugSOPs
        │
        ▼
  Repeat from top for each mutation
```

---

## 11. Key Design Patterns

### Factory / closure injection

Every agent that needs `llms` or `knowledge_stores` is built through a factory
function:

```python
# graph/planner.py
def make_planner_node(llms: dict) -> Callable:
    planner_llm = llms['planner']          # captured once
    def planner_agent(state): ...          # uses the captured client
    return planner_agent
```

This means:
- No module-level globals in `graph/`.
- Easy to test — inject mock dicts.
- Graph can be rebuilt with different LLMs without touching node code.

### `_invoke_structured` — structured output from local LLMs

`ChatOllama` does not support `.with_structured_output()`. The shared helper
`_invoke_structured(llm, prompt, schema, inputs)` in `evaluator.py` solves this
by appending a strict field-name instruction to every prompt and calling
`schema.model_validate_json()` on the output. All LLM-driven structured
outputs (GradedScore, Diagnosis, EvolvedSOPs) rely on this pattern.

### DrugSOP as the only mutation surface

The graph code, retriever code, and evaluator code are all immutable. The only
thing that changes between runs is the `DrugSOP` object passed in the state.
This makes the evolution loop safe — mutations cannot break the graph topology.

### Relative imports throughout `graph/`

All imports within `graph/` use the full package path (`from graph.states import ...`,
not `from states import ...`) so the package works regardless of the working
directory.

---

## 12. Known Issues and Mitigations

| Issue | Where | Mitigation |
|-------|--------|------------|
| LLM prepends prose before SQL | `graph/analyst.py` | `re.search(r'((?:WITH\|SELECT)\b.+)')` extracts from first SQL keyword |
| DrugBank/DDI download fails | `data/download_raw_data.py` | Fallback stub writers (`_write_fallback_*`); skip-if-exists prevents repeated failures |
| `mimic_db_path` is `None` when no MIMIC data | `graph/analyst.py` | Early return with MIMIC-format output containing 0 admissions |
| `drugbank_db_path` is `None` | `graph/analyst.py` | Early return with "0 known interactions" message |
| KEGG REST API rate limits | `data/download_raw_data.py` | 0.2 s sleep between requests; skip-if-exists caching |
| Any retriever is `None` when source folder empty | `graph/retriever.py` | Returns `AgentOutput` with `"not available"` message instead of crashing |
| `ChatOllama` no `with_structured_output` | `graph/evaluator.py` | `_invoke_structured()` helper with explicit field-name enforcement |
| Streamlit blocks on long LLM calls | `ui/runner.py` | `ThreadPoolExecutor` + `queue.Queue` polling pattern |
| Pareto chart crashes with 1 entry | `ui/charts.py` | Returns `None` if `len(pareto_sops) < 2` |
| `synthesizer_model` must be pre-pulled in Ollama | `graph/synthesizer.py` | `ChatOllama` instantiated at call time — fails gracefully if model missing |

---

## 13. Where to Make Common Changes

| Goal | File(s) to edit |
|------|----------------|
| Change which LLM model is used | `llm.py` — update the model string in the relevant key |
| Change the default planner/synthesizer prompt | `main.py` (`_BASELINE_PLANNER_PROMPT` / `_BASELINE_SYNTHESIZER_PROMPT` constants) |
| Add a new specialist agent | `graph/graph.py` (add routing branch), `graph/retriever.py` or new file, `data/process_unstructured.py` (add FAISS store), `graph/drugsop.py` (add toggle field) |
| Add a 6th evaluation dimension | `graph/evaluator.py` (new function + add to `EvaluationResult`), `graph/diagnostician.py` (update `Literal`), `ui/components.py` (`render_evaluation_scores`), `main.py` (leaderboard print) |
| Add a new evolvable SOP field | `graph/drugsop.py` (add field), then propagate reads in whichever node uses it |
| Change the PubMed query topic | `main.py:initialize_pipeline()` — `pubmed_query` variable |
| Change KEGG drug list | `data/download_raw_data.py` — default `drugs` list in `download_kegg_drug_entries()` |
| Change retrieval chunk size | `data/process_unstructured.py` — `RecursiveCharacterTextSplitter` params |
| Add more DDI corpus entries | `data/ddi/ddi_corpus.txt` — one entry per line in `DrugA + DrugB / Severity: X / Mechanism: Y` format |
| Persist data across CLI runs | `main.py` — call `save_gene_pool()` / `load_gene_pool()` from `ui/session.py` at start and end of `main()` |
| Swap from Ollama to an API model | `llm.py` — replace `ChatOllama` with `ChatOpenAI` / `ChatAnthropic`; no other files need changing |
