Self-reflection for RAG is the system that reviews its own answers and decides whether to retrieve more relevant data or not.[1]
Corrective RAG is a RAG system that corrects the retrieved documents before answering.[2]
Adaptive RAG is a system that decides whether, when and how much to retrieve external information based on the input query.[3]

## Architecture

```
┌──────────┐
│  Router  │
└────┬─────┘
     ├──► [RETRIEVE] ──► [GRADE_DOCUMENTS] ──┬──► [GENERATE] ──┬──► END (useful)
     │                                       │                 ├──► [GENERATE] (hallucinated, retry)
     └──► [WEBSEARCH] ──► [GENERATE] ────────┘                 └──► [WEBSEARCH] (not useful)
```

- **Router** (`graph/chains/router.py`): Classifies the incoming question as `vectorstore` or `websearch`. Questions about Korean competition law cases are routed to the vector store; all other queries go to web search.
- **Retrieve** (`graph/nodes/retrieve.py`): Fetches top-k semantically similar document chunks from ChromaDB.
- **Grade Documents** (`graph/nodes/grade_documents.py`): Scores all retrieved documents in parallel with a batch LLM grader. Sets the `web_search` fallback flag only when no relevant documents remain.
- **Generate** (`graph/nodes/generate.py`): Synthesizes a Korean-language answer grounded in the provided documents, citing case names, ruling numbers, and dates where available. Tracks a `retry_count` to prevent infinite loops (cap: 3).
- **Hallucination Grader** (`graph/chains/hallucination_grader.py`): Verifies the generated answer is supported by the retrieved documents.
- **Answer Grader** (`graph/chains/answer_grader.py`): Verifies the generated answer actually addresses the user's question. If not useful, the system falls back to web search. If retries are exhausted, the best available answer is returned.

# How to Run

## Step 1: Environment Setup

```bash
git clone <repo-url>
cd adaptive_rag
uv sync
```

## Step 2: Configure API Keys

Copy `.env.example` to `.env` and fill in:

```
OPENROUTER_API_KEY=your_openrouter_key_here
TAVILY_API_KEY=your_tavily_key_here

# Model to use via OpenRouter (default: openai/gpt-4o-mini)
OPENROUTER_MODEL=openai/gpt-4o-mini

# Required only for LlamaParse ingestion
LLAMA_CLOUD_API_KEY=your_llama_cloud_key_here

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
```

## Step 3: Data Ingestion

Place PDF files in `data/korean_competition_law_cases/`, then run:

```bash
# Default (PyPDF, no table parsing)
uv run python data/ingest.py --pdf_path=./data/korean_competition_law_cases

# With Unstructured table extraction (local, heavy deps)
uv run python data/ingest.py --table_parser=unstructured --pdf_path=./data/korean_competition_law_cases

# With LlamaParse (best table quality, requires LLAMA_CLOUD_API_KEY)
uv run python data/ingest.py --table_parser=llamaparse --pdf_path=./data/korean_competition_law_cases

# LlamaParse with custom page limit and cache directory
uv run python data/ingest.py --table_parser=llamaparse --pdf_path=./data/korean_competition_law_cases --llamaparse_page_limit=1000 --llamaparse_cache_dir=./data/llamaparse_cache
```

| Flag | Default | Description |
|---|---|---|
| `--pdf_path` | `./data/korean_competition_law_cases` | Path to PDF directory |
| `--table_parser` | `none` | `"none"` (PyPDF), `"unstructured"`, or `"llamaparse"` |
| `--llamaparse_page_limit` | `1000` | Max pages to process per run (LlamaParse only) |
| `--llamaparse_cache_dir` | `./data/llamaparse_cache` | Cache dir for resumable LlamaParse runs |

For LlamaParse, results are cached per-file — rerun to resume where you left off.

> **Note**: The ChromaDB retriever is initialized lazily on first query. If the collection is empty (ingest has never been run), the graph will raise a `RuntimeError` with instructions to run ingestion first.

See [preprocessing.md](data/preprocessing.md) for details on all preprocessing steps applied.

## Step 4: Execute the Graph

```bash
uv run python main.py
```

## References

[1] Asai, Akari, et al. "Self-rag: Learning to retrieve, generate, and critique through self-reflection." The Twelfth International Conference on Learning Representations. 2023.

[2] Yan, Shi-Qi, et al. "Corrective retrieval augmented generation." (2024).

[3] Jeong, Soyeong, et al. "Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity." Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2024.
