Inspired by https://pub.towardsai.net/how-to-build-advanced-rag-with-langgraph-a70350396589

Self-reflection for RAG is the system that reviews its own answers and decide whether to retrieve more relevant data or not.[1]
Corrective RAG is a RAG system that corrects the retrieved documents before answering.[2]
Adaptive RAG is a system that decides whether, when and how much to retrieve external information based on the input query. [3]       


- The Router (router.py or Router Node): This is the "brain" at the entry point. It analyzes the incoming query and decides if it can be answered using the local Vector Store (proprietary/internal data) or if it requires a Web Search (for recent events or general knowledge).
- The Retriever (retriever.py): If the router chooses the vector store, this component fetches the top-$k$ most relevant document chunks using semantic search (usually via FAISS or ChromaDB).
- The Document Grader (grader.py): This is a critical "Self-Correction" step. An LLM reviews each retrieved document and gives it a binary score (Relevant/Not Relevant). If no documents are relevant, the system may trigger a web search or rewrite the query.
- The Generator (generator.py): This node takes the relevant documents and the original question to synthesize a final answer.
- The Hallucination & Answer Grader: After generation, the system checks:
    - Hallucination Check: Is the answer grounded in the retrieved documents?
    - Answer Check: Does the generated answer actually address the user's question?
- The Question Re-writer: If the retrieved documents are poor or the answer is deemed irrelevant, this node uses an LLM to optimize the user's query for a better second-pass retrieval.

# How to Run
## Step 1: Environment Setup
```bash
git clone https://github.com/
cd /adaptive_rag
```

## Step 2: Configure API Keys
```
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
LANGCHAIN_TRACING_V2=true  # Optional: for LangSmith visualization
LANGCHAIN_API_KEY=your_langsmith_key_here
```

## Step 3: Data Ingestion

Place PDF files in `data/korean_competition_law_cases/`, then run:

```bash
# Default (PyPDF, no table parsing)
python data/ingest.py --pdf_path=./data/korean_competition_law_cases

# With Unstructured table extraction
python data/ingest.py --table_parser=unstructured --pdf_path=./data/korean_competition_law_cases

# With LlamaParse (best table quality)
python data/ingest.py --table_parser=llamaparse --pdf_path=./data/korean_competition_law_cases

# LlamaParse with custom page limit and PDF path
python data/ingest.py --table_parser=llamaparse --pdf_path=./data/korean_competition_law_cases --llamaparse_page_limit=1000 --llamaparse_cache_dir=./data/llamaparse_cache
```

| Flag | Default | Description |
|---|---|---|
| `--pdf_path` | `./data/korean_competition_law_cases` | Path to PDF directory |
| `--table_parser` | `none` | `"none"`, `"unstructured"`, or `"llamaparse"` |
| `--llamaparse_page_limit` | `1000` | Max pages per run (LlamaParse only) |
| `--llamaparse_cache_dir` | `./data/llamaparse_cache` | Cache dir for resumable LlamaParse |

For LlamaParse, results are cached per-file — rerun to resume where you left off.

See [preprocessing.md](data/preprocessing.md) for details on all preprocessing steps applied.

## Step 4: Execute the Graph
```
python main.py
```

Reference
[1] Asai, Akari, et al. "Self-rag: Learning to retrieve, generate, and critique through self-reflection." The Twelfth International Conference on Learning Representations. 2023.
[2] Yan, Shi-Qi, et al. "Corrective retrieval augmented generation." (2024).
[3] Jeong, Soyeong, et al. "Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity." Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2024.

utilize chroma