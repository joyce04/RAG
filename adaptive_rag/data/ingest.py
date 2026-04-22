import os
import re
import json
from pathlib import Path

import fire

from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader #WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# ──────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────
DEFAULT_PDF_PATH = "./data/korean_competition_law_cases"
DEFAULT_TABLE_PARSER = "none"  # "none" | "unstructured" | "llamaparse"
DEFAULT_LLAMAPARSE_PAGE_LIMIT = 1000
DEFAULT_LLAMAPARSE_CACHE_DIR = "./data/llamaparse_cache"


# ──────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────
def normalize_korean_spacing(text: str) -> str:
    """
    Collapse runs of 3+ single-spaced Korean characters.
    e.g. '공 정 거 래 위 원 회' → '공정거래위원회'
    Preserves 2-char phrases like '피심인 삼성'.
    """
    # Match a run of 3 or more Korean chars separated by single spaces
    # e.g. "공 정 거 래 위 원 회"
    run_pattern = re.compile(r'(?:[가-힣] ){2,}[가-힣]')

    def collapse_run(match: re.Match) -> str:
        return match.group(0).replace(' ', '')

    return run_pattern.sub(collapse_run, text)


def preprocess_page(text: str) -> str:
    """Apply preprocessing steps 1, 2, 4 from preprocessing.md."""
    # 1. Remove page number headers like "- 1 -", "- 46 -"
    text = re.sub(r'^\s*-\s*\d+\s*-\s*\n?', '', text)
    # 4. Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # 2. Normalize Korean character spacing
    text = normalize_korean_spacing(text)
    return text


# ──────────────────────────────────────────────
# Document loaders
# ──────────────────────────────────────────────
def load_with_pypdf(pdf_path: str) -> list[Document]:
    """Load PDFs with PyPDF (default). No table extraction."""
    loader = PyPDFDirectoryLoader(pdf_path)
    docs = loader.load()

    # Preprocess each page
    for doc in docs:
        doc.page_content = preprocess_page(doc.page_content)
        doc.metadata["content_type"] = "text"

    # 3. Filter sparse pages (< 30 chars after preprocessing)
    docs = [d for d in docs if len(d.page_content.strip()) > 30]

    return docs


def load_with_unstructured(pdf_path: str) -> list[Document]:
    """Load PDFs with Unstructured — extracts tables as HTML."""
    from unstructured.partition.pdf import partition_pdf

    pdf_files = sorted(Path(pdf_path).glob("*.pdf"))
    docs = []

    for pdf_file in pdf_files:
        print(f"[unstructured] Processing: {pdf_file.name}")
        elements = partition_pdf(
            filename=str(pdf_file),
            strategy="hi_res",
            infer_table_structure=True,
            languages=["kor"],
        )
        for el in elements:
            if el.category == "Table":
                content = f"[TABLE]\n{el.metadata.text_as_html}\n[/TABLE]"
                content_type = "table"
            else:
                content = str(el)
                content_type = "text"

            content = preprocess_page(content)
            if len(content.strip()) > 30:
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": str(pdf_file),
                        "content_type": content_type,
                        "element_type": el.category,
                    },
                ))

    return docs


def load_with_llamaparse(pdf_path: str, page_limit: int = 1000, llamaparse_cache_dir: str = DEFAULT_LLAMAPARSE_CACHE_DIR) -> list[Document]:
    """
    Load PDFs with LlamaParse — best table extraction quality.
    Caches results per-file as JSON. Resumes from where it left off.
    Stops after processing `page_limit` pages (cumulative across runs).
    """
    from llama_parse import LlamaParse
    from pypdf import PdfReader

    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")

    cache_dir = Path(llamaparse_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        language="ko",
    )

    pdf_files = sorted(Path(pdf_path).glob("*.pdf"))
    docs = []
    pages_processed = 0

    for pdf_file in pdf_files:
        cache_file = cache_dir / f"{pdf_file.stem}.json"

        # Load from cache if already processed
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            for item in cached:
                doc = Document(
                    page_content=preprocess_page(item["page_content"]),
                    metadata=item["metadata"],
                )
                if len(doc.page_content.strip()) > 30:
                    docs.append(doc)
            print(f"[llamaparse] Loaded from cache: {pdf_file.name} ({len(cached)} chunks)")
            continue

        # Check page count before processing
        try:
            reader = PdfReader(str(pdf_file))
            file_pages = len(reader.pages)
        except Exception:
            file_pages = 0

        if pages_processed + file_pages > page_limit:
            remaining = page_limit - pages_processed
            print(
                f"[llamaparse] Stopping — limit {page_limit} reached. "
                f"Processed {pages_processed} pages so far. "
                f"{pdf_file.name} has {file_pages} pages (need {remaining} remaining)."
            )
            break

        # Parse with LlamaParse
        print(f"[llamaparse] Parsing: {pdf_file.name} ({file_pages} pages)...")
        try:
            results = parser.load_data(str(pdf_file))
        except Exception as e:
            print(f"[llamaparse] ERROR on {pdf_file.name}: {e}")
            continue

        # Cache results
        cached_items = []
        for result in results:
            metadata = {
                "source": str(pdf_file),
                "content_type": "text",  # LlamaParse returns markdown; tables are inline
            }
            cached_items.append({
                "page_content": result.text,
                "metadata": metadata,
            })
            doc = Document(
                page_content=preprocess_page(result.text),
                metadata=metadata,
            )
            if len(doc.page_content.strip()) > 30:
                docs.append(doc)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cached_items, f, ensure_ascii=False, indent=2)

        pages_processed += file_pages
        print(f"[llamaparse] Done: {pdf_file.name} — {pages_processed}/{page_limit} pages used")

    print(f"[llamaparse] Total documents loaded: {len(docs)}")
    return docs


# ──────────────────────────────────────────────
# Main ingestion pipeline
# ──────────────────────────────────────────────
def ingest(
    pdf_path: str = DEFAULT_PDF_PATH,
    table_parser: str = DEFAULT_TABLE_PARSER,
    llamaparse_page_limit: int = DEFAULT_LLAMAPARSE_PAGE_LIMIT,
    llamaparse_cache_dir: str = DEFAULT_LLAMAPARSE_CACHE_DIR,
):
    """
    Run the full ingestion pipeline: load PDFs, preprocess, split, embed, and store in ChromaDB.

    Args:
        pdf_path: Path to directory containing PDF files.
        table_parser: Parser backend — "none" (PyPDF), "unstructured", or "llamaparse".
        llamaparse_page_limit: Max pages to process per run (LlamaParse only). Cached results don't count.
        llamaparse_cache_dir: Directory to cache LlamaParse results for resumable processing.
    """
    # 1. Load documents with selected parser
    if table_parser == "llamaparse":
        docs = load_with_llamaparse(pdf_path, page_limit=llamaparse_page_limit, llamaparse_cache_dir=llamaparse_cache_dir)
    elif table_parser == "unstructured":
        docs = load_with_unstructured(pdf_path)
    else:
        docs = load_with_pypdf(pdf_path)

    # 2. Separate tables from text (Strategy 2)
    #    Tables are stored as whole chunks; text is stitched + split
    table_docs = [d for d in docs if d.metadata.get("content_type") == "table"]
    text_docs = [d for d in docs if d.metadata.get("content_type") != "table"]

    # 3. Stitch text pages by source file, embedding [PAGE:N] markers to preserve page info
    pdf_pages: dict[str, list[tuple[int, str]]] = {}
    for doc in text_docs:
        source = doc.metadata["source"]
        page_num = doc.metadata.get("page", 0)
        page_num = (page_num + 1) if isinstance(page_num, int) else 1
        if source not in pdf_pages:
            pdf_pages[source] = []
        pdf_pages[source].append((page_num, doc.page_content))

    stitched_docs = []
    for source, pages in pdf_pages.items():
        pages.sort(key=lambda x: x[0])
        stitched = "\n".join(f"[PAGE:{p}]\n{content}" for p, content in pages)
        stitched_docs.append(
            Document(page_content=stitched, metadata={"source": source, "content_type": "text"})
        )

    # 4. Split: MarkdownHeaderTextSplitter (primary) → RecursiveCharacterTextSplitter (secondary)
    #    Page numbers are propagated from [PAGE:N] markers embedded in the stitched text.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    _page_marker = re.compile(r'\[PAGE:(\d+)\]')

    text_splits = []
    for doc in stitched_docs:
        # Primary split by markdown headers (preserves [PAGE:N] markers in text)
        md_splits = markdown_splitter.split_text(doc.page_content)

        # Secondary split by character count for large chunks (e.g. PDFs without headers)
        sub_chunks: list[Document] = []
        for md_split in md_splits:
            md_split.metadata.update(doc.metadata)
            sub_chunks.extend(char_splitter.split_documents([md_split]))

        # Propagate page numbers: carry the last seen [PAGE:N] marker forward
        current_page = 1
        for chunk in sub_chunks:
            markers = _page_marker.findall(chunk.page_content)
            if markers:
                current_page = int(markers[0])
            chunk.metadata["page"] = current_page
            chunk.page_content = _page_marker.sub("", chunk.page_content).strip()

        text_splits.extend(sub_chunks)

    # 5. Combine: split text + whole tables
    all_chunks = text_splits + table_docs

    print(f"Text chunks: {len(text_splits)}, Table chunks: {len(table_docs)}, Total: {len(all_chunks)}")

    # 6. Embed and store in ChromaDB
    ## embedding generation : text chunk to vectors
    ## OpenAIEmbeddings [text-embedding-3] : SOTA/cost-effective
    ## HuggingFaceEmbeddings / OllamaEmbeddings [BAAI/bge-m3, sentence-transformers/all-MiniLM-L6-v2, Nomic-embed-text] : local/privacy/possible to fine-tune
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        collection_name='rag-chroma',
        persist_directory='./data/chroma_db',
        embedding=OpenAIEmbeddings(
            model='text-embedding-3-small',
            openai_api_base='https://openrouter.ai/api/v1',
            openai_api_key=os.environ.get('OPENROUTER_API_KEY'),
        ),
    )
    retriever = vectorstore.as_retriever()

    retriever = vectorstore.as_retriever()
    print("Ingestion complete.")
    return retriever


# ──────────────────────────────────────────────
# Lazy retriever (loads from existing ChromaDB)
# ──────────────────────────────────────────────
_retriever = None


def get_retriever():
    """
    Return a retriever backed by the existing ChromaDB collection.
    Initialized on first call (lazy) so importing this module during ingestion
    does not open a DB connection prematurely.

    Raises RuntimeError if the collection is empty (ingest() has not been run).
    """
    global _retriever
    if _retriever is None:
        vectorstore = Chroma(
            collection_name='rag-chroma',
            persist_directory='./data/chroma_db',
            embedding_function=OpenAIEmbeddings(
                model='text-embedding-3-small',
                openai_api_base='https://openrouter.ai/api/v1',
                openai_api_key=os.environ.get('OPENROUTER_API_KEY'),
            ),
        )
        if vectorstore._collection.count() == 0:
            raise RuntimeError(
                "ChromaDB collection 'rag-chroma' is empty. "
                "Run `uv run python data/ingest.py` first to populate it."
            )
        _retriever = vectorstore.as_retriever()
    return _retriever


if __name__ == "__main__":
    fire.Fire(ingest)
