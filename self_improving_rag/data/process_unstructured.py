"""
process_unstructured.py
-----------------------
Builds FAISS vector stores from the three text corpora:
  - PubMed abstracts   → pubmed_retriever
  - FDA guidelines     → fda_retriever
  - Ethics guidelines  → ethics_retriever

Also exposes `create_retrievers`, which is the single entry point
called by main.py after data ingestion is complete.
"""

import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader  # 'loaders' (plural)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from data.download_raw_data import data_paths


def create_vector_store(folder_path: str, embedding_model, store_name: str):
    """
    Load all .txt files from `folder_path`, split them into chunks,
    embed them, and return a FAISS vector store.

    The store is also saved to disk under data/<store_name>/ so it can
    be reloaded on subsequent runs without re-embedding.

    Returns None if no documents are found in the folder.
    """
    loader = DirectoryLoader(
        folder_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs = loader.load()

    if not docs:
        print(f"No documents found in {folder_path}")
        return None

    # Split into overlapping chunks so context is not cut off at boundaries
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    print(f"Loaded {len(docs)} documents, split into {len(texts)} chunks")

    vector_store = FAISS.from_documents(texts, embedding_model)

    save_path = os.path.join(data_paths['base'], store_name)
    vector_store.save_local(save_path)
    print(f"Vector store '{store_name}' saved to {save_path}")

    return vector_store


def create_retrievers(embedding_model, db_path: str) -> dict:
    """
    Build all three vector stores and return a dict containing:
      - pubmed_retriever  : retriever over PubMed abstracts
      - fda_retriever     : retriever over FDA guideline text
      - ethics_retriever  : retriever over Belmont Report summary
      - mimic_db_path     : path to the DuckDB file used by the SQL analyst

    `db_path` is the path returned by `load_real_mimic_data()` and is
    forwarded here so the analyst node can open the database.
    """
    pubmed_db = create_vector_store(data_paths['pubmed'], embedding_model, 'faiss_pubmed')
    fda_db    = create_vector_store(data_paths['fda'],    embedding_model, 'faiss_fda')
    ethics_db = create_vector_store(data_paths['ethics'], embedding_model, 'faiss_ethics')

    return {
        # as_retriever wraps the FAISS store in a standard LangChain retriever
        'pubmed_retriever':  pubmed_db.as_retriever(search_kwargs={'k': 3}) if pubmed_db else None,
        'fda_retriever':     fda_db.as_retriever(search_kwargs={'k': 3})    if fda_db    else None,
        'ethics_retriever':  ethics_db.as_retriever(search_kwargs={'k': 2}) if ethics_db else None,
        'mimic_db_path':     db_path,
    }
