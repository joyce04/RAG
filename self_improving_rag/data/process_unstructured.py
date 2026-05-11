"""
process_unstructured.py
-----------------------
Builds FAISS vector stores from the four text corpora:
  - DrugBank vocabulary  → drugbank_retriever
  - DDI corpus           → ddi_retriever
  - PubMed abstracts     → pubmed_retriever
  - KEGG DRUG entries    → kegg_retriever

Also exposes `create_retrievers`, the single entry point called by
main.py after data download and DB construction.
"""

import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from data.download_raw_data import data_paths


def create_vector_store(folder_path: str, embedding_model, store_name: str):
    """
    Load all .txt files from `folder_path`, split into chunks, embed,
    and return a FAISS vector store.

    Saved to data/<store_name>/ for inspection. Returns None if the
    folder has no documents.
    """
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None

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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    print(f"[{store_name}] {len(docs)} docs → {len(texts)} chunks")

    vector_store = FAISS.from_documents(texts, embedding_model)

    save_path = os.path.join(data_paths['base'], store_name)
    vector_store.save_local(save_path)
    print(f"Vector store '{store_name}' saved to {save_path}")

    return vector_store


def create_retrievers(
    embedding_model,
    mimic_db_path: str,
    drugbank_db_path: str,
) -> dict:
    """
    Build all four FAISS vector stores and return a knowledge_stores dict:

      drugbank_retriever  : DrugBank vocabulary + MOA text
      ddi_retriever       : DDI corpus (severity + mechanism)
      pubmed_retriever    : PubMed pharmacology abstracts
      kegg_retriever      : KEGG DRUG pathway entries
      mimic_db_path       : path to MIMIC-III DuckDB (for MIMIC Prescribing Analyst)
      drugbank_db_path    : path to DrugBank/DDI DuckDB (for Interaction DB Analyst)
    """
    drugbank_db  = create_vector_store(data_paths['drugbank'], embedding_model, 'faiss_drugbank')
    ddi_db       = create_vector_store(data_paths['ddi'],      embedding_model, 'faiss_ddi')
    pubmed_db    = create_vector_store(data_paths['pubmed'],   embedding_model, 'faiss_pubmed')
    kegg_db      = create_vector_store(data_paths['kegg'],     embedding_model, 'faiss_kegg')

    return {
        'drugbank_retriever': drugbank_db.as_retriever(search_kwargs={'k': 5}) if drugbank_db else None,
        'ddi_retriever':      ddi_db.as_retriever(search_kwargs={'k': 8})      if ddi_db      else None,
        'pubmed_retriever':   pubmed_db.as_retriever(search_kwargs={'k': 5})   if pubmed_db   else None,
        'kegg_retriever':     kegg_db.as_retriever(search_kwargs={'k': 3})     if kegg_db     else None,
        'mimic_db_path':      mimic_db_path,
        'drugbank_db_path':   drugbank_db_path,
    }
