"""
download_raw_data.py
--------------------
Handles all external data acquisition:
  - PubMed abstracts via Biopython/Entrez
  - FDA guideline PDFs via HTTP
  - Belmont Report ethics summary (local generation)

Also defines `data_paths`, the single source of truth for all data
directory locations used across the project.
"""

import os
import io
from pathlib import Path

import requests
from pypdf import PdfReader
from Bio import Entrez, Medline

# ---------------------------------------------------------------------------
# Data directory layout
# Paths are anchored to this file's location so they resolve correctly
# regardless of which directory the user runs Python from.
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent

data_paths = {
    'base':   str(_DATA_DIR),
    'pubmed': str(_DATA_DIR / 'pubmed'),
    'fda':    str(_DATA_DIR / 'fda'),
    'ethics': str(_DATA_DIR / 'ethical_guidelines'),
    # MIMIC CSV files live directly in data/mimic/
    'mimic':  str(_DATA_DIR / 'mimic'),
}


def prep_paths():
    """Create all data directories if they do not already exist."""
    for path in data_paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")


# ---------------------------------------------------------------------------
# PubMed download
# ---------------------------------------------------------------------------

def download_pubmed_articles(query: str, max_articles: int = 20) -> int:
    """
    Search PubMed for `query` and save each article's title + abstract
    as a plain-text file under data/pubmed/<PMID>.txt.

    Requires ENTREZ_EMAIL env var to comply with NCBI usage policy.
    Returns the number of articles successfully saved.
    """
    Entrez.email = os.environ.get('ENTREZ_EMAIL')

    # Step 1: search for matching PMIDs
    handle = Entrez.esearch(db='pubmed', term=query, retmax=max_articles, sort='relevance')
    record = Entrez.read(handle)
    id_list = record['IdList']
    print(f"Found {len(id_list)} PubMed articles for query: '{query}'")

    # Step 2: fetch full Medline records for those PMIDs
    handle = Entrez.efetch(db='pubmed', id=id_list, rettype='medline', retmode='text')
    records = Medline.parse(handle)

    count = 0
    for record in records:
        pmid = record.get('PMID', '')
        title = record.get('TI', '')
        abstract = record.get('AB', '')

        if pmid:
            file_path = os.path.join(data_paths['pubmed'], f"{pmid}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Title: {title}\n\nAbstract: {abstract}")
            count += 1

    print(f"Saved {count} PubMed articles to {data_paths['pubmed']}")
    return count


# ---------------------------------------------------------------------------
# FDA guideline PDF download
# ---------------------------------------------------------------------------

def download_extract_from_pdf(url: str, output_path: str) -> bool:
    """
    Download a PDF from `url`, save the raw PDF to `output_path`,
    and write extracted plain text to the same path with a .txt extension.

    Returns True on success, False on network error.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise for 4xx/5xx responses

        # Save raw PDF
        with open(output_path, 'wb') as f:
            f.write(response.content)

        # Extract text from PDF pages
        reader = PdfReader(io.BytesIO(response.content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + '\n\n'

        # Write text alongside the PDF (same name, .txt extension)
        text_output_path = os.path.splitext(output_path)[0] + '.txt'  # was splittext (typo)
        with open(text_output_path, 'w') as f:
            f.write(text)

        print(f"PDF downloaded and extracted to {text_output_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False


# ---------------------------------------------------------------------------
# Ethics guidelines
# ---------------------------------------------------------------------------

def prep_ethics_guidelines():
    """
    Write a short Belmont Report summary to data/ethical_guidelines/belmont_summary.txt.
    This acts as the retrieval corpus for the Ethics Specialist agent.
    """
    ethics_content = """Title: Summary of the Belmont Report Principles for Clinical Research

1. Respect for Persons: This principle requires that individuals be treated as autonomous
agents and that persons with diminished autonomy are entitled to protection. This translates
to robust informed consent processes. Inclusion/exclusion criteria must not unduly target or
coerce vulnerable populations, such as economically disadvantaged individuals, prisoners, or
those with severe cognitive impairments, unless the research is directly intended to benefit
that population.

2. Beneficence: This principle involves two complementary rules: (1) do not harm and
(2) maximize possible benefits and minimize possible harms. The criteria must be designed to
select a population that is most likely to benefit and least likely to be harmed by the
intervention. The risks to subjects must be reasonable in relation to anticipated benefits.

3. Justice: This principle concerns the fairness of distribution of the burdens and benefits
of research. The selection of research subjects must be equitable. Criteria should not be
designed to exclude certain groups without a sound scientific or safety-related justification.
For example, excluding participants based on race, gender, or socioeconomic status is unjust
unless there is a clear rationale related to the drug's mechanism or risk profile.
"""
    ethics_path = os.path.join(data_paths['ethics'], 'belmont_summary.txt')
    with open(ethics_path, 'w') as f:
        f.write(ethics_content)
    print(f"Created ethics guidelines at {ethics_path}")
