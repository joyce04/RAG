"""
download_raw_data.py
--------------------
Handles all external data acquisition for the Drug Relationship RAG system:
  - DrugBank vocabulary (CC BY 4.0 open data)
  - DDI interaction corpus (NLP-annotated public dataset)
  - PubMed abstracts (pharmacology-scoped via Entrez)
  - KEGG DRUG pathway entries (REST API, no key required)

Also defines `data_paths`, the single source of truth for all data
directory locations used across the project.
"""

import os
import time
import requests
from pathlib import Path

from Bio import Entrez, Medline

_DATA_DIR = Path(__file__).parent

data_paths = {
    'base':        str(_DATA_DIR),
    'pubmed':      str(_DATA_DIR / 'pubmed'),
    'drugbank':    str(_DATA_DIR / 'drugbank'),
    'ddi':         str(_DATA_DIR / 'ddi'),
    'kegg':        str(_DATA_DIR / 'kegg'),
    'mimic':       str(_DATA_DIR / 'mimic'),
    'chembl_db':   str(_DATA_DIR / 'chembl'),
    'drugbank_db': str(_DATA_DIR / 'drugbank_db'),
}


def prep_paths() -> None:
    """Create all data directories if they do not already exist."""
    for path in data_paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")


# ---------------------------------------------------------------------------
# PubMed download (reused, pharmacology-scoped query)
# ---------------------------------------------------------------------------

def download_pubmed_articles(
    query: str = "(drug interactions) AND (pharmacology) AND (clinical pharmacokinetics)",
    max_articles: int = 20,
) -> int:
    """
    Search PubMed for `query` and save each article's title + abstract
    as a plain-text file under data/pubmed/<PMID>.txt.

    Requires ENTREZ_EMAIL env var to comply with NCBI usage policy.
    Returns the number of articles successfully saved.
    """
    Entrez.email = os.environ.get('ENTREZ_EMAIL', 'researcher@example.com')

    handle = Entrez.esearch(db='pubmed', term=query, retmax=max_articles, sort='relevance')
    record = Entrez.read(handle)
    id_list = record['IdList']
    print(f"Found {len(id_list)} PubMed articles for query: '{query}'")

    handle = Entrez.efetch(db='pubmed', id=id_list, rettype='medline', retmode='text')
    records = Medline.parse(handle)

    count = 0
    for record in records:
        pmid     = record.get('PMID', '')
        title    = record.get('TI', '')
        abstract = record.get('AB', '')
        if pmid:
            file_path = os.path.join(data_paths['pubmed'], f"{pmid}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Title: {title}\n\nAbstract: {abstract}")
            count += 1

    print(f"Saved {count} PubMed articles to {data_paths['pubmed']}")
    return count


# ---------------------------------------------------------------------------
# DrugBank Open Vocabulary
# ---------------------------------------------------------------------------

def download_drugbank_vocab() -> int:
    """
    Download DrugBank Open Data vocabulary (CC BY 4.0) and write one .txt
    file per drug chunk to data/drugbank/.

    Falls back to a hardcoded vocabulary of common drugs if the download
    fails (network issue or the URL changes).
    """
    # DrugBank open vocabulary CSV — CC BY 4.0, no account required
    url = "https://go.drugbank.com/releases/latest/downloads/all-drug-links"
    output_path = os.path.join(data_paths['drugbank'], 'drugbank_vocab.txt')

    if os.path.exists(output_path):
        print("DrugBank vocabulary already on disk, skipping download.")
        return _count_txt_files(data_paths['drugbank'])

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(output_path, 'w') as f:
            f.write(resp.text)
        print(f"DrugBank vocabulary downloaded to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"DrugBank download failed ({e}). Writing fallback vocabulary.")
        _write_fallback_drugbank_vocab(output_path)

    return _count_txt_files(data_paths['drugbank'])


def _write_fallback_drugbank_vocab(output_path: str) -> None:
    """Hardcoded vocabulary for the most commonly queried drugs."""
    content = """Drug: Metformin | Class: Biguanide | Mechanism: Inhibits hepatic gluconeogenesis via AMPK activation; reduces insulin resistance | Targets: AMP-activated protein kinase (AMPK), Complex I of mitochondrial respiratory chain | Half-life: 4-8.7 hours | Clearance: Renal (unchanged)

Drug: Warfarin | Class: Vitamin K antagonist (anticoagulant) | Mechanism: Inhibits VKORC1, blocking recycling of vitamin K epoxide; reduces synthesis of clotting factors II, VII, IX, X | Targets: Vitamin K epoxide reductase complex subunit 1 (VKORC1), CYP2C9 | Half-life: 20-60 hours | Clearance: Hepatic (CYP2C9)

Drug: Furosemide | Class: Loop diuretic | Mechanism: Inhibits NKCC2 cotransporter in thick ascending loop of Henle; blocks Na/K/2Cl reabsorption | Targets: Solute carrier family 12 member 1 (SLC12A1/NKCC2) | Half-life: 2 hours | Clearance: Renal 66%, hepatic 33%

Drug: Atorvastatin | Class: HMG-CoA reductase inhibitor (statin) | Mechanism: Competitively inhibits HMG-CoA reductase, rate-limiting enzyme in cholesterol synthesis | Targets: 3-hydroxy-3-methylglutaryl-CoA reductase (HMGCR) | Half-life: 14 hours | Clearance: Hepatic (CYP3A4)

Drug: Amiodarone | Class: Class III antiarrhythmic | Mechanism: Blocks multiple ion channels (K+, Na+, Ca2+); also has beta-blocking and vasodilatory properties | Targets: Potassium channels (hERG/KCNH2), sodium channels (SCN5A), calcium channels | Half-life: 40-55 days | Clearance: Hepatic (CYP3A4, CYP2C8)

Drug: Aspirin | Class: NSAID / antiplatelet | Mechanism: Irreversibly acetylates COX-1 and COX-2, blocking thromboxane A2 synthesis; at low doses selectively inhibits platelet aggregation | Targets: Cyclooxygenase-1 (COX-1/PTGS1), Cyclooxygenase-2 (COX-2/PTGS2) | Half-life: 15-20 min (aspirin), 2-3 hours (salicylate) | Clearance: Hepatic

Drug: Heparin | Class: Anticoagulant (unfractionated) | Mechanism: Binds antithrombin III, accelerating its inhibition of thrombin (factor IIa) and factor Xa by 1000-fold | Targets: Antithrombin III (SERPINC1), Thrombin (F2), Factor Xa (F10) | Half-life: 1-2 hours | Clearance: Reticuloendothelial system

Drug: Metoprolol | Class: Beta-1 selective adrenergic blocker | Mechanism: Competitively blocks beta-1 adrenergic receptors in cardiac tissue; reduces heart rate and contractility | Targets: Beta-1 adrenergic receptor (ADRB1) | Half-life: 3-7 hours | Clearance: Hepatic (CYP2D6)

Drug: Digoxin | Class: Cardiac glycoside | Mechanism: Inhibits Na+/K+-ATPase pump, increasing intracellular calcium; positive inotropic effect; slows AV node conduction via vagal activation | Targets: Na+/K+-ATPase (ATP1A1) | Half-life: 36-48 hours | Clearance: Renal 60-80%

Drug: Spironolactone | Class: Aldosterone antagonist (potassium-sparing diuretic) | Mechanism: Competitively blocks aldosterone receptor in distal tubule; reduces sodium retention and potassium excretion | Targets: Mineralocorticoid receptor (NR3C2) | Half-life: 1.4 hours (active metabolite canrenone 17 hours) | Clearance: Hepatic

Drug: Empagliflozin | Class: SGLT2 inhibitor | Mechanism: Inhibits SGLT2 in proximal renal tubule, reducing glucose reabsorption and promoting glucosuria | Targets: Sodium-glucose cotransporter 2 (SLC5A2/SGLT2) | Half-life: 12.4 hours | Clearance: Renal and hepatic

Drug: Dapagliflozin | Class: SGLT2 inhibitor | Mechanism: Selective SGLT2 inhibitor; reduces renal glucose threshold, increasing urinary glucose excretion | Targets: Sodium-glucose cotransporter 2 (SLC5A2/SGLT2) | Half-life: 12.9 hours | Clearance: Hepatic (UGT1A9)

Drug: Canagliflozin | Class: SGLT2 inhibitor | Mechanism: Inhibits SGLT2 and to lesser extent SGLT1; reduces glucose reabsorption; osmotic diuresis and natriuresis | Targets: SGLT2 (SLC5A2), SGLT1 (SLC5A1) | Half-life: 10.6-13.1 hours | Clearance: Hepatic

Drug: Insulin (Regular) | Class: Hormone / antidiabetic | Mechanism: Binds insulin receptor, activating tyrosine kinase signalling; promotes glucose uptake in muscle and adipose, suppresses hepatic gluconeogenesis | Targets: Insulin receptor (INSR) | Half-life: 30-60 min | Clearance: Hepatic and renal

Drug: Potassium Chloride | Class: Electrolyte supplement | Mechanism: Replenishes potassium, essential for Na+/K+-ATPase function, membrane potential, cardiac rhythm | Targets: N/A (electrolyte replacement) | Half-life: N/A | Clearance: Renal
"""
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Wrote fallback DrugBank vocabulary to {output_path}")


# ---------------------------------------------------------------------------
# DDI corpus
# ---------------------------------------------------------------------------

def download_ddi_corpus() -> int:
    """
    Download a publicly available DDI corpus and write one .txt file per
    drug to data/ddi/ for FAISS indexing.

    Falls back to a hardcoded DDI summary if the download fails.
    """
    output_path = os.path.join(data_paths['ddi'], 'ddi_corpus.txt')

    if os.path.exists(output_path):
        print("DDI corpus already on disk, skipping download.")
        return _count_txt_files(data_paths['ddi'])

    try:
        # Public DDI data from NLM/DailyMed structured product labeling (sample)
        url = "https://raw.githubusercontent.com/tticoin/DRUG-NER/master/data/ddi2013_train_pharmacokinetics.txt"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with open(output_path, 'w') as f:
            f.write(resp.text)
        print(f"DDI corpus downloaded to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"DDI corpus download failed ({e}). Writing fallback DDI data.")
        _write_fallback_ddi_corpus(output_path)

    return _count_txt_files(data_paths['ddi'])


def _write_fallback_ddi_corpus(output_path: str) -> None:
    """Hardcoded DDI summary for clinically significant pairs."""
    content = """DRUG INTERACTION CORPUS — Clinically Significant Pairs

=== CONTRAINDICATED INTERACTIONS ===

Warfarin + Aspirin (high-dose)
Severity: Contraindicated (combined use at therapeutic doses)
Mechanism: Pharmacodynamic synergy — both impair haemostasis via different mechanisms (VKORC1 inhibition + COX-1 inhibition). Aspirin also displaces warfarin from protein binding.
Management: Avoid concurrent use unless benefit clearly outweighs risk (e.g. mechanical heart valves). Monitor INR closely if unavoidable.

=== MAJOR INTERACTIONS ===

Warfarin + Amiodarone
Severity: Major
Mechanism: Amiodarone inhibits CYP2C9, the primary enzyme metabolising warfarin (S-warfarin). Increases warfarin plasma levels 30-50%. Also inhibits CYP3A4 (metabolises R-warfarin).
Management: Reduce warfarin dose by 30-50% when starting amiodarone. Monitor INR weekly for first month, then monthly. Effect persists weeks after amiodarone discontinuation due to long half-life (40-55 days).

Furosemide + Digoxin
Severity: Major
Mechanism: Furosemide causes hypokalaemia and hypomagnesaemia. Low potassium increases myocardial sensitivity to digoxin by competing with digoxin for Na+/K+-ATPase binding. Risk of digoxin toxicity (arrhythmias) even at normal digoxin levels.
Management: Monitor serum electrolytes and digoxin levels closely. Supplement potassium. Consider potassium-sparing diuretic combination.

Metformin + Contrast Media (iodinated)
Severity: Major
Mechanism: Iodinated contrast can precipitate acute kidney injury; impaired renal clearance leads to metformin accumulation and risk of lactic acidosis.
Management: Hold metformin 48 hours before and after contrast administration. Resume only when renal function confirmed normal.

Spironolactone + ACE Inhibitors
Severity: Major
Mechanism: Pharmacodynamic synergy in potassium retention. ACE inhibitors reduce aldosterone production; spironolactone blocks aldosterone receptors. Combined effect causes hyperkalaemia, especially in CKD.
Management: Monitor serum potassium closely. Use lowest effective doses. Avoid in significant renal impairment (eGFR < 30).

Digoxin + Amiodarone
Severity: Major
Mechanism: Amiodarone inhibits P-glycoprotein (MDR1), reducing digoxin renal and biliary clearance. Also inhibits renal tubular secretion of digoxin. Digoxin levels increase 70-100%.
Management: Reduce digoxin dose by 50% when starting amiodarone. Monitor digoxin levels and ECG.

=== MODERATE INTERACTIONS ===

Furosemide + Metformin
Severity: Moderate
Mechanism: Furosemide can cause volume depletion and pre-renal azotaemia, reducing metformin clearance and increasing risk of lactic acidosis.
Management: Monitor renal function. Ensure adequate hydration.

Aspirin + Furosemide
Severity: Moderate
Mechanism: High-dose aspirin (>3g/day) competitively inhibits organic acid secretion in the proximal tubule, reducing furosemide's entry into tubular lumen and blunting diuretic effect. Also promotes sodium retention.
Management: Avoid high-dose aspirin in patients on furosemide. Low-dose aspirin (75-325 mg) has minimal effect.

Warfarin + Metformin
Severity: Moderate
Mechanism: Metformin may enhance anticoagulant effect of warfarin (mechanism unclear, possibly altered gut flora affecting vitamin K synthesis).
Management: Monitor INR when starting or stopping metformin.

Atorvastatin + Amiodarone
Severity: Moderate
Mechanism: Amiodarone inhibits CYP3A4, the primary enzyme metabolising atorvastatin. Increased atorvastatin exposure raises risk of myopathy and rhabdomyolysis.
Management: Limit atorvastatin dose to 20 mg/day with amiodarone. Monitor for muscle symptoms (myalgia, CK elevation).

SGLT2 inhibitors + Loop diuretics (Furosemide, Bumetanide)
Severity: Moderate
Mechanism: Additive volume depletion. SGLT2 inhibitors cause osmotic diuresis; loop diuretics inhibit NKCC2. Combined effect risks dehydration, hypotension, acute kidney injury.
Management: Monitor blood pressure, renal function, and signs of volume depletion. Reduce loop diuretic dose when starting SGLT2 inhibitor.

Metoprolol + Digoxin
Severity: Moderate
Mechanism: Pharmacodynamic synergy in AV node conduction slowing. Both reduce heart rate — beta-blocker via sympathetic blockade, digoxin via vagal enhancement. Risk of bradycardia and heart block.
Management: Monitor ECG and heart rate. Use low doses of each.

=== MINOR INTERACTIONS ===

Metformin + Aspirin
Severity: Minor
Mechanism: Salicylates may enhance insulin sensitivity, potentiating metformin's glucose-lowering effect. Risk of hypoglycaemia with concurrent sulfonylurea.
Management: Monitor blood glucose if used together with insulin secretagogues.

Furosemide + Spironolactone
Severity: Minor (intended combination in many protocols)
Mechanism: Synergistic diuresis — furosemide blocks NKCC2, spironolactone blocks aldosterone. Spironolactone partially offsets furosemide-induced hypokalaemia.
Management: Monitor electrolytes (potassium, sodium, magnesium). Standard combination in heart failure.
"""
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Wrote fallback DDI corpus to {output_path}")


# ---------------------------------------------------------------------------
# KEGG DRUG pathway entries
# ---------------------------------------------------------------------------

_KEGG_DRUG_IDS = {
    'metformin':     'D04966',
    'warfarin':      'D00348',
    'furosemide':    'D00320',
    'atorvastatin':  'D00983',
    'amiodarone':    'D00283',
    'aspirin':       'D00109',
    'heparin':       'D04088',
    'metoprolol':    'D02358',
    'digoxin':       'D00298',
    'spironolactone':'D00443',
    'empagliflozin': 'D09813',
    'dapagliflozin': 'D09526',
    'canagliflozin': 'D10024',
    'insulin':       'D04540',
}


def download_kegg_drug_entries(drug_ids: dict = None) -> int:
    """
    Download KEGG DRUG entries via the free REST API (no key required).
    Saves one .txt file per drug to data/kegg/.

    Respects KEGG's rate limit with a 0.2s delay between requests.
    Uses skip-if-exists so repeated calls are idempotent.
    """
    if drug_ids is None:
        drug_ids = _KEGG_DRUG_IDS

    saved = 0
    for drug_name, kegg_id in drug_ids.items():
        output_path = os.path.join(data_paths['kegg'], f"{drug_name}.txt")
        if os.path.exists(output_path):
            saved += 1
            continue

        try:
            url = f"https://rest.kegg.jp/get/{kegg_id}"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            with open(output_path, 'w') as f:
                f.write(f"Drug: {drug_name.capitalize()} (KEGG ID: {kegg_id})\n\n")
                f.write(resp.text)
            print(f"Downloaded KEGG entry for {drug_name} ({kegg_id})")
            saved += 1
            time.sleep(0.2)  # KEGG rate limit: max 10 req/s
        except requests.exceptions.RequestException as e:
            print(f"KEGG download failed for {drug_name}: {e}")
            _write_fallback_kegg_entry(output_path, drug_name)
            saved += 1

    print(f"KEGG entries ready: {saved} drugs in {data_paths['kegg']}")
    return saved


def _write_fallback_kegg_entry(output_path: str, drug_name: str) -> None:
    """Write a minimal KEGG-style entry when the API is unavailable."""
    fallback_pathways = {
        'metformin':     'Insulin signalling pathway (hsa04910), AMPK signalling (hsa04152), Type II diabetes mellitus (hsa04930)',
        'warfarin':      'Vitamin K metabolism, Complement and coagulation cascades (hsa04610)',
        'furosemide':    'Aldosterone synthesis and secretion (hsa04925), Fluid/electrolyte regulation',
        'atorvastatin':  'Steroid biosynthesis (hsa00100), Metabolic pathways (hsa01100)',
        'amiodarone':    'Cardiac muscle contraction (hsa04260), Adrenergic signalling in cardiomyocytes (hsa04022)',
        'aspirin':       'Arachidonic acid metabolism (hsa00590), Platelet activation (hsa04611)',
        'digoxin':       'Cardiac muscle contraction (hsa04260), Calcium signalling pathway (hsa04020)',
        'spironolactone':'Aldosterone-regulated sodium reabsorption (hsa04960), Mineral absorption (hsa04978)',
        'empagliflozin': 'Insulin signalling pathway (hsa04910), Type II diabetes mellitus (hsa04930)',
        'dapagliflozin': 'Insulin signalling pathway (hsa04910), Type II diabetes mellitus (hsa04930)',
        'canagliflozin': 'Insulin signalling pathway (hsa04910), Type II diabetes mellitus (hsa04930)',
    }
    pathways = fallback_pathways.get(drug_name, 'Metabolic pathways (hsa01100)')
    content = f"""Drug: {drug_name.capitalize()} (KEGG fallback entry)

PATHWAY
{pathways}

TARGET
See DrugBank entry for molecular target details.

REMARK
This is a fallback entry generated when the KEGG REST API was unavailable.
"""
    with open(output_path, 'w') as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_txt_files(folder: str) -> int:
    return sum(1 for f in os.listdir(folder) if f.endswith('.txt')) if os.path.exists(folder) else 0
