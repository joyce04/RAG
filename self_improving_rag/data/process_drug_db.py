"""
process_drug_db.py
------------------
Loads DrugBank Open vocabulary and DDI corpus into a DuckDB database
for the Interaction DB Analyst agent.

Returns (drugbank_db_path,) — a single DB containing both datasets.
Falls back gracefully if source files are absent.
"""

import os
import duckdb

from data.download_raw_data import data_paths


def load_drug_databases() -> str | None:
    """
    Build a DuckDB database from DrugBank vocabulary and DDI corpus files.

    Tables created:
      - drug_vocab       (drug_name, drug_class, mechanism, targets)
      - drug_interactions (drug_a, drug_b, severity, mechanism_text)

    Returns the path to the .db file, or None if no source files exist.
    """
    db_path = os.path.join(data_paths['drugbank_db'], 'drugs.db')
    os.makedirs(data_paths['drugbank_db'], exist_ok=True)

    # Rebuild if the DB doesn't exist yet
    if os.path.exists(db_path):
        print(f"Drug database already exists at {db_path}")
        return db_path

    con = duckdb.connect(db_path)

    # ------------------------------------------------------------------
    # Table 1: Drug vocabulary from data/drugbank/*.txt
    # ------------------------------------------------------------------
    drugbank_dir = data_paths['drugbank']
    vocab_rows = []

    if os.path.exists(drugbank_dir):
        for fname in os.listdir(drugbank_dir):
            if not fname.endswith('.txt'):
                continue
            with open(os.path.join(drugbank_dir, fname)) as f:
                content = f.read()
            for block in content.strip().split('\n\n'):
                if not block.strip():
                    continue
                row = _parse_drug_block(block)
                if row:
                    vocab_rows.append(row)

    if vocab_rows:
        con.execute("""
            CREATE TABLE drug_vocab (
                drug_name   VARCHAR,
                drug_class  VARCHAR,
                mechanism   VARCHAR,
                targets     VARCHAR,
                half_life   VARCHAR,
                clearance   VARCHAR
            )
        """)
        con.executemany(
            "INSERT INTO drug_vocab VALUES (?, ?, ?, ?, ?, ?)",
            vocab_rows,
        )
        print(f"[DrugDB] Loaded {len(vocab_rows)} drug vocab entries.")
    else:
        con.execute("""
            CREATE TABLE drug_vocab (
                drug_name   VARCHAR,
                drug_class  VARCHAR,
                mechanism   VARCHAR,
                targets     VARCHAR,
                half_life   VARCHAR,
                clearance   VARCHAR
            )
        """)
        print("[DrugDB] No drug vocab entries found — drug_vocab table is empty.")

    # ------------------------------------------------------------------
    # Table 2: DDI interactions from data/ddi/*.txt
    # ------------------------------------------------------------------
    ddi_dir = data_paths['ddi']
    ddi_rows = []

    if os.path.exists(ddi_dir):
        for fname in os.listdir(ddi_dir):
            if not fname.endswith('.txt'):
                continue
            with open(os.path.join(ddi_dir, fname)) as f:
                content = f.read()
            ddi_rows.extend(_parse_ddi_blocks(content))

    if ddi_rows:
        con.execute("""
            CREATE TABLE drug_interactions (
                drug_a          VARCHAR,
                drug_b          VARCHAR,
                severity        VARCHAR,
                mechanism_text  VARCHAR
            )
        """)
        con.executemany(
            "INSERT INTO drug_interactions VALUES (?, ?, ?, ?)",
            ddi_rows,
        )
        print(f"[DrugDB] Loaded {len(ddi_rows)} DDI entries.")
    else:
        con.execute("""
            CREATE TABLE drug_interactions (
                drug_a          VARCHAR,
                drug_b          VARCHAR,
                severity        VARCHAR,
                mechanism_text  VARCHAR
            )
        """)
        print("[DrugDB] No DDI entries found — drug_interactions table is empty.")

    con.close()
    print(f"Drug database created at: {db_path}")
    return db_path


def _parse_drug_block(block: str) -> tuple | None:
    """Parse a 'Drug: X | Class: Y | ...' line into a 6-tuple."""
    try:
        parts = {
            k.strip(): v.strip()
            for segment in block.split('|')
            for k, _, v in [segment.partition(':')]
            if _
        }
        return (
            parts.get('Drug', ''),
            parts.get('Class', ''),
            parts.get('Mechanism', ''),
            parts.get('Targets', ''),
            parts.get('Half-life', ''),
            parts.get('Clearance', ''),
        )
    except Exception:
        return None


def _parse_ddi_blocks(content: str) -> list:
    """
    Parse the fallback DDI corpus format into (drug_a, drug_b, severity, mechanism).
    Looks for blocks starting with drug pair lines followed by Severity: and Mechanism:.
    """
    rows = []
    blocks = content.split('\n\n')
    for block in blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        # First line: "DrugA + DrugB"
        if '+' not in lines[0]:
            continue
        try:
            parts = lines[0].split('+')
            drug_a = parts[0].strip()
            drug_b = parts[1].split('(')[0].strip()
            severity = ''
            mechanism = ''
            for line in lines[1:]:
                if line.startswith('Severity:'):
                    severity = line.replace('Severity:', '').strip()
                elif line.startswith('Mechanism:'):
                    mechanism = line.replace('Mechanism:', '').strip()
            if drug_a and drug_b and severity:
                rows.append((drug_a, drug_b, severity, mechanism))
        except Exception:
            continue
    return rows
