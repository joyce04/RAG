"""
process_mimic.py
----------------
Loads MIMIC-III CSV files from disk into a DuckDB database.

The DuckDB file is written to data/mimic/mimic.db and is read at query
time by the Patient Cohort Analyst agent (graph/analyst.py).

Expected CSV files (gzipped) under data/mimic/:
  PATIENTS.csv.gz, DIAGNOSES_ICD.csv.gz, PROCEDURES_ICD.csv.gz,
  PRESCRIPTIONS.csv.gz, LABEVENTS.csv.gz
"""

import os
import duckdb

from data.download_raw_data import data_paths


def load_real_mimic_data() -> str | None:
    """
    Read MIMIC-III CSV files and populate a DuckDB database.

    Returns the path to the created .db file, or None if any required
    CSV is missing.

    Tables created:
      - patients        (SUBJECT_ID, GENDER, DOB, DOD)
      - diagnoses_icd   (SUBJECT_ID, ICD9_CODE)
      - procedures      (SUBJECT_ID, ICD9_CODE)
      - prescriptions   (SUBJECT_ID, DRUG)
      - labevents       (SUBJECT_ID, ITEMID int, VALUENUM float)
                        filtered to creatinine (50912) and HbA1c (50852)
    """
    db_path = os.path.join(data_paths['mimic'], 'mimic.db')

    # CSV files live directly in the mimic directory
    csv_dir = data_paths['mimic']

    required_files = {
        'patients':      os.path.join(csv_dir, 'PATIENTS.csv.gz'),
        'diagnoses':     os.path.join(csv_dir, 'DIAGNOSES_ICD.csv.gz'),
        'procedures':    os.path.join(csv_dir, 'PROCEDURES_ICD.csv.gz'),
        'prescriptions': os.path.join(csv_dir, 'PRESCRIPTIONS.csv.gz'),
        'labevents':     os.path.join(csv_dir, 'LABEVENTS.csv.gz'),
    }

    # Abort early if any file is missing
    for name, path in required_files.items():
        if not os.path.exists(path):
            print(f"Missing required MIMIC file '{name}': {path}")
            return None

    # Always rebuild from scratch to keep the DB in sync with the CSVs
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)

    # Each CREATE TABLE reads the gzipped CSV directly via DuckDB's
    # read_csv_auto — no intermediate unzip step needed.
    con.execute(
        f"CREATE TABLE patients AS "
        f"SELECT SUBJECT_ID, GENDER, DOB, DOD "
        f"FROM read_csv_auto('{required_files['patients']}')"
    )

    con.execute(
        f"CREATE TABLE diagnoses_icd AS "
        f"SELECT SUBJECT_ID, ICD9_CODE "
        f"FROM read_csv_auto('{required_files['diagnoses']}')"
    )

    con.execute(
        f"CREATE TABLE procedures AS "
        f"SELECT SUBJECT_ID, ICD9_CODE "
        f"FROM read_csv_auto('{required_files['procedures']}')"
    )

    con.execute(
        f"CREATE TABLE prescriptions AS "
        f"SELECT SUBJECT_ID, DRUG "
        f"FROM read_csv_auto('{required_files['prescriptions']}')"
    )

    # labevents is large; load only the two ITEMIDs used by the analyst:
    #   50912 = creatinine  (proxy for renal impairment)
    #   50852 = HbA1c       (proxy for uncontrolled diabetes)
    # VALUENUM is stored as varchar in some MIMIC versions, so we filter
    # for numeric strings and cast explicitly.
    con.execute(f"""
        CREATE TABLE labevents_staging AS
        SELECT SUBJECT_ID, ITEMID, VALUENUM
        FROM read_csv_auto('{required_files['labevents']}', all_varchar=True)
        WHERE ITEMID IN ('50912', '50852')
          AND VALUENUM IS NOT NULL
          AND VALUENUM ~ '^[0-9]+(\\.[0-9]+)?$'
    """)

    con.execute("""
        CREATE TABLE labevents AS
        SELECT
            SUBJECT_ID,
            CAST(ITEMID   AS INTEGER) AS ITEMID,
            CAST(VALUENUM AS DOUBLE)  AS VALUENUM
        FROM labevents_staging
    """)
    con.execute("DROP TABLE labevents_staging")

    con.close()
    print(f"MIMIC-III database created at: {db_path}")
    return db_path
