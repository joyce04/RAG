"""
analyst.py
----------
The Patient Cohort Analyst agent.

Uses the `sql_coder` LLM to generate a DuckDB SQL query that estimates
how many real patients in the MIMIC-III database would qualify for the
trial criteria, providing a programmatic feasibility signal.

Uses a factory pattern so `knowledge_stores` and `llms` are injected
rather than accessed as globals.
"""

import duckdb
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph.states import TeamState, AgentOutput


def make_analyst(knowledge_stores: dict, llms: dict) -> Callable:
    """
    Factory that binds `knowledge_stores` (for the DB path) and `llms`
    (for the sql_coder model), and returns a patient_cohort_analyst
    function with signature:

        patient_cohort_analyst(task_description, state) -> AgentOutput
    """
    sql_coder_llm = llms['sql_coder']

    def patient_cohort_analyst(task_description: str, state: TeamState) -> AgentOutput:
        """
        Estimate the number of eligible patients for the described criteria.

        Steps:
          1. If the SOP disables the SQL analyst, return a skipped notice.
          2. Read the DB schema to give the LLM accurate column names.
          3. Generate a COUNT(*) SQL query via the sql_coder LLM.
          4. Execute the query against DuckDB and return the count.

        Key MIMIC item IDs used:
          50912 = creatinine  (ITEMID proxy for renal impairment: 1.5–3.0)
          50852 = HbA1c       (ITEMID proxy for uncontrolled T2DM: > 8.0)
          ICD9  = '25000'     (Type 2 Diabetes Mellitus)
        """
        if not state['sop'].use_sql_analyst:
            return AgentOutput(
                agent_name="Patient Cohort Analyst",
                findings="Analysis skipped as per SOP.",
            )

        db_path = knowledge_stores['mimic_db_path']

        # Step 1: pull the schema so the LLM knows exact column names
        con = duckdb.connect(db_path)
        schema = con.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, column_name
        """).df()
        con.close()

        # Step 2: build the SQL generation chain
        sql_generation_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"You are an expert SQL writer specialising in DuckDB. "
                f"The database contains MIMIC-III patient data with this schema:\n"
                f"{schema.to_string()}\n\n"
                f"IMPORTANT: all column names in your query MUST be uppercase "
                f"(e.g. SELECT SUBJECT_ID, ICD9_CODE ...).\n\n"
                f"Key mappings:\n"
                f"  - Type 2 Diabetes (T2DM)         → ICD9_CODE = '25000'\n"
                f"  - Creatinine (renal impairment)  → ITEMID 50912, VALUENUM 1.5–3.0\n"
                f"  - HbA1c (uncontrolled T2DM)      → ITEMID 50852, VALUENUM > 8.0",
            ),
            (
                "human",
                "Write a SQL query to count the number of unique patients "
                "who meet the following criteria: {task}",
            ),
        ])

        sql_chain = sql_generation_prompt | sql_coder_llm | StrOutputParser()

        print(f"[Analyst] Generating SQL for: {task_description}")
        raw_sql = sql_chain.invoke({"task": task_description})

        # Strip markdown code fences if the LLM wrapped the query
        sql_query = raw_sql.strip().replace("```sql", "").replace("```", "").strip()
        print(f"[Analyst] Generated SQL:\n{sql_query}")

        # Step 3: execute the query and extract the patient count
        try:
            con = duckdb.connect(db_path)
            result = con.execute(sql_query).fetchone()
            patient_count = result[0] if result else 0
            con.close()

            findings = (
                f"Generated SQL Query:\n{sql_query}\n\n"
                f"Estimated eligible patient count from the database: {patient_count}."
            )
            print(f"[Analyst] Estimated eligible patients: {patient_count}")

        except Exception as e:
            findings = f"Error executing SQL query: {e}. Defaulting to a count of 0."
            print(f"[Analyst] Query execution error: {e}")

        return AgentOutput(agent_name="Patient Cohort Analyst", findings=findings)

    return patient_cohort_analyst
