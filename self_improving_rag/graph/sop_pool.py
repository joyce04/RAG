"""
sop_pool.py
-----------
The SOP Gene Pool: an in-memory store that accumulates every TeamSOP
variant tried during the evolution loop, together with its evaluation
scores and parent lineage.

Used by `main.py` to:
  - track the version history of SOPs
  - select the latest entry as the parent for the next mutation cycle
  - enumerate all entries for Pareto-front analysis
"""

from typing import List, Dict, Any, Optional

from graph.teamsop import TeamSOP
from graph.evaluator import EvaluationResult


class SOPGenePool:
    """
    Append-only store of (sop, evaluation, parent_version) entries.

    Version numbers are auto-incremented starting from 1.
    The baseline SOP added in main.py is always version 1 with parent=None.
    """

    def __init__(self):
        self.pool: List[Dict[str, Any]] = []
        self.version_counter: int = 0

    def add(
        self,
        sop: TeamSOP,
        eval_result: EvaluationResult,
        parent_version: Optional[int] = None,
    ) -> None:
        """
        Append a new entry to the pool.

        Parameters
        ----------
        sop            : the TeamSOP that was evaluated
        eval_result    : its five-dimensional evaluation scores
        parent_version : version number of the SOP this was mutated from
                         (None for the baseline)
        """
        self.version_counter += 1
        entry = {
            "version":    self.version_counter,
            "sop":        sop,
            "evaluation": eval_result,
            "parent":     parent_version,
        }
        self.pool.append(entry)
        print(f"[GenePool] Added SOP v{self.version_counter} (parent: {parent_version}).")

    def get_latest_entry(self) -> Optional[Dict[str, Any]]:
        """Return the most recently added entry, or None if the pool is empty."""
        return self.pool[-1] if self.pool else None
