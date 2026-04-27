"""
display.py
----------
Visualisation utilities for the SOP Gene Pool's Pareto front.

Two plots are produced side-by-side:
  1. 2D scatter  — Rigor (x) vs Feasibility (y), one point per SOP version
  2. Parallel coordinates — all five evaluation dimensions at once,
     showing trade-off curves across Pareto-optimal SOPs
"""

import matplotlib.pyplot as plt
import pandas as pd


def visualize_frontier(pareto_sops: list) -> None:
    """
    Render a Rigor-vs-Feasibility scatter and a 5D parallel-coordinates
    plot for the SOPs on the Pareto front.

    Parameters
    ----------
    pareto_sops : list of gene-pool entries (dicts with 'version' and
                  'evaluation' keys) that are non-dominated solutions.
    """
    if not pareto_sops:
        print("No SOPs on the Pareto front to visualise.")
        return

    # ------------------------------------------------------------------
    # 1. 2D scatter: Rigor vs Feasibility
    # ------------------------------------------------------------------
    labels            = [f"v{s['version']}" for s in pareto_sops]
    rigor_scores      = [s['evaluation'].rigor.score       for s in pareto_sops]
    feasibility_scores= [s['evaluation'].feasibility.score for s in pareto_sops]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(rigor_scores, feasibility_scores, s=150, alpha=0.7)

    # Annotate each point with its SOP version label
    for i, label in enumerate(labels):
        ax1.annotate(
            label,
            (rigor_scores[i], feasibility_scores[i]),
            xytext=(10, -10),
            textcoords='offset points',
            fontsize=12,
        )

    ax1.set_title('Pareto Frontier: Rigor vs. Feasibility', fontsize=14)
    ax1.set_xlabel('Scientific Rigor Score',       fontsize=12)
    ax1.set_ylabel('Recruitment Feasibility Score', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Give a small margin so points are not clipped at the axes
    ax1.set_xlim(min(rigor_scores)       - 0.05, max(rigor_scores)       + 0.05)
    ax1.set_ylim(min(feasibility_scores) - 0.10, max(feasibility_scores) + 0.10)

    # ------------------------------------------------------------------
    # 2. Parallel coordinates: all five dimensions
    # ------------------------------------------------------------------
    data = []
    for s in pareto_sops:
        eval_dict = s['evaluation'].model_dump()
        # Flatten to {dimension: score} for plotting
        scores = {k: v['score'] for k, v in eval_dict.items()}
        scores['SOP Version'] = f"v{s['version']}"
        data.append(scores)

    df = pd.DataFrame(data)
    pd.plotting.parallel_coordinates(
        df,
        'SOP Version',
        colormap=plt.get_cmap("viridis"),
        ax=ax2,
    )

    ax2.set_title('5D Performance Trade-offs on Pareto Front', fontsize=14)
    ax2.set_ylabel('Normalised Score', fontsize=12)
    ax2.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
