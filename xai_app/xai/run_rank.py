# xai_app/evaluation/run_rank.py

import pandas as pd
import numpy as np
from pathlib import Path


# -------------------------
# Metric → Aspect mappings
# -------------------------
HIGHER_BETTER = {
    "faithfulnesscorrelation", "faithfulnessestimate", "monotonicitycorrelation",
    "insertion", "irof", "sufficiency", "maxsensitivity", "sparseness",
    "dice", "iou", "precision_at_k", "recall_at_k", "pointing_game", "api",
}

FAITHFULNESS = {
    "faithfulnesscorrelation", "faithfulnessestimate", "monotonicitycorrelation",
    "pixelflipping", "regionperturbation", "insertion", "deletion", "irof", "road",
    "sufficiency", "infidelity"
}
ROBUSTNESS = {
    "locallipschitzestimate", "maxsensitivity", "continuity", "relativeinputstability",
    "relativeoutputstability", "relativerepresentationstability"
}
COMPLEXITY = {"sparseness", "complexity", "effectivecomplexity"}
PLAUSIBILITY = {"dice", "iou", "precision_at_k", "recall_at_k", "pointing_game", "api"}


def metric_to_aspect(m):
    if m in FAITHFULNESS: return "Faithfulness"
    if m in ROBUSTNESS: return "Robustness"
    if m in COMPLEXITY: return "Complexity"
    if m in PLAUSIBILITY: return "Plausibility"
    return None


# ================================================================
# 🔹 MAIN FUNCTION: compute rankings from summary CSV
# ================================================================
def compute_rankings(summary_csv: str, save_csv: str = "xai_ranking_summary.csv"):
    """
    summary_csv should be the output from run_eval_multi:
        columns = Method, Metric, Value_mean, Value_std, Value_median
    """

    summary = pd.read_csv(summary_csv)

    # Add aspect column
    summary["Aspect"] = summary["Metric"].apply(metric_to_aspect)

    # -------------------------
    # 1) Rank for each metric
    # -------------------------
    ranked_list = []
    for metric, sub in summary.groupby("Metric"):
        sub = sub.copy()
        ascending = metric not in HIGHER_BETTER
        sub["Rank"] = sub["Value_mean"].rank(ascending=ascending, method="min")
        ranked_list.append(sub)

    rank_df = pd.concat(ranked_list, ignore_index=True)

    # -------------------------
    # 2) Aggregate per aspect
    # -------------------------
    aspect_rank = (
        rank_df.groupby(["Method", "Aspect"])["Rank"]
        .agg(["mean", "median", "std"])
        .reset_index()
    )

    aspect_wide = aspect_rank.pivot(index="Method", columns="Aspect", values=["mean", "median", "std"])
    aspect_wide.columns = [f"{stat}_{aspect}" for stat, aspect in aspect_wide.columns]
    aspect_wide = aspect_wide.reset_index()

    # -------------------------
    # 3) All metrics table
    # -------------------------
    metric_wide = rank_df.pivot_table(
        index="Method", columns="Metric", values="Rank", aggfunc="mean"
    )

    # -------------------------
    # 4) Global mean rank
    # -------------------------
    global_mean = metric_wide.mean(axis=1).rename("MeanRank_AllMetrics")

    # -------------------------
    # 5) Merge everything
    # -------------------------
    final = (
        metric_wide
        .merge(aspect_wide, on="Method", how="left")
        .merge(global_mean, on="Method", how="left")
        .reset_index()
    )

    # -------------------------
    # 6) Save
    # -------------------------
    final.to_csv(save_csv, index=False)
    print(f"[INFO] Saved ranking summary → {save_csv}")

    return final
