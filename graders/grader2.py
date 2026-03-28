"""
Grader 2: Experiment Replication scorer
Scores 0.0-1.0. 100% deterministic numeric comparison. No pydantic dependency.

  AUC within +-0.01 -> 0.45 pts
  AUC within +-0.03 -> 0.25 pts
  F1  within +-0.01 -> 0.35 pts
  F1  within +-0.03 -> 0.20 pts
  Interpretation keyword match -> up to 0.20 pts
"""
from __future__ import annotations

_INTERPRETATION_KEYWORDS = [
    "imbalance", "imbalanced", "class weight", "stratif", "weighted",
    "recall", "precision", "minority", "balanced", "skewed",
    "unequal classes", "class distribution",
]


def grade_results(payload, ground_truth: dict) -> float:
    """
    payload: object with .auc, .f1, .interpretation
    ground_truth: dict with 'auc', 'f1'
    """
    gt_auc = float(ground_truth.get("auc", 0.0))
    gt_f1  = float(ground_truth.get("f1",  0.0))

    auc  = float(payload.auc)
    f1   = float(payload.f1)
    interp = (payload.interpretation or "").lower()

    score = 0.0

    auc_diff = abs(auc - gt_auc)
    if auc_diff <= 0.01:
        score += 0.45
    elif auc_diff <= 0.03:
        score += 0.25

    f1_diff = abs(f1 - gt_f1)
    if f1_diff <= 0.01:
        score += 0.35
    elif f1_diff <= 0.03:
        score += 0.20

    hits = sum(1 for kw in _INTERPRETATION_KEYWORDS if kw in interp)
    if hits >= 2:
        score += 0.20
    elif hits == 1:
        score += 0.10

    return round(max(0.0, min(1.0, score)), 4)
