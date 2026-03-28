"""
Grader 1: Methodology Audit scorer
Scores 0.0-1.0. 100% deterministic — no LLM, no pydantic dependency.

Scoring:
  Each correctly identified flaw = 0.25
  Partial credit (right taxonomy, wrong location) = 0.10
  False positives: -0.05 each, capped at -0.20
"""
from __future__ import annotations

_FLAW_SYNONYMS: dict[str, list[str]] = {
    "wrong_statistical_test": [
        "wrong test", "incorrect test", "t-test", "t test", "chi-square",
        "chi square", "anova", "parametric", "non-parametric",
        "wrong statistical", "inappropriate test", "statistical test",
    ],
    "underpowered_sample": [
        "sample size", "underpowered", "small n", "insufficient participants",
        "low power", "power analysis", "n too small", "power",
    ],
    "undisclosed_exclusion": [
        "exclusion", "excluded", "outlier", "removed", "undisclosed",
        "missing data", "data exclusion", "selective", "missing participants",
    ],
    "p_value_manipulation": [
        "p-value", "p value", "fishing", "hacking", "selective reporting",
        "multiple comparison", "bonferroni", "significance threshold",
        "multiple outcomes", "multiple testing",
    ],
    "class_imbalance_ignored": [
        "class imbalance", "imbalanced", "stratif", "oversampl", "undersampl",
        "weighted", "recall", "precision",
    ],
}


def _type_matches(submitted_type: str, taxonomy: str) -> bool:
    s = submitted_type.lower().strip()
    return any(syn in s for syn in _FLAW_SYNONYMS.get(taxonomy, [taxonomy.replace("_", " ")]))


def _location_matches(submitted_loc: str, gt_location: str) -> bool:
    s = submitted_loc.lower().strip()
    g = gt_location.lower().strip()
    return g in s or s in g


def grade_audit(payload, ground_truth: dict) -> float:
    """
    payload: any object with .flaws list; each flaw has .flaw_type and .location
    ground_truth: dict with 'flaws' list of {id, taxonomy, location}
    """
    gt_flaws  = ground_truth.get("flaws", [])
    submitted = payload.flaws

    matched_gt_ids: set[str] = set()
    fp_count = 0
    score    = 0.0

    for report in submitted:
        best_id    = None
        best_score = 0.0

        for gt in gt_flaws:
            gt_id = gt["id"]
            if gt_id in matched_gt_ids:
                continue
            type_ok = _type_matches(report.flaw_type, gt["taxonomy"])
            loc_ok  = _location_matches(report.location, gt.get("location", ""))
            if type_ok and loc_ok:
                best_id    = gt_id
                best_score = 0.25
                break
            elif type_ok and best_score < 0.10:
                best_id    = gt_id
                best_score = 0.10

        if best_id:
            score += best_score
            matched_gt_ids.add(best_id)
        else:
            fp_count += 1

    score -= min(fp_count * 0.05, 0.20)
    return round(max(0.0, min(1.0, score)), 4)
