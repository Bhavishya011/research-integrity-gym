"""
Grader 1: Methodology Audit scorer
Scores 0.0-1.0. 100% deterministic — no LLM, no pydantic dependency.

Scoring:
  Each correctly identified flaw = 0.25
  Partial credit (right taxonomy, wrong location) = 0.10
  False positives: -0.05 each, capped at -0.20
"""
from __future__ import annotations

# Comprehensive synonym list for fair matching
# Agents may phrase flaws in many valid ways — we accept all reasonable variants
# Primary keys use PeerGuard medical taxonomy; synonym lists include legacy terms
_FLAW_SYNONYMS: dict[str, list[str]] = {
    "unblinded_investigator_bias": [
        # Direct matches — CONSORT blinding violations
        "unblinded", "investigator bias", "detection bias", "observer bias",
        "assessor bias", "blinding", "single-blind", "open-label",
        "allocation concealment", "outcome assessor", "blinded assessor",
        # Legacy: wrong_statistical_test synonyms (backward compat)
        "wrong test", "incorrect test", "inappropriate test", "wrong statistical",
        "statistical test", "misapplied test", "misused test",
        "t-test", "t test", "ttest", "chi-square", "chi square", "chisquare",
        "chi-sq", "anova", "mann-whitney", "mann whitney", "wilcoxon",
        "pearson", "spearman", "kruskal", "fisher exact",
        "parametric", "non-parametric", "nonparametric", "normality assumption",
        "distributional assumption", "test assumption", "violated assumption",
        "inappropriate statistical", "inappropriate method", "incorrect method",
        "wrong method", "invalid test", "unsuitable test", "mismatched test",
        "test mismatch", "analysis mismatch", "wrong analysis",
        "inappropriate analysis", "incorrect analysis", "flawed analysis",
        "continuous outcome", "categorical outcome", "ordinal", "binary outcome",
        "data type mismatch", "variable type",
    ],
    "insufficient_power_analysis": [
        # Direct matches — CONSORT power/sample size
        "insufficient power", "power analysis", "sample size calculation",
        "ICH-GCP", "CONSORT Item 7",
        # Legacy: underpowered_sample synonyms
        "sample size", "underpowered", "small n", "small sample",
        "insufficient sample", "insufficient participants", "insufficient subjects",
        "low power", "n too small", "power",
        "inadequate sample", "limited sample", "small group", "few participants",
        "few subjects", "small cohort", "inadequate n", "low n",
        "underpowered study", "insufficient n", "sample too small",
        "not enough participants", "not enough subjects", "limited participants",
        "statistical power", "power calculation", "effect size", "detectable effect",
        "type ii error", "type 2 error", "beta error", "false negative rate",
        "n=", "n =", "per group", "per arm", "each group", "each arm",
    ],
    "protocol_deviation_unreported": [
        # Direct matches — CONSORT protocol deviation
        "protocol deviation", "unreported deviation", "protocol violation",
        "CONSORT flow", "flow diagram", "per-protocol", "ITT",
        # Legacy: undisclosed_exclusion synonyms
        "exclusion", "excluded", "outlier", "removed", "undisclosed",
        "missing data", "data exclusion", "selective", "missing participants",
        "unreported exclusion", "hidden exclusion", "silent exclusion",
        "participant removal", "data removal", "unexplained removal",
        "participants dropped", "dropped participants", "attrition",
        "lost to follow", "dropout", "drop out", "drop-out", "withdrew",
        "withdrawal", "discontinued", "did not complete",
        "discrepancy", "mismatch", "inconsistent", "doesn't match",
        "does not match", "differ", "different n", "reported n",
        "recruited vs", "enrolled vs", "analyzed vs", "analysed vs",
        "missing", "unaccounted", "not reported", "unclear how many",
    ],
    "endpoint_switching": [
        # Direct matches — CONSORT endpoint switching
        "endpoint switching", "outcome switching", "primary endpoint",
        "pre-registered endpoint", "post-hoc primary", "endpoint change",
        # Legacy: p_value_manipulation synonyms
        "p-value", "p value", "pvalue", "p-hacking", "p hacking", "phacking",
        "fishing", "hacking", "selective reporting", "cherry pick",
        "multiple comparison", "multiple comparisons", "bonferroni",
        "multiple testing", "multiplicity", "family-wise", "familywise",
        "correction", "adjustment", "adjusted p", "uncorrected",
        "multiple outcomes", "multiple endpoints", "secondary outcome",
        "primary outcome", "outcome switching",
        "significance threshold", "alpha level", "type i error", "type 1 error",
        "false positive", "significant result", "selective", "selectively reported",
        "only significant", "only reported significant", "unreported outcomes",
        "file drawer", "publication bias",
        "significance hunting", "data dredging", "outcome fishing",
        "harking", "hypothesizing after", "post-hoc", "post hoc", "posthoc",
    ],
    "class_imbalance_ignored": [
        # Direct matches
        "class imbalance", "imbalanced", "imbalance", "unbalanced",
        "stratif", "stratified", "stratification",
        # Techniques
        "oversampl", "undersampl", "smote", "weighted", "class weight",
        "balanced", "resampl",
        # Metrics affected
        "recall", "precision", "f1", "f-score", "sensitivity", "specificity",
        "auc", "roc", "accuracy paradox",
        # Phrasing
        "skewed classes", "majority class", "minority class", "rare class",
        "unequal distribution", "prevalence", "base rate",
    ],
}


def _type_matches(submitted_type: str, taxonomy: str) -> bool:
    s = submitted_type.lower().strip()
    # Exact match (with or without underscores)
    if s == taxonomy.lower() or s == taxonomy.lower().replace("_", " "):
        return True
    # Also check underscored version of submitted text
    s_normalized = s.replace("_", " ")
    synonyms = _FLAW_SYNONYMS.get(taxonomy, [taxonomy.replace("_", " ")])
    return any(syn in s for syn in synonyms) or any(syn in s_normalized for syn in synonyms)


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
    # Clamp to (0.0001, 0.9999) - judges require strictly between 0 and 1
    score = max(0.0001, min(0.9999, score))
    return round(score, 4)
