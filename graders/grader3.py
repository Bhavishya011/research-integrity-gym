"""
Grader 3: Claim Verification scorer
Scores 0.0-1.0. 100% deterministic — zero LLM, zero subjectivity.

  Correct verdict ("invalid")          0.35
  Effect size close to true value      0.20
  p-value correctly > 0.05             0.15
  Flagged undisclosed exclusion        0.20
  Justification keyword check          0.10
"""
from __future__ import annotations

_EXCLUSION_KEYWORDS = [
    "exclusion", "excluded", "outlier", "removed", "undisclosed",
    "selective", "missing", "n=", "fewer participants", "hidden removal",
]

# Minimum requirements to prevent gaming
_MIN_JUSTIFICATION_WORDS = 20
_MIN_UNIQUE_WORDS_RATIO = 0.4  # At least 40% unique words


def _is_coherent_justification(justif: str) -> bool:
    """Check if justification has basic coherence (not keyword stuffing)."""
    words = justif.lower().split()
    if len(words) < _MIN_JUSTIFICATION_WORDS:
        return False
    unique_words = set(words)
    unique_ratio = len(unique_words) / len(words)
    return unique_ratio >= _MIN_UNIQUE_WORDS_RATIO


def grade_verdict(payload, ground_truth: dict) -> float:
    """
    payload: object with .verdict.value, .effect_size, .p_value, .justification
    ground_truth: dict from task3_claim_verify.generate_episode()
    """
    true_verdict  = ground_truth.get("true_verdict", "invalid")
    true_effect   = float(ground_truth.get("true_effect", 0.0))
    effect_tol    = float(ground_truth.get("effect_tolerance", 0.5))
    p_threshold   = float(ground_truth.get("p_threshold", 0.05))
    has_exclusion = bool(ground_truth.get("has_undisclosed_exclusion", True))
    req_keywords  = ground_truth.get("required_keywords", [])
    req_count     = int(ground_truth.get("required_keyword_count", 2))

    verdict = payload.verdict.value
    effect  = float(payload.effect_size)
    p_val   = float(payload.p_value)
    justif  = (payload.justification or "").lower()

    score = 0.0

    # 1. Verdict (0.35)
    if verdict == true_verdict:
        score += 0.35
    elif verdict == "partially_valid" and true_verdict == "invalid":
        score += 0.15

    # 2. Effect size (0.20)
    if abs(effect - true_effect) <= effect_tol:
        score += 0.20
    elif abs(effect - true_effect) <= effect_tol * 2:
        score += 0.10

    # 3. p-value direction (0.15)
    if p_val > p_threshold:
        score += 0.15
    elif p_val > p_threshold * 0.8:
        score += 0.05

    # 4. Exclusion detection (0.20) — keyword check WITH coherence requirement
    # Must have coherent justification (prevents keyword stuffing)
    if has_exclusion and _is_coherent_justification(justif):
        if any(kw in justif for kw in _EXCLUSION_KEYWORDS):
            score += 0.20

    # 5. Justification keyword quality (0.10) — only if coherent
    if _is_coherent_justification(justif):
        hits = sum(1 for kw in req_keywords if kw in justif)
        if hits >= req_count:
            score += 0.10
        elif hits == 1:
            score += 0.05

    # Clamp to (0.0001, 0.9999) - judges require strictly between 0 and 1
    score = max(0.0001, min(0.9999, score))
    return round(score, 4)
