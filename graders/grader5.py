"""
Grader 5: FDA Approval Verdict scorer
Scores 0.0-1.0. 100% deterministic — no LLM, no subjectivity.

Evaluates the submit_fda_verdict action by checking:
  1. Correct verdict (REJECT)                    0.20
  2. Protocol violations flagged (from T1)        0.20
  3. Class imbalance / adverse events (from T2)   0.20
  4. Deleted patients / exclusion (from T3)       0.20
  5. Citation fabrication caught (from T4)         0.20

Total: 1.0 max.

Scoring is based on:
  - The agent's decision and justification_flags (from the payload)
  - The agent's episode history (flags_raised, code_calls from EpisodeState)
  - The combined ground truth from all sub-tasks
"""
from __future__ import annotations

from env.state import EpisodeState


# ---------------------------------------------------------------------------
# Keywords the grader looks for in justification_flags and flags_raised
# ---------------------------------------------------------------------------

# T1: CONSORT protocol violations
_T1_KEYWORDS = [
    "unblinded", "investigator bias", "detection bias", "blinding",
    "power analysis", "underpowered", "sample size", "insufficient power",
    "protocol deviation", "exclusion", "excluded", "CONSORT",
    "endpoint switching", "outcome switching", "primary endpoint",
    "multiple comparison", "selective reporting",
]

# T2: Class imbalance / adverse events
_T2_KEYWORDS = [
    "class imbalance", "imbalanced", "imbalance", "adverse event",
    "cardiovascular", "readmission", "stratif", "weighted",
    "balanced", "minority class", "oversampl", "undersampl",
]

# T3: Deleted / excluded patients
_T3_KEYWORDS = [
    "deleted patient", "excluded patient", "missing patient",
    "silently excluded", "undisclosed exclusion", "removed patient",
    "outlier removal", "data manipulation", "n mismatch",
    "fewer participants", "exclusion", "protocol deviation",
    "tumor", "efficacy", "p-value", "not significant",
]

# T4: Citation fabrication
_T4_KEYWORDS = [
    "citation fabricat", "fabricated", "misrepresent", "directional",
    "teratogenic", "contraindicated", "pregnancy", "safe vs harmful",
    "citation mismatch", "source contradiction", "wrong direction",
    "malformation",
]


def _count_keyword_hits(text_sources: list[str], keywords: list[str]) -> int:
    """Count how many distinct keywords appear in any of the text sources."""
    combined = " ".join(s.lower() for s in text_sources)
    return sum(1 for kw in keywords if kw.lower() in combined)


def grade_fda_verdict(
    payload,
    ground_truth: dict,
    episode_state: EpisodeState,
) -> float:
    """
    Grade the Task 5 FDA verdict submission.

    Args:
        payload: SubmitFDAVerdictPayload with .decision.value and .justification_flags
        ground_truth: combined truth dict with t1_truth..t4_truth + expected_verdict
        episode_state: full EpisodeState with flags_raised, code_calls, etc.

    Returns:
        float score in [0.0001, 0.9999]
    """
    score = 0.0

    decision = payload.decision.value.upper()
    flags = payload.justification_flags or []
    expected = ground_truth.get("expected_verdict", "REJECT").upper()

    # Collect all text the agent produced during the episode for keyword matching
    flag_texts = []
    for f in episode_state.flags_raised:
        flag_texts.append(f.get("flaw_type", ""))
        flag_texts.append(f.get("description", ""))
        flag_texts.append(f.get("fabrication_type", ""))
        flag_texts.append(f.get("evidence", ""))
    flag_texts.extend(flags)

    # ------------------------------------------------------------------
    # 1. Correct verdict (0.20)
    # ------------------------------------------------------------------
    if decision == expected:
        score += 0.20

    # ------------------------------------------------------------------
    # 2. Protocol violations flagged — Task 1 (0.20)
    # ------------------------------------------------------------------
    t1_hits = _count_keyword_hits(flag_texts, _T1_KEYWORDS)
    if t1_hits >= 3:
        score += 0.20
    elif t1_hits >= 2:
        score += 0.15
    elif t1_hits >= 1:
        score += 0.08

    # ------------------------------------------------------------------
    # 3. Class imbalance / adverse events — Task 2 (0.20)
    # ------------------------------------------------------------------
    # Agent must have executed code on the dataset AND flagged imbalance
    agent_ran_code = episode_state.code_calls >= 1
    t2_hits = _count_keyword_hits(flag_texts, _T2_KEYWORDS)

    if agent_ran_code and t2_hits >= 2:
        score += 0.20
    elif agent_ran_code and t2_hits >= 1:
        score += 0.12
    elif t2_hits >= 1:
        score += 0.05  # Mentioned but didn't verify with code

    # ------------------------------------------------------------------
    # 4. Deleted patients / exclusion — Task 3 (0.20)
    # ------------------------------------------------------------------
    t3_hits = _count_keyword_hits(flag_texts, _T3_KEYWORDS)
    if t3_hits >= 3:
        score += 0.20
    elif t3_hits >= 2:
        score += 0.15
    elif t3_hits >= 1:
        score += 0.08

    # ------------------------------------------------------------------
    # 5. Citation fabrication caught — Task 4 (0.20)
    # ------------------------------------------------------------------
    t4_hits = _count_keyword_hits(flag_texts, _T4_KEYWORDS)
    # Also check if agent used check_citation action
    citation_checked = any(
        f.get("citation_id") is not None for f in episode_state.flags_raised
    )

    if t4_hits >= 2 and citation_checked:
        score += 0.20
    elif t4_hits >= 2:
        score += 0.15
    elif t4_hits >= 1:
        score += 0.08

    # ------------------------------------------------------------------
    # Final: clamp to (0.0001, 0.9999)
    # ------------------------------------------------------------------
    score = max(0.0001, min(0.9999, score))
    return round(score, 4)
