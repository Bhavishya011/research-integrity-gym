"""
Reward engine — every signal is deterministic pure-Python logic.
NO LLM calls. NO randomness. Same inputs always produce same outputs.

Design principle: mid-episode signals are scaled so they never exceed 0.30
of total possible reward. The grader terminal score always dominates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from env.state import EpisodeState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mid-episode positive signals
R_RELEVANT_SECTION_READ  = 0.02   # read a section that contains a planted flaw
R_DATASET_FIRST_READ     = 0.03   # first time agent reads the dataset
R_CODE_EXECUTES_OK       = 0.05   # execute_code returns stdout (tasks 2 & 3)
R_FLAG_CORRECT_EARLY     = 0.08   # flags a real flaw before terminal submit
R_INTERMEDIATE_CLOSE     = 0.05   # task 2: intermediate metric within 20% of GT

# Mid-episode negative signals
R_CODE_EXCEPTION         = -0.03  # execute_code throws an exception
R_REPEAT_ACTION          = -0.05  # exact same action called again
R_FALSE_POSITIVE_FLAG    = -0.05  # flagged a non-existent flaw (per occurrence)
R_FALSE_POSITIVE_CAP     = -0.20  # cap on total false positive penalty
R_STEP_BUDGET_EXCEEDED   = -0.10  # exceeded max_steps

# Terminal (grader) weight
# Final reward = grader_score * GRADER_WEIGHT + mid_episode_total * MID_WEIGHT
# Ensures grader always dominates.
GRADER_WEIGHT   = 0.80
MID_WEIGHT      = 0.20
MID_EPISODE_MAX = 0.30  # mid-episode signals are clamped to this


@dataclass
class RewardComponents:
    exploration:    float = 0.0
    data_grounding: float = 0.0
    code_quality:   float = 0.0
    flaw_detection: float = 0.0
    grader_score:   float = 0.0
    penalties:      float = 0.0

    def total_mid_episode(self) -> float:
        raw = (self.exploration + self.data_grounding +
               self.code_quality + self.flaw_detection)
        return min(raw, MID_EPISODE_MAX)

    def to_dict(self) -> dict[str, float]:
        return {
            "exploration":    round(self.exploration, 4),
            "data_grounding": round(self.data_grounding, 4),
            "code_quality":   round(self.code_quality, 4),
            "flaw_detection": round(self.flaw_detection, 4),
            "grader_score":   round(self.grader_score, 4),
            "penalties":      round(self.penalties, 4),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_step_reward(
    action_type: str,
    action_payload: dict,
    state: EpisodeState,
    execution_ok: Optional[bool] = None,
    flagged_flaw_id: Optional[str] = None,
    is_false_positive: Optional[bool] = None,
    intermediate_metric_close: Optional[bool] = None,
) -> tuple[float, RewardComponents]:
    """
    Compute the reward signal for a single non-terminal step.
    Returns (step_reward, components).
    """
    c = RewardComponents()

    # --- Exploration: reading a relevant section ---
    if action_type == "read_section":
        section_name = action_payload.get("section", "")
        relevant_sections = state.ground_truth.get("flaw_sections", [])
        if section_name in relevant_sections and section_name not in state.sections_read:
            c.exploration += R_RELEVANT_SECTION_READ

    # --- Data grounding: first dataset read ---
    if action_type in ("read_dataset", "request_dataset_summary"):
        if not state.dataset_read:
            c.data_grounding += R_DATASET_FIRST_READ

    # --- Code quality ---
    if action_type == "execute_code":
        if execution_ok is True:
            c.code_quality += R_CODE_EXECUTES_OK
        elif execution_ok is False:
            c.code_quality += R_CODE_EXCEPTION
        if intermediate_metric_close is True:
            c.code_quality += R_INTERMEDIATE_CLOSE

    # --- Flaw detection: flagging a real flaw mid-episode ---
    if action_type in ("flag_flaw", "flag_concern"):
        if flagged_flaw_id and is_false_positive is False:
            # Only give credit if this flaw hasn't been flagged before
            already_flagged = [f.get("flaw_id") for f in state.flags_raised]
            if flagged_flaw_id not in already_flagged:
                c.flaw_detection += R_FLAG_CORRECT_EARLY
        elif is_false_positive is True:
            fp_so_far = state.false_positive_count
            penalty_so_far = fp_so_far * R_FALSE_POSITIVE_FLAG
            if penalty_so_far > R_FALSE_POSITIVE_CAP:
                pass  # cap already hit, no more penalty
            else:
                c.penalties += R_FALSE_POSITIVE_FLAG

    # --- Repeat action penalty ---
    payload_repr = str(sorted(action_payload.items()))
    count = state.register_action(action_type, payload_repr)
    if count > 1:
        c.penalties += R_REPEAT_ACTION

    step_r = round(
        c.exploration + c.data_grounding + c.code_quality +
        c.flaw_detection + c.penalties,
        6
    )
    return step_r, c


def compute_terminal_reward(
    grader_score: float,
    state: EpisodeState,
) -> tuple[float, RewardComponents]:
    """
    Compute the terminal reward combining grader score with mid-episode total.
    Returns (terminal_step_reward, components).
    """
    c = RewardComponents()
    c.grader_score = grader_score

    # Budget exceeded penalty
    if state.is_over_budget():
        c.penalties += R_STEP_BUDGET_EXCEEDED

    mid_total = min(state.cumulative_reward, MID_EPISODE_MAX)
    terminal_r = round(
        grader_score * GRADER_WEIGHT + mid_total * MID_WEIGHT + c.penalties,
        6
    )
    # Clamp final to [0.0, 1.0]
    terminal_r = max(0.0, min(1.0, terminal_r))
    return terminal_r, c
