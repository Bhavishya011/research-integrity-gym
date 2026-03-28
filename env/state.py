"""
EpisodeState — mutable state for a single episode.
The environment owns one of these per active session.
Designed to be serialisable to dict for the state() endpoint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EpisodeState:
    # ------------------------------------------------------------------ identity
    task_id:        str = ""
    session_id:     str = ""
    episode_number: int = 0

    # ------------------------------------------------------------------ timing
    step:           int   = 0
    max_steps:      int   = 20
    started_at:     float = field(default_factory=time.time)

    # ------------------------------------------------------------------ paper / data
    paper_text:     str         = ""
    paper_sections: dict        = field(default_factory=dict)  # section_name -> text
    dataset_path:   Optional[str] = None
    dataset_loaded: bool        = False

    # ------------------------------------------------------------------ agent tracking
    sections_read:  list[str]   = field(default_factory=list)
    dataset_read:   bool        = False
    code_calls:     int         = 0
    code_errors:    int         = 0

    # flags raised mid-episode (before terminal submit)
    flags_raised:   list[dict]  = field(default_factory=list)

    # last code execution result visible to agent
    last_code_result: Optional[str] = None

    # action deduplication — (action_type, hash(payload)) -> count
    action_counts:  dict        = field(default_factory=dict)

    # ------------------------------------------------------------------ reward tracking
    cumulative_reward: float    = 0.0
    reward_history:    list[float] = field(default_factory=list)

    # penalty accumulator — false positives, loops, etc.
    false_positive_count:  int  = 0
    repeat_action_count:   int  = 0

    # ------------------------------------------------------------------ terminal
    done:           bool        = False
    terminal_action: Optional[dict] = None   # the submit_* payload
    grader_score:   Optional[float] = None

    # ------------------------------------------------------------------ ground truth (hidden from agent)
    ground_truth:   dict        = field(default_factory=dict)

    def is_over_budget(self) -> bool:
        return self.step >= self.max_steps

    def action_key(self, action_type: str, payload_repr: str) -> str:
        return f"{action_type}::{payload_repr}"

    def register_action(self, action_type: str, payload_repr: str) -> int:
        """Returns how many times this exact action has been taken (including now)."""
        key = self.action_key(action_type, payload_repr)
        self.action_counts[key] = self.action_counts.get(key, 0) + 1
        return self.action_counts[key]

    def add_reward(self, r: float) -> None:
        self.cumulative_reward = round(self.cumulative_reward + r, 6)
        self.reward_history.append(r)

    def to_dict(self) -> dict:
        """Serialise for the state() API endpoint. Ground truth is EXCLUDED."""
        return {
            "task_id":           self.task_id,
            "session_id":        self.session_id,
            "step":              self.step,
            "max_steps":         self.max_steps,
            "done":              self.done,
            "cumulative_reward": self.cumulative_reward,
            "flags_raised":      self.flags_raised,
            "sections_read":     self.sections_read,
            "dataset_read":      self.dataset_read,
            "code_calls":        self.code_calls,
            "code_errors":       self.code_errors,
            "grader_score":      self.grader_score,
            "elapsed_seconds":   round(time.time() - self.started_at, 2),
        }
