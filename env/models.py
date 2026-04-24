"""
OpenEnv typed models — Observation, Action, Reward.
These must pass `openenv validate`. Every field is explicit; no Optional abuse.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    read_section        = "read_section"
    read_dataset        = "read_dataset"
    execute_code        = "execute_code"
    flag_flaw           = "flag_flaw"           # Task 1
    flag_concern        = "flag_concern"        # Task 3
    check_citation      = "check_citation"      # Task 4
    flag_fabrication    = "flag_fabrication"    # Task 4
    submit_audit        = "submit_audit"        # Task 1 terminal
    submit_results      = "submit_results"      # Task 2 terminal
    submit_verdict      = "submit_verdict"      # Task 3 terminal
    submit_report       = "submit_report"       # Task 4 terminal
    submit_fda_verdict  = "submit_fda_verdict"  # Task 5 terminal


class FlawReport(BaseModel):
    flaw_type:   str = Field(..., description="e.g. wrong_statistical_test, underpowered_sample")
    location:    str = Field(..., description="Section or sentence reference")
    description: str = Field(..., description="Agent's explanation")


class SubmitAuditPayload(BaseModel):
    flaws: list[FlawReport]


class SubmitResultsPayload(BaseModel):
    auc:            float
    f1:             float
    interpretation: str = Field(..., max_length=1000)


class Verdict(str, Enum):
    valid            = "valid"
    partially_valid  = "partially_valid"
    invalid          = "invalid"


class SubmitVerdictPayload(BaseModel):
    verdict:        Verdict
    effect_size:    float
    p_value:        float
    justification:  str = Field(..., min_length=100, max_length=2000)

    @field_validator("p_value")
    @classmethod
    def p_value_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("p_value must be between 0.0 and 1.0")
        return v
    
    @field_validator("justification")
    @classmethod
    def justification_has_structure(cls, v: str) -> str:
        # Require minimum word count to prevent keyword stuffing
        word_count = len(v.split())
        if word_count < 20:
            raise ValueError("Justification must contain at least 20 words")
        return v


class SubmitCitationReportPayload(BaseModel):
    fabricated_citation_id:     Optional[int] = Field(None, description="Which citation is fake (1-4)")
    fabrication_type:           str = Field(..., max_length=500, description="Type of fabrication detected")
    verified_correct_citations: list[int] = Field(default_factory=list, description="Which citations are accurate")
    evidence:                   str = Field(..., min_length=20, max_length=1000, description="Specific quote showing mismatch")


class FDADecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT  = "REJECT"
    REVISE  = "REVISE"


class SubmitFDAVerdictPayload(BaseModel):
    """Terminal payload for Task 5: FDA Approval capstone."""
    decision: FDADecision = Field(
        ..., description="APPROVE | REJECT | REVISE"
    )
    justification_flags: list[str] = Field(
        default_factory=list,
        description="List of flags justifying the decision, e.g. "
                    "['protocol_deviation', 'class_imbalance', 'deleted_patients', 'citation_fabrication']"
    )


class Action(BaseModel):
    action_type:  ActionType

    # read_section
    section:      Optional[str] = None

    # execute_code
    code:         Optional[str] = None

    # flag_flaw / flag_concern
    flaw_type:    Optional[str] = None
    location:     Optional[str] = None
    description:  Optional[str] = None
    concern_type: Optional[str] = None
    evidence:     Optional[str] = None

    # Task 4: check_citation / flag_fabrication
    citation_id:  Optional[int] = None

    # terminal submit payloads — exactly one will be populated per terminal action
    audit_payload:       Optional[SubmitAuditPayload]           = None
    results_payload:     Optional[SubmitResultsPayload]         = None
    verdict_payload:     Optional[SubmitVerdictPayload]         = None
    report_payload:      Optional[SubmitCitationReportPayload]  = None
    fda_verdict_payload: Optional[SubmitFDAVerdictPayload]      = None

    # generic overflow for future extensibility
    payload: Optional[dict] = None


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    task_id:          str
    step:             int  = Field(..., ge=0)
    paper_text:       str  = Field(..., description="Full paper stub visible to agent")
    dataset_summary:  Optional[str] = None
    code_result:      Optional[str] = None
    last_reward:      float = 0.0
    flags_raised:     list[str] = Field(default_factory=list)
    available_actions: list[str] = Field(default_factory=list)
    done:             bool = False
    info:             dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    total:       float
    components:  dict[str, float] = Field(default_factory=dict)
    step_reward: float
    cumulative:  float
    is_terminal: bool
    grader_score: Optional[float] = None   # only populated on terminal step

    @field_validator("total", "step_reward", "cumulative")
    @classmethod
    def clamp_reward(cls, v: float) -> float:
        # Rewards are not clamped here — clamping happens in the environment.
        # Validator just ensures they are finite floats.
        if not (-1e6 < v < 1e6):
            raise ValueError(f"Reward value {v} out of reasonable range")
        return round(v, 6)
