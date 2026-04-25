"""
FastAPI application — OpenEnv-compliant HTTP interface.

Required endpoints (from spec):
  POST /reset      → Observation
  POST /step       → {observation, reward, done, info}
  GET  /state      → current episode state dict
  GET  /tasks      → list of tasks + action schemas
  POST /grader     → run grader on completed episode
  POST /baseline   → run baseline inference script, return scores
  GET  /health     → 200 OK (used by Docker HEALTHCHECK and judge ping)
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json

# Ensure project root is on path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import ResearchIntegrityEnv
from env.models import Action, ActionType

app = FastAPI(
    title="Research Integrity Gym",
    description=(
        "OpenEnv environment for training and evaluating AI agents on "
        "scientific research integrity tasks. Agents must audit methodology, "
        "replicate experiments, and verify statistical claims."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One global environment instance per server process
_env = ResearchIntegrityEnv()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1_methodology_audit"
    seed:    int | None = None

    class Config:
        extra = "ignore"  # Ignore extra fields


class StepRequest(BaseModel):
    action: Action


class GraderRequest(BaseModel):
    task_id: str
    episode_state: dict   # serialised state from a completed episode


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Root endpoint - redirects to API documentation."""
    return {
        "name": "Research Integrity Gym",
        "description": "OpenEnv environment for AI agents to evaluate scientific research integrity",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "tasks": "GET /tasks",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "environment": "research-integrity-gym"}


@app.post("/reset")
async def reset(request: Request):
    """Start a new episode. Returns initial Observation.
    
    Accepts:
      - Empty body
      - Body with just "null"
      - JSON body with task_id and/or seed
    """
    global _env
    
    # Parse body manually to handle empty/missing/null body
    body_bytes = await request.body()
    body_text = body_bytes.decode("utf-8").strip() if body_bytes else ""
    
    # Handle empty body, "null", "{}", or actual JSON
    body_data = {}
    if body_text and body_text != "null":
        try:
            parsed = json.loads(body_text)
            if isinstance(parsed, dict):
                body_data = parsed
            # If parsed is None or not a dict, keep body_data as empty dict
        except json.JSONDecodeError:
            pass  # Keep body_data as empty dict
    
    task_id = body_data.get("task_id", "task1_methodology_audit")
    seed = body_data.get("seed", None)
    
    if seed is not None:
        _env = ResearchIntegrityEnv(seed=seed)
    
    try:
        obs = _env.reset(task_id=task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Execute one action. Returns observation, reward, done, info."""
    try:
        obs, reward, done, info = _env.step(req.action)
        return {
            "observation": obs.model_dump(),
            "reward":      reward.model_dump(),
            "done":        done,
            "info":        info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state():
    """Return current episode state (excludes ground truth)."""
    return _env.state()


@app.get("/tasks")
def tasks():
    """Return all available tasks with their action schemas."""
    from tasks.task1_methodology_audit import MethodologyAuditTask
    from tasks.task2_replication import ReplicationTask
    from tasks.task3_claim_verify import ClaimVerifyTask
    from tasks.task4_citation_check import CitationCheckTask
    from tasks.task5_fda_approval import FDAApprovalTask

    task_list = [
        MethodologyAuditTask().task_info(),
        ReplicationTask().task_info(),
        ClaimVerifyTask().task_info(),
        CitationCheckTask().task_info(),
        FDAApprovalTask().task_info(),
    ]
    return {"tasks": task_list}


@app.post("/grader")
def grader(req: GraderRequest):
    """
    Run the grader for a completed episode externally.
    Accepts a serialised terminal_action and ground_truth.
    Used by the judge's automated evaluation pipeline.
    """
    from graders.grader1 import grade_audit
    from graders.grader2 import grade_results
    from graders.grader3 import grade_verdict
    from graders.grader4 import grade_citation_report
    from graders.grader5 import grade_fda_verdict
    from env.models import (
        SubmitAuditPayload, SubmitResultsPayload, SubmitVerdictPayload,
        SubmitCitationReportPayload, SubmitFDAVerdictPayload, FlawReport,
    )

    task_id      = req.task_id
    state_dict   = req.episode_state
    gt           = state_dict.get("ground_truth", {})
    terminal_act = state_dict.get("terminal_action", {})

    try:
        if task_id == "task1_methodology_audit":
            flaws   = [FlawReport(**f) for f in terminal_act.get("flaws", [])]
            payload = SubmitAuditPayload(flaws=flaws)
            score   = grade_audit(payload, gt)

        elif task_id == "task2_replication":
            payload = SubmitResultsPayload(**terminal_act)
            score   = grade_results(payload, gt)

        elif task_id == "task3_claim_verify":
            payload = SubmitVerdictPayload(**terminal_act)
            score   = grade_verdict(payload, gt)

        elif task_id == "task4_citation_check":
            payload = SubmitCitationReportPayload(**terminal_act)
            score   = grade_citation_report(payload, gt)

        elif task_id == "task5_fda_approval":
            payload = SubmitFDAVerdictPayload(**terminal_act)
            # For external grader calls, we create a minimal EpisodeState
            from env.state import EpisodeState
            mock_state = EpisodeState(
                task_id=task_id,
                flags_raised=state_dict.get("flags_raised", []),
                code_calls=state_dict.get("code_calls", 0),
            )
            score = grade_fda_verdict(payload, gt, mock_state)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

        return {"task_id": task_id, "grader_score": score}

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/baseline")
def baseline():
    """
    Trigger the baseline inference script and return scores for all 4 tasks.
    Requires HF_TOKEN in environment.
    """
    import subprocess
    import json

    api_key = os.environ.get("HF_TOKEN", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="HF_TOKEN not set. Add it to Space secrets.",
        )

    result = subprocess.run(
        [sys.executable, "baseline.py", "--output-json"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "HF_TOKEN": api_key},
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Baseline script failed:\n{result.stderr[:2000]}",
        )

    try:
        scores = json.loads(result.stdout)
        return scores
    except json.JSONDecodeError:
        return {"raw_output": result.stdout[:3000]}


# ---------------------------------------------------------------------------
# Mount Gradio demo UI at root
# ---------------------------------------------------------------------------
import gradio as gr
from app import demo as gradio_demo

app = gr.mount_gradio_app(app, gradio_demo, path="/")

