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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
def reset(req: ResetRequest):
    """Start a new episode. Returns initial Observation."""
    global _env
    if req.seed is not None:
        _env = ResearchIntegrityEnv(seed=req.seed)
    try:
        obs = _env.reset(task_id=req.task_id)
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

    task_list = [
        MethodologyAuditTask().task_info(),
        ReplicationTask().task_info(),
        ClaimVerifyTask().task_info(),
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
    from env.models import (
        SubmitAuditPayload, SubmitResultsPayload, SubmitVerdictPayload,
        FlawReport,
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

        else:
            raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

        return {"task_id": task_id, "grader_score": score}

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/baseline")
def baseline():
    """
    Trigger the baseline inference script and return scores for all 3 tasks.
    Requires GROQ_API_KEY in environment.
    """
    import subprocess
    import json

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY not set. Add it to Space secrets.",
        )

    result = subprocess.run(
        [sys.executable, "baseline.py", "--output-json"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "GROQ_API_KEY": api_key},
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
