"""
ResearchIntegrityEnv — the main OpenEnv environment class.

Trap mitigations:
  1. Subjectivity Trap  → graders are imported from graders/ and are 100% deterministic code.
                          No LLM judge anywhere in this file.
  2. Sandboxing         → execute_code runs in a subprocess jail: 5-second timeout,
                          no network, restricted builtins. step() NEVER hangs.
  3. Data Leakage       → paper stubs are procedurally generated at reset() via
                          PaperGenerator. Each episode sees different surface text
                          with the same underlying flaw taxonomy.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import textwrap
import uuid
from typing import Optional

# Windows compatibility: resource module is Unix-only
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

from env.models import (
    Action, ActionType, Observation, Reward,
    SubmitAuditPayload, SubmitResultsPayload, SubmitVerdictPayload,
    SubmitCitationReportPayload, SubmitFDAVerdictPayload,
)
from env.reward import compute_step_reward, compute_terminal_reward
from env.state import EpisodeState

# Tasks are loaded lazily to keep imports clean
_TASK_REGISTRY: dict[str, type] = {}


def _load_task_registry() -> None:
    global _TASK_REGISTRY
    if _TASK_REGISTRY:
        return
    from tasks.task1_methodology_audit import MethodologyAuditTask
    from tasks.task2_replication import ReplicationTask
    from tasks.task3_claim_verify import ClaimVerifyTask
    from tasks.task4_citation_check import CitationCheckTask
    from tasks.task5_fda_approval import FDAApprovalTask
    _TASK_REGISTRY = {
        "task1_methodology_audit": MethodologyAuditTask,
        "task2_replication":       ReplicationTask,
        "task3_claim_verify":      ClaimVerifyTask,
        "task4_citation_check":    CitationCheckTask,
        "task5_fda_approval":      FDAApprovalTask,
    }


# ---------------------------------------------------------------------------
# Sandbox constants
# ---------------------------------------------------------------------------

_SANDBOX_TIMEOUT_SECS  = 5
_SANDBOX_MEMORY_MB     = 384
_SANDBOX_MAX_OUTPUT    = 4_000   # chars — truncate stdout beyond this

# Dangerous imports blocked in the sandbox wrapper
_BLOCKED_IMPORTS = [
    "os", "sys", "subprocess", "socket", "urllib",
    "http", "requests", "shutil", "pathlib", "glob",
    "importlib", "ctypes", "multiprocessing", "threading",
    "signal", "resource", "gc", "eval", "exec",
]

_SANDBOX_PRELUDE_UNIX = textwrap.dedent("""
import builtins as _b
_real_import = _b.__import__
_allowed_roots = {{
    "math", "statistics", "collections", "itertools", "functools",
    "csv", "json", "io", "re", "numpy", "pandas", "sklearn", "scipy"
}}
_blocked_roots = {blocked_imports}

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root in _blocked_roots or root not in _allowed_roots:
        raise ImportError(f"Import '{{name}}' is not allowed in sandbox")
    return _real_import(name, globals, locals, fromlist, level)

def _safe_open(path, mode="r", *args, **kwargs):
    dataset_path = str(globals().get("DATASET_PATH", ""))
    path_str = str(path)
    if any(flag in mode for flag in ("w", "a", "+", "x")):
        raise PermissionError("sandbox open() is read-only")
    if dataset_path and path_str == dataset_path:
        return _b.open(path, mode, *args, **kwargs)
    raise PermissionError("sandbox open() is only allowed for DATASET_PATH")

_safe_builtins = {{k: getattr(_b, k) for k in dir(_b)
                  if k not in ('compile','eval','exec',
                               'breakpoint','input','memoryview','vars',
                               '__loader__','__spec__')}}
_safe_builtins["__import__"] = _safe_import
_safe_builtins["open"] = _safe_open
__builtins__ = _safe_builtins

import signal as _signal
_signal.alarm({timeout})

import resource as _res
_res.setrlimit(_res.RLIMIT_AS, ({mem}, {mem}))
""").strip()

_SANDBOX_PRELUDE_WINDOWS = textwrap.dedent("""
import builtins as _b
_real_import = _b.__import__
_allowed_roots = {{
    "math", "statistics", "collections", "itertools", "functools",
    "csv", "json", "io", "re", "numpy", "pandas", "sklearn", "scipy"
}}
_blocked_roots = {blocked_imports}

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root in _blocked_roots or root not in _allowed_roots:
        raise ImportError(f"Import '{{name}}' is not allowed in sandbox")
    return _real_import(name, globals, locals, fromlist, level)

def _safe_open(path, mode="r", *args, **kwargs):
    dataset_path = str(globals().get("DATASET_PATH", ""))
    path_str = str(path)
    if any(flag in mode for flag in ("w", "a", "+", "x")):
        raise PermissionError("sandbox open() is read-only")
    if dataset_path and path_str == dataset_path:
        return _b.open(path, mode, *args, **kwargs)
    raise PermissionError("sandbox open() is only allowed for DATASET_PATH")

_safe_builtins = {{k: getattr(_b, k) for k in dir(_b)
                  if k not in ('compile','eval','exec',
                               'breakpoint','input','memoryview','vars',
                               '__loader__','__spec__')}}
_safe_builtins["__import__"] = _safe_import
_safe_builtins["open"] = _safe_open
__builtins__ = _safe_builtins

# Windows: signal.alarm and resource.setrlimit not available
# Timeout is handled by subprocess timeout parameter
""").strip()

_SANDBOX_PRELUDE = _SANDBOX_PRELUDE_UNIX if HAS_RESOURCE else _SANDBOX_PRELUDE_WINDOWS


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ResearchIntegrityEnv:
    """
    OpenEnv-compliant environment for research integrity evaluation.

    Usage:
        env = ResearchIntegrityEnv()
        obs = env.reset("task1_methodology_audit")
        obs, reward, done, info = env.step(action)
        state = env.state()
    """

    AVAILABLE_TASKS = [
        "task1_methodology_audit",
        "task2_replication",
        "task3_claim_verify",
        "task4_citation_check",
        "task5_fda_approval",
    ]

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._state: Optional[EpisodeState] = None
        self._task = None
        _load_task_registry()

    # ------------------------------------------------------------------
    # OpenEnv spec methods
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1_methodology_audit") -> Observation:
        """
        Begin a new episode. Returns the initial Observation.
        Procedurally generates a fresh paper stub — mitigates data leakage.
        """
        if task_id not in self.AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Choose from: {self.AVAILABLE_TASKS}"
            )

        # Build task instance — task generates its own paper procedurally
        task_cls = _TASK_REGISTRY[task_id]
        self._task = task_cls(seed=self._seed)
        episode_data = self._task.generate_episode()

        # Build fresh state
        self._state = EpisodeState(
            task_id        = task_id,
            session_id     = str(uuid.uuid4()),
            paper_text     = episode_data["paper_text"],
            paper_sections = episode_data["paper_sections"],
            dataset_path   = episode_data.get("dataset_path"),
            ground_truth   = episode_data["ground_truth"],   # hidden
            max_steps      = 20,
        )

        return self._build_observation(last_reward=0.0)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute one action. Returns (observation, reward, done, info).
        step() is guaranteed to return — sandbox timeout enforces this.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode is over. Call reset() to start a new one.")

        state = self._state
        state.step += 1

        # -- Dispatch action -------------------------------------------------
        info: dict = {}
        step_r = 0.0
        components = {}
        done = False
        grader_score: Optional[float] = None

        atype = action.action_type

        if atype == ActionType.read_section:
            step_r, comp = self._handle_read_section(action, state)

        elif atype == ActionType.read_dataset:
            step_r, comp = self._handle_read_dataset(action, state)

        elif atype == ActionType.execute_code:
            step_r, comp = self._handle_execute_code(action, state)

        elif atype in (ActionType.flag_flaw, ActionType.flag_concern):
            step_r, comp = self._handle_flag(action, state)
        
        elif atype in (ActionType.check_citation, ActionType.flag_fabrication):
            step_r, comp = self._handle_citation_flag(action, state)

        elif atype == ActionType.submit_audit:
            step_r, comp, grader_score = self._handle_submit_audit(action, state)
            done = True

        elif atype == ActionType.submit_results:
            step_r, comp, grader_score = self._handle_submit_results(action, state)
            done = True

        elif atype == ActionType.submit_verdict:
            step_r, comp, grader_score = self._handle_submit_verdict(action, state)
            done = True

        elif atype == ActionType.submit_report:
            step_r, comp, grader_score = self._handle_submit_report(action, state)
            done = True

        elif atype == ActionType.submit_fda_verdict:
            step_r, comp, grader_score = self._handle_submit_fda_verdict(action, state)
            done = True

        else:
            # Unknown action type — small penalty, do not crash
            step_r = -0.02
            comp = {}
            info["warning"] = f"Unknown action_type: {atype}"

        # -- Budget check ----------------------------------------------------
        if not done and state.is_over_budget():
            done = True
            # Apply budget penalty via terminal reward calc
            grader_score = grader_score or 0.0

        # -- Terminal reward -------------------------------------------------
        if done:
            if grader_score is None:
                grader_score = 0.0
            terminal_r, terminal_comp = compute_terminal_reward(grader_score, state)
            state.grader_score = grader_score
            state.done = True
            reward_obj = Reward(
                total        = terminal_r,
                components   = terminal_comp.to_dict(),
                step_reward  = terminal_r,
                cumulative   = round(state.cumulative_reward + terminal_r, 6),
                is_terminal  = True,
                grader_score = grader_score,
            )
        else:
            state.add_reward(step_r)
            reward_obj = Reward(
                total        = step_r,
                components   = comp.to_dict() if hasattr(comp, "to_dict") else comp,
                step_reward  = step_r,
                cumulative   = state.cumulative_reward,
                is_terminal  = False,
            )

        obs = self._build_observation(last_reward=step_r)
        obs.done = done
        return obs, reward_obj, done, info

    def state(self) -> dict:
        """Return current episode state (excludes ground truth)."""
        if self._state is None:
            return {"status": "no_active_episode"}
        return self._state.to_dict()

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_read_section(
        self, action: Action, state: EpisodeState
    ) -> tuple[float, object]:
        section = (action.section or "").strip().lower()
        sections = state.paper_sections

        # Find matching section (case-insensitive prefix match)
        matched_key = next(
            (k for k in sections if k.lower().startswith(section)), None
        )
        if matched_key:
            state.sections_read.append(matched_key)
            # Update last_code_result field to show section text to agent
            state.last_code_result = None  # clear code result
            # (section text is surfaced via paper_sections in observation)
        else:
            state.last_code_result = f"[No section matching '{section}' found]"

        step_r, comp = compute_step_reward(
            action_type     = "read_section",
            action_payload  = {"section": section},
            state           = state,
        )
        return step_r, comp

    def _handle_read_dataset(
        self, action: Action, state: EpisodeState
    ) -> tuple[float, object]:
        if not state.dataset_path or not os.path.exists(state.dataset_path):
            state.last_code_result = "[No dataset available for this task]"
            return -0.01, {}

        import pandas as pd
        try:
            df = pd.read_csv(state.dataset_path)
            n_rows = len(df)
            n_cols = len(df.columns)
            preview = df.head(5).to_string()
            dtypes  = df.dtypes.to_string()
            summary = (
                f"Shape: {n_rows} rows × {n_cols} cols\n"
                f"Columns & types:\n{dtypes}\n\n"
                f"First 5 rows:\n{preview}\n\n"
                f"Describe:\n{df.describe().to_string()}"
            )
            state.last_code_result = summary[:_SANDBOX_MAX_OUTPUT]
            state.dataset_read = True
            state.dataset_loaded = True
        except Exception as e:
            state.last_code_result = f"[Dataset read error: {e}]"

        step_r, comp = compute_step_reward(
            action_type    = "read_dataset",
            action_payload = {},
            state          = state,
        )
        return step_r, comp

    def _handle_execute_code(
        self, action: Action, state: EpisodeState
    ) -> tuple[float, object]:
        """
        SANDBOXED execution.
        Uses subprocess with a hard 5-second wall-clock timeout.
        Parent process NEVER hangs — subprocess.run(timeout=...) raises
        TimeoutExpired which we catch and handle gracefully.
        """
        code = action.code or ""
        state.code_calls += 1

        dataset_path = state.dataset_path or ""
        result = _run_sandboxed(code, dataset_path=dataset_path)

        execution_ok = result["ok"]
        output = result["output"]

        if not execution_ok:
            state.code_errors += 1

        state.last_code_result = output[:_SANDBOX_MAX_OUTPUT]

        # Check if intermediate metric is close (task 2 specific)
        intermediate_close = _check_intermediate_close(output, state.ground_truth)

        step_r, comp = compute_step_reward(
            action_type               = "execute_code",
            action_payload            = {"code_hash": hashlib.md5(code.encode()).hexdigest()},
            state                     = state,
            execution_ok              = execution_ok,
            intermediate_metric_close = intermediate_close,
        )
        return step_r, comp

    def _handle_flag(
        self, action: Action, state: EpisodeState
    ) -> tuple[float, object]:
        flaw_type    = (action.flaw_type or action.concern_type or "").strip()
        location     = (action.location or "").strip()
        description  = (action.description or action.evidence or "").strip()

        gt_flaws = state.ground_truth.get("flaws", [])

        # Deterministic match: check flaw_type against known planted flaws
        matched_id, is_fp = _match_flaw(flaw_type, location, gt_flaws)

        state.flags_raised.append({
            "flaw_id":    matched_id,
            "flaw_type":  flaw_type,
            "location":   location,
            "description": description,
            "is_fp":       is_fp,
        })

        if is_fp:
            state.false_positive_count += 1

        step_r, comp = compute_step_reward(
            action_type      = action.action_type.value,
            action_payload   = {"flaw_type": flaw_type, "location": location},
            state            = state,
            flagged_flaw_id  = matched_id,
            is_false_positive = is_fp,
        )
        return step_r, comp

    def _handle_submit_audit(
        self, action: Action, state: EpisodeState
    ) -> tuple[float, object, float]:
        from graders.grader1 import grade_audit
        payload = action.audit_payload or SubmitAuditPayload(flaws=[])
        state.terminal_action = payload.model_dump()
        grader_score = grade_audit(payload, state.ground_truth)
        step_r, comp = compute_terminal_reward(grader_score, state)
        return step_r, comp, grader_score

    def _handle_submit_results(
        self, action: Action, state: EpisodeState
    ) -> tuple[float, object, float]:
        from graders.grader2 import grade_results
        payload = action.results_payload
        if payload is None:
            return 0.0, {}, 0.0
        state.terminal_action = payload.model_dump()
        grader_score = grade_results(payload, state.ground_truth)
        step_r, comp = compute_terminal_reward(grader_score, state)
        return step_r, comp, grader_score

    def _handle_submit_verdict(
        self, action: Action, state: EpisodeState
    ) -> tuple[float, object, float]:
        from graders.grader3 import grade_verdict
        payload = action.verdict_payload
        if payload is None:
            return 0.0, {}, 0.0
        state.terminal_action = payload.model_dump()
        grader_score = grade_verdict(payload, state.ground_truth)
        step_r, comp = compute_terminal_reward(grader_score, state)
        return step_r, comp, grader_score

    def _handle_citation_flag(
        self, action: Action, state: EpisodeState
    ) -> tuple[float, object]:
        """Handle check_citation and flag_fabrication actions for Task 4."""
        citation_id = action.citation_id
        
        if action.action_type == ActionType.check_citation:
            # Show citation details
            citations = state.ground_truth.get("all_citations", [])
            match = next((c for c in citations if c["id"] == citation_id), None)
            if match:
                state.last_code_result = (
                    f"Citation [{citation_id}] {match['author']} ({match['year']}):\n"
                    f"Excerpt: \"{match['excerpt']}\""
                )
                return 0.03, {}
            state.last_code_result = f"[Citation {citation_id} not found]"
            return 0.0, {}
        
        elif action.action_type == ActionType.flag_fabrication:
            # Track flagged citation
            state.flags_raised.append({
                "citation_id": citation_id,
                "flaw_type": action.flaw_type or "",
                "fabrication_type": action.flaw_type or "",
                "evidence": action.description or "",
            })
            gt_id = state.ground_truth.get("fabricated_id")
            if citation_id == gt_id:
                return 0.08, {}  # Small reward for finding it
            return -0.05, {}  # Penalty for false positive
        
        return 0.0, {}

    def _handle_submit_report(
        self, action: Action, state: EpisodeState
    ) -> tuple[float, object, float]:
        """Terminal action for Task 4."""
        from graders.grader4 import grade_citation_report
        payload = action.report_payload
        if payload is None:
            return 0.0, {}, 0.0
        state.terminal_action = {
            "fabricated_citation_id":     payload.fabricated_citation_id,
            "fabrication_type":           payload.fabrication_type,
            "verified_correct_citations": payload.verified_correct_citations,
            "evidence":                   payload.evidence,
        }
        grader_score = grade_citation_report(payload, state.ground_truth)
        step_r, comp = compute_terminal_reward(grader_score, state)
        return step_r, comp, grader_score

    def _handle_submit_fda_verdict(
        self, action: Action, state: EpisodeState
    ) -> tuple[float, object, float]:
        """Terminal action for Task 5: FDA Approval capstone."""
        from graders.grader5 import grade_fda_verdict
        payload = action.fda_verdict_payload
        if payload is None:
            return 0.0, {}, 0.0
        state.terminal_action = {
            "decision":            payload.decision.value,
            "justification_flags": payload.justification_flags,
        }
        grader_score = grade_fda_verdict(payload, state.ground_truth, state)
        step_r, comp = compute_terminal_reward(grader_score, state)
        return step_r, comp, grader_score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self, last_reward: float) -> Observation:
        state = self._state
        assert state is not None

        # Surface the currently-read section's text if any
        last_section_text = ""
        if state.sections_read:
            last_key = state.sections_read[-1]
            last_section_text = state.paper_sections.get(last_key, "")

        paper_visible = state.paper_text
        if last_section_text:
            paper_visible += f"\n\n--- Section '{state.sections_read[-1]}' ---\n{last_section_text}"

        # Prepend FDA persona prompt for Task 5
        if state.task_id == "task5_fda_approval":
            persona = (
                "You are an autonomous FDA Lead Regulator. Your job is to audit "
                "clinical trial submissions using tools, execute Python to verify "
                "raw patient datasets, and catch statistical manipulation.\n\n"
            )
            paper_visible = persona + paper_visible

        rendered_flags = []
        for f in state.flags_raised:
            if f.get("citation_id") is not None:
                fab_type = f.get("fabrication_type") or f.get("flaw_type") or "citation_flag"
                rendered_flags.append(f"citation[{f.get('citation_id')}] {fab_type}")
            else:
                flaw_type = f.get("flaw_type") or f.get("concern_type") or "flag"
                location = f.get("location") or "unspecified"
                rendered_flags.append(f"{flaw_type} @ {location}")

        return Observation(
            task_id           = state.task_id,
            step              = state.step,
            paper_text        = paper_visible,
            dataset_summary   = state.last_code_result if state.dataset_read else None,
            code_result       = state.last_code_result,
            last_reward       = last_reward,
            flags_raised      = rendered_flags,
            available_actions = _available_actions(state),
            done              = state.done,
            info              = {"step": state.step, "max_steps": state.max_steps},
        )


# ---------------------------------------------------------------------------
# Subprocess sandbox
# ---------------------------------------------------------------------------

def _run_sandboxed(code: str, dataset_path: str = "") -> dict:
    """
    Run agent-supplied code in a subprocess with hard timeout.
    Returns {"ok": bool, "output": str}.

    Security layers:
      1. subprocess — agent code can't affect parent process memory
      2. signal.alarm(5) inside child — kills the child from inside
      3. subprocess.run(timeout=6) — kills the child from outside if alarm fails
      4. __builtins__ replacement — restricted open()/imports/eval/exec
      5. Memory limit via resource.setrlimit
    """
    mem_bytes = _SANDBOX_MEMORY_MB * 1024 * 1024

    prelude = _SANDBOX_PRELUDE.format(
        timeout         = _SANDBOX_TIMEOUT_SECS,
        mem             = mem_bytes,
        blocked_imports = set(_BLOCKED_IMPORTS),
    )

    # Inject dataset path as a constant the agent code can reference
    dataset_inject = f'DATASET_PATH = {repr(dataset_path)}\n' if dataset_path else ""

    full_code = prelude + "\n" + dataset_inject + "\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="rig_sandbox_"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output = True,
            text           = True,
            timeout        = _SANDBOX_TIMEOUT_SECS + 1,  # outer wall-clock guard
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode == 0:
            return {"ok": True,  "output": stdout[:_SANDBOX_MAX_OUTPUT]}
        else:
            return {"ok": False, "output": f"[Error]\n{stderr[:1000]}"}

    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "[Execution timed out after 5 seconds]"}
    except Exception as e:
        return {"ok": False, "output": f"[Sandbox error: {e}]"}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Flaw matching — deterministic, no LLM
# ---------------------------------------------------------------------------

# Normalised synonyms for flaw taxonomy
_FLAW_SYNONYMS: dict[str, list[str]] = {
    # --- Medical taxonomy (primary keys for PeerGuard) ---
    "unblinded_investigator_bias": [
        "unblinded", "investigator bias", "detection bias", "observer bias",
        "assessor bias", "blinding", "single-blind", "open-label",
        # Legacy synonyms for backward compat
        "wrong test", "incorrect test", "t-test", "t test",
        "chi-square", "anova", "parametric", "non-parametric",
        "wrong statistical", "inappropriate test",
    ],
    "insufficient_power_analysis": [
        "power analysis", "insufficient power", "underpowered",
        "sample size", "small n", "insufficient participants",
        "low power", "n too small",
    ],
    "protocol_deviation_unreported": [
        "protocol deviation", "unreported deviation", "protocol violation",
        "exclusion", "excluded", "outlier", "removed", "undisclosed",
        "missing data", "data exclusion", "selective",
    ],
    "endpoint_switching": [
        "endpoint switching", "outcome switching", "primary endpoint",
        "p-value", "p value", "fishing", "hacking", "selective reporting",
        "multiple comparison", "bonferroni", "significance threshold",
    ],
    # --- Legacy taxonomy (kept for backward compat with old episodes) ---
    "wrong_statistical_test": [
        "wrong test", "incorrect test", "t-test", "t test",
        "chi-square", "anova", "parametric", "non-parametric",
        "wrong statistical", "inappropriate test",
    ],
    "underpowered_sample": [
        "sample size", "underpowered", "small n", "insufficient participants",
        "low power", "power analysis", "n too small",
    ],
    "undisclosed_exclusion": [
        "exclusion", "excluded", "outlier", "removed", "undisclosed",
        "missing data", "data exclusion", "selective",
    ],
    "p_value_manipulation": [
        "p-value", "p value", "fishing", "hacking", "selective reporting",
        "multiple comparison", "bonferroni", "significance threshold",
    ],
    "class_imbalance_ignored": [
        "class imbalance", "imbalanced", "stratif", "oversampl", "undersampl",
        "weighted", "recall", "precision",
    ],
}


def _match_flaw(
    flaw_type: str, location: str, gt_flaws: list[dict]
) -> tuple[Optional[str], bool]:
    """
    Returns (flaw_id, is_false_positive).
    Match is deterministic taxonomy + substring matching. No LLM.
    """
    flaw_lower = flaw_type.lower()
    loc_lower  = location.lower()

    for gt_flaw in gt_flaws:
        gt_id       = gt_flaw["id"]
        gt_taxonomy = gt_flaw["taxonomy"]
        gt_location = gt_flaw.get("location", "").lower()

        synonyms = _FLAW_SYNONYMS.get(gt_taxonomy, [gt_taxonomy])

        # Type match: flaw_type contains any synonym
        type_match = any(syn in flaw_lower for syn in synonyms)

        # Location match: allow partial (agent might say "methods" for "methods section")
        loc_match = (
            not gt_location or
            gt_location in loc_lower or
            loc_lower in gt_location
        )

        if type_match and loc_match:
            return gt_id, False   # correct flag

    # No match found → false positive
    return None, True


# ---------------------------------------------------------------------------
# Intermediate metric check (task 2 only)
# ---------------------------------------------------------------------------

def _check_intermediate_close(output: str, ground_truth: dict) -> bool:
    """
    Returns True if the agent's code output contains a metric close to GT.
    Purely string-based — no LLM.
    """
    gt_auc = ground_truth.get("auc")
    gt_f1  = ground_truth.get("f1")
    if gt_auc is None and gt_f1 is None:
        return False

    import re
    numbers = re.findall(r"\b0\.\d{2,4}\b", output)
    for n in numbers:
        v = float(n)
        if gt_auc and abs(v - gt_auc) <= gt_auc * 0.20:
            return True
        if gt_f1 and abs(v - gt_f1) <= gt_f1 * 0.20:
            return True
    return False


# ---------------------------------------------------------------------------
# Available actions helper
# ---------------------------------------------------------------------------

def _available_actions(state: EpisodeState) -> list[str]:
    base = ["read_section", "read_dataset", "execute_code"]
    tid  = state.task_id

    if tid == "task1_methodology_audit":
        return base + ["flag_flaw", "submit_audit"]
    elif tid == "task2_replication":
        return base + ["submit_results"]
    elif tid == "task3_claim_verify":
        return base + ["flag_concern", "submit_verdict"]
    elif tid == "task4_citation_check":
        return base + ["check_citation", "flag_fabrication", "submit_report"]
    elif tid == "task5_fda_approval":
        # CRITICAL: Only investigatory actions from all sub-tasks.
        # NO sub-task terminal actions (submit_audit, submit_results, etc.)
        # — those would trigger done=True and kill the episode prematurely.
        # The ONLY way to end Task 5 is submit_fda_verdict.
        return base + [
            "flag_flaw",           # from Task 1
            "flag_concern",        # from Task 3
            "check_citation",      # from Task 4
            "flag_fabrication",    # from Task 4
            "submit_fda_verdict",  # Task 5 sole terminal
        ]
    return base
