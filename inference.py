"""
===================================
MANDATORY COMPLIANCE — OpenEnv Hackathon
===================================
- Uses environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
- Uses OpenAI Client for all LLM calls
- Named inference.py at repo root

This script runs the baseline agent against all 3 tasks and produces scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ResearchIntegrityEnv
from env.models import (
    Action, ActionType,
    FlawReport, SubmitAuditPayload,
    SubmitResultsPayload, SubmitVerdictPayload, Verdict,
)

# ---------------------------------------------------------------------------
# MANDATORY: Use environment variables as specified by hackathon
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

MAX_STEPS = 15
SEED = 42


# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "task1_methodology_audit": textwrap.dedent("""
        You are a scientific peer reviewer. Read the paper stub and identify
        methodological flaws. There are exactly 4 planted flaws.

        Available actions (respond with JSON only, no prose):
          {"action_type": "read_section", "section": "<name>"}
          {"action_type": "flag_flaw", "flaw_type": "<type>", "location": "<section>", "description": "<text>"}
          {"action_type": "submit_audit", "audit_payload": {"flaws": [{"flaw_type":"...","location":"...","description":"..."}]}}

        Flaw types: wrong_statistical_test, underpowered_sample,
        undisclosed_exclusion, p_value_manipulation

        Submit when you have all 4 flaws. Respond ONLY with valid JSON.
    """).strip(),

    "task2_replication": textwrap.dedent("""
        You are a data scientist replicating an ML experiment.
        Follow the methods section exactly and report AUC-ROC and F1-score.

        Available actions (respond with JSON only, no prose):
          {"action_type": "read_dataset"}
          {"action_type": "execute_code", "code": "<python code>"}
          {"action_type": "submit_results", "results_payload": {"auc": 0.0, "f1": 0.0, "interpretation": "..."}}

        Use stratified train_test_split, StandardScaler, LogisticRegression
        with class_weight='balanced', random_state=42. Dataset path: DATASET_PATH.
        Respond ONLY with valid JSON.
    """).strip(),

    "task3_claim_verify": textwrap.dedent("""
        You are a statistical auditor verifying a paper's claim using raw data.

        Available actions (respond with JSON only, no prose):
          {"action_type": "read_dataset"}
          {"action_type": "execute_code", "code": "<python code>"}
          {"action_type": "flag_concern", "concern_type": "<type>", "evidence": "<text>"}
          {"action_type": "submit_verdict", "verdict_payload": {
              "verdict": "valid|partially_valid|invalid",
              "effect_size": 0.0,
              "p_value": 0.0,
              "justification": "<text with at least 50 words explaining your reasoning>"
          }}

        Run your own t-test. Check if claimed n matches dataset rows.
        Look for undisclosed exclusions. Respond ONLY with valid JSON.
    """).strip(),
}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str, env: ResearchIntegrityEnv) -> dict:
    """Run a single task and return the result dict."""
    obs = env.reset(task_id=task_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
        {"role": "user", "content": f"PAPER:\n{obs.paper_text}\n\nBegin your analysis."},
    ]

    final_reward = 0.0
    grader_score = 0.0
    steps_taken = 0

    for step_num in range(MAX_STEPS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=800,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  LLM call failed: {e}", file=sys.stderr)
            raw = '{"action_type": "submit_audit", "audit_payload": {"flaws": []}}'

        messages.append({"role": "assistant", "content": raw})

        action = _parse_action(raw, task_id)
        if action is None:
            messages.append({
                "role": "user",
                "content": "Invalid JSON. Respond with a valid action JSON only.",
            })
            continue

        obs, reward, done, info = env.step(action)
        steps_taken += 1
        final_reward = reward.total

        if reward.grader_score is not None:
            grader_score = reward.grader_score

        if done:
            break

        # Feed observation back to agent
        feedback_parts = [f"Step {step_num + 1} result:"]
        if obs.code_result:
            feedback_parts.append(f"Code output:\n{obs.code_result[:1000]}")
        if obs.flags_raised:
            feedback_parts.append(f"Flags raised so far: {obs.flags_raised}")
        feedback_parts.append(f"Step reward: {reward.step_reward}")
        feedback_parts.append(f"Steps remaining: {MAX_STEPS - step_num - 1}")
        feedback_parts.append("Continue your analysis or submit when ready.")

        messages.append({"role": "user", "content": "\n".join(feedback_parts)})

    return {
        "task_id": task_id,
        "grader_score": round(grader_score, 4),
        "final_reward": round(final_reward, 4),
        "steps_taken": steps_taken,
    }


def _parse_action(raw: str, task_id: str) -> Action | None:
    """Parse agent JSON into an Action object. Returns None on failure."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())

        atype = data.get("action_type")

        if atype == "read_section":
            return Action(action_type=ActionType.read_section, section=data.get("section"))

        if atype == "read_dataset":
            return Action(action_type=ActionType.read_dataset)

        if atype == "execute_code":
            return Action(action_type=ActionType.execute_code, code=data.get("code", ""))

        if atype == "flag_flaw":
            return Action(
                action_type=ActionType.flag_flaw,
                flaw_type=data.get("flaw_type", ""),
                location=data.get("location", ""),
                description=data.get("description", ""),
            )

        if atype == "flag_concern":
            return Action(
                action_type=ActionType.flag_concern,
                concern_type=data.get("concern_type", ""),
                evidence=data.get("evidence", ""),
            )

        if atype == "submit_audit":
            ap = data.get("audit_payload", {})
            flaws = [FlawReport(**f) for f in ap.get("flaws", [])]
            return Action(
                action_type=ActionType.submit_audit,
                audit_payload=SubmitAuditPayload(flaws=flaws),
            )

        if atype == "submit_results":
            rp = data.get("results_payload", {})
            return Action(
                action_type=ActionType.submit_results,
                results_payload=SubmitResultsPayload(
                    auc=float(rp.get("auc", 0)),
                    f1=float(rp.get("f1", 0)),
                    interpretation=rp.get("interpretation", ""),
                ),
            )

        if atype == "submit_verdict":
            vp = data.get("verdict_payload", {})
            return Action(
                action_type=ActionType.submit_verdict,
                verdict_payload=SubmitVerdictPayload(
                    verdict=Verdict(vp.get("verdict", "invalid")),
                    effect_size=float(vp.get("effect_size", 0)),
                    p_value=float(vp.get("p_value", 0.5)),
                    justification=vp.get("justification", ""),
                ),
            )

    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Research Integrity Gym Inference Script")
    parser.add_argument("--output-json", action="store_true",
                        help="Print JSON output (for automated evaluation)")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: No API key found. Set HF_TOKEN, API_KEY, or GROQ_API_KEY.", file=sys.stderr)
        sys.exit(1)

    if not args.output_json:
        print(f"Using API: {API_BASE_URL}")
        print(f"Model: {MODEL_NAME}")

    # MANDATORY: Use OpenAI client with specified env vars
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    env = ResearchIntegrityEnv(seed=SEED)

    task_ids = [
        "task1_methodology_audit",
        "task2_replication",
        "task3_claim_verify",
    ]

    results = []
    for task_id in task_ids:
        if not args.output_json:
            print(f"\nRunning {task_id}...", flush=True)
        result = run_task(client, task_id, env)
        results.append(result)
        if not args.output_json:
            print(f"  Grader score : {result['grader_score']:.4f}")
            print(f"  Final reward : {result['final_reward']:.4f}")
            print(f"  Steps taken  : {result['steps_taken']}")

    avg = round(sum(r["grader_score"] for r in results) / len(results), 4)
    output = {
        "model": MODEL_NAME,
        "seed": SEED,
        "tasks": results,
        "avg_grader_score": avg,
    }

    if args.output_json:
        print(json.dumps(output))
    else:
        print(f"\n{'='*40}")
        print(f"Average grader score: {avg:.4f}")
        print(f"{'='*40}")


if __name__ == "__main__":
    main()
