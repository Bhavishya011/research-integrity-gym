"""
OpenEnv baseline inference script.

Environment variables (set by judge's evaluation system):
  API_BASE_URL - LLM API endpoint (default: HF router)
  MODEL_NAME   - Model identifier (default: Llama 3.3 70B)
  HF_TOKEN     - API token (optional, tries multiple env vars)

Usage:
  python inference.py              # human-readable scores
  python inference.py --output-json # JSON output
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
    SubmitResultsPayload, SubmitVerdictPayload, SubmitCitationReportPayload,
    Verdict,
)

# ---------------------------------------------------------------------------
# Environment variables with defaults (per OpenEnv spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

# Try multiple possible API key environment variables
API_KEY = (
    os.getenv("HF_TOKEN") or 
    os.getenv("API_KEY") or 
    os.getenv("GROQ_API_KEY") or
    os.getenv("OPENAI_API_KEY")
)

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

    "task4_citation_check": textwrap.dedent("""
        You are a citation integrity checker. A paper cites 3 sources, but ONE citation 
        is fabricated - the claim in the paper doesn't match what the cited source says.

        STRATEGY:
        1. Read the paper carefully - note what each citation claims
        2. Compare each claim to the citation excerpts provided
        3. Find the ONE citation where the claim contradicts the excerpt
        4. Submit your report identifying the fabricated citation

        Available actions (respond with JSON only):
          {"action_type": "read_section", "section": "citations"}
          {"action_type": "submit_report", "report_payload": {
              "fabricated_citation_id": 2,
              "fabrication_type": "directional - paper claims X increased but source says decreased",
              "verified_correct_citations": [1, 3],
              "evidence": "Quote the specific text showing the mismatch"
          }}

        Fabrication types to look for:
        - directional: paper says "increased" but source says "decreased" (or vice versa)
        - magnitude: paper claims 25% but source shows 2.5% (wrong numbers)
        - population: paper generalizes to children but source studied adults only
        - significance: paper claims p<0.05 but source shows p>0.05
        - absent: paper claims a finding the source never mentions

        Respond ONLY with valid JSON. No prose.
    """).strip(),
}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str, env: ResearchIntegrityEnv) -> dict:
    """Run a single task and return the result dict."""
    obs = env.reset(task_id=task_id)
    
    # Structured logging: [START]
    print(f"[START] task={task_id}", flush=True)
    
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
            print(f"LLM call failed: {e}", file=sys.stderr)
            # Return empty submission on API failure
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
        
        # Structured logging: [STEP]
        print(f"[STEP] task={task_id} step={steps_taken} reward={reward.step_reward:.4f}", flush=True)

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
    
    # Structured logging: [END]
    print(f"[END] task={task_id} score={grader_score:.4f} steps={steps_taken}", flush=True)
    
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

        if atype == "check_citation":
            return Action(
                action_type=ActionType.check_citation,
                citation_id=int(data.get("citation_id", 0)),
            )

        if atype == "flag_fabrication":
            return Action(
                action_type=ActionType.flag_fabrication,
                citation_id=int(data.get("citation_id", 0)),
                flaw_type=data.get("fabrication_type", ""),
                description=data.get("evidence", ""),
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

        if atype == "submit_report":
            rp = data.get("report_payload", {})
            return Action(
                action_type=ActionType.submit_report,
                report_payload=SubmitCitationReportPayload(
                    fabricated_citation_id=rp.get("fabricated_citation_id"),
                    fabrication_type=rp.get("fabrication_type", ""),
                    verified_correct_citations=rp.get("verified_correct_citations", []),
                    evidence=rp.get("evidence", ""),
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

    if not args.output_json:
        print(f"Using API: {API_BASE_URL}")
        print(f"Model: {MODEL_NAME}")
        print(f"API Key: {'Set' if API_KEY else 'Missing'}")

    # Validate API key
    if not API_KEY:
        print("ERROR: No API key found. Set HF_TOKEN, API_KEY, GROQ_API_KEY, or OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # Create OpenAI client with error handling
    # Use minimal parameters to avoid version/environment conflicts
    try:
        import httpx
        # Create a basic HTTP client with short timeouts to fail fast
        http_client = httpx.Client(
            timeout=httpx.Timeout(10.0, connect=5.0),  # Short timeouts
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL,
            http_client=http_client,
            max_retries=0,  # No retries for faster failure
        )
    except ImportError:
        # Fallback if httpx import fails
        try:
            client = OpenAI(
                api_key=API_KEY,
                base_url=API_BASE_URL,
                max_retries=0,
                timeout=10.0,
            )
        except Exception as e:
            print(f"ERROR: Failed to create OpenAI client: {e}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        # Final fallback - try absolute minimal client
        try:
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=10.0, max_retries=0)
        except Exception as e2:
            print(f"ERROR: Failed to create OpenAI client: {e2}", file=sys.stderr)
            print(f"Original error: {e}", file=sys.stderr)
            sys.exit(1)

    env = ResearchIntegrityEnv(seed=SEED)

    task_ids = [
        "task1_methodology_audit",
        "task2_replication",
        "task3_claim_verify",
        "task4_citation_check",
    ]

    results = []
    for task_id in task_ids:
        result = run_task(client, task_id, env)
        results.append(result)

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
        print(f"\nAverage grader score: {avg:.4f}")


if __name__ == "__main__":
    main()
