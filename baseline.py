"""
Baseline inference script — OpenEnv spec requirement.

Uses Groq's Llama 3.3 70B via OpenAI-compatible endpoint.
Reads GROQ_API_KEY from environment variables.

Usage:
  python baseline.py              # human-readable scores
  python baseline.py --output-json # JSON output (used by /baseline endpoint)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ResearchIntegrityEnv
from env.models import (
    Action, ActionType,
    FlawReport, SubmitAuditPayload,
    SubmitResultsPayload, SubmitVerdictPayload, Verdict,
)

# HuggingFace router — required by OpenEnv submission spec
HF_ROUTER_URL = "https://router.huggingface.co/v1"
MODEL         = "meta-llama/Llama-3.3-70B-Instruct"
MAX_STEPS     = 15
SEED          = 42


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
              "justification": "<text>"
          }}

        Run your own t-test. Check if claimed n matches dataset rows.
        Look for undisclosed exclusions. Respond ONLY with valid JSON.
    """).strip(),

    "task4_citation_check": textwrap.dedent("""
        You are a research integrity auditor. A paper cites sources but one
        citation is fabricated — the paper's claim doesn't match the excerpt.

        Available actions (respond with JSON only, no prose):
          {"action_type": "read_section", "section": "citations"}
          {"action_type": "check_citation", "citation_id": <int>}
          {"action_type": "flag_fabrication", "citation_id": <int>, "flaw_type": "<type>", "description": "<evidence>"}
          {"action_type": "submit_report", "report_payload": {
              "fabricated_citation_id": <int>,
              "fabrication_type": "<directional|magnitude|population|significance|absent>",
              "verified_correct_citations": [<int>, <int>],
              "evidence": "<quote showing mismatch>"
          }}

        Fabrication types explained:
          - directional: claim says "increased" but source says "decreased" (or vice versa)
          - magnitude: wrong numbers (e.g., "180%" vs "18%", "25%" vs "2.5%")
          - population: wrong demographic (e.g., "adults" vs "children")
          - significance: wrong p-value or significance claim
          - absent: claim not mentioned in source at all

        Strategy: Read citations section first, compare each citation to paper claims,
        identify the fabricated one, verify the others are correct, then submit_report.
        Respond ONLY with valid JSON.
    """).strip(),
}


def run_task(client: OpenAI, task_id: str, env: ResearchIntegrityEnv) -> dict:
    obs = env.reset(task_id=task_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
        {"role": "user",   "content": f"PAPER:\n{obs.paper_text}\n\nBegin your analysis."},
    ]

    final_reward = 0.0
    grader_score = 0.0
    steps_taken  = 0

    for step_num in range(MAX_STEPS):
        response = client.chat.completions.create(
            model       = MODEL,
            messages    = messages,
            temperature = 0.0,
            max_tokens  = 800,
        )
        raw = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": raw})

        action = _parse_action(raw)
        if action is None:
            messages.append({
                "role": "user",
                "content": "Invalid JSON. Respond with a valid action JSON only.",
            })
            continue

        obs, reward, done, info = env.step(action)
        steps_taken  += 1
        final_reward  = reward.total

        if reward.grader_score is not None:
            grader_score = reward.grader_score

        if done:
            break

        parts = [f"Step {step_num + 1} result:"]
        if obs.code_result:
            parts.append(f"Code output:\n{obs.code_result[:1000]}")
        if obs.flags_raised:
            parts.append(f"Flags raised: {obs.flags_raised}")
        parts.append(f"Step reward: {reward.step_reward}")
        parts.append(f"Steps remaining: {MAX_STEPS - step_num - 1}")
        parts.append("Continue or submit when ready.")
        messages.append({"role": "user", "content": "\n".join(parts)})

    return {
        "task_id":      task_id,
        "grader_score": round(grader_score, 4),
        "final_reward": round(final_reward, 4),
        "steps_taken":  steps_taken,
    }


def _parse_action(raw: str) -> Action | None:
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
        if atype == "check_citation":
            return Action(
                action_type=ActionType.check_citation,
                citation_id=int(data.get("citation_id", 1)),
            )
        if atype == "flag_fabrication":
            return Action(
                action_type=ActionType.flag_fabrication,
                citation_id=int(data.get("citation_id", 1)),
                flaw_type=data.get("flaw_type", ""),
                description=data.get("description", ""),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", action="store_true",
                        help="Output JSON for /baseline endpoint")
    args = parser.parse_args()

    api_key = os.environ.get("HF_TOKEN", "")
    if not api_key:
        print("ERROR: HF_TOKEN not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key  = api_key,
        base_url = HF_ROUTER_URL,
    )
    env = ResearchIntegrityEnv(seed=SEED)

    task_ids = [
        "task1_methodology_audit",
        "task2_replication",
        "task3_claim_verify",
        "task4_citation_check",
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
    output = {"model": MODEL, "seed": SEED, "tasks": results, "avg_grader_score": avg}

    if args.output_json:
        print(json.dumps(output))
    else:
        print(f"\n{'='*40}\nAverage grader score: {avg:.4f}\n{'='*40}")


if __name__ == "__main__":
    main()
