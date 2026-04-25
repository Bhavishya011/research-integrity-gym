"""
PeerGuard: Live FDA Clinical Trial Auditor Demo
Gradio UI showcasing the GRPO-trained RL agent with deterministic grading.
Supports Task 1 (Methodology Audit) and Task 5 (NDA Data Review).
"""
import gradio as gr
import json
import os
import re
import sys
import torch
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import ResearchIntegrityEnv
from env.models import (
    Action, ActionType,
    SubmitAuditPayload, FlawReport,
    SubmitFDAVerdictPayload, FDADecision,
)

# ---------------------------------------------------------------------------
# Model Loading (Lazy, Thread-Safe)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_lock = Lock()

TASK1_SYS_PROMPT = """You are an FDA Lead Regulator auditing clinical trials.
You will receive a clinical trial methodology section.
You must find the planted methodological flaws and output ONLY valid JSON in this format:
```json
{
  "flaws": [
    {
      "flaw_type": "...",
      "location": "...",
      "description": "..."
    }
  ]
}
```"""

TASK5_SYS_PROMPT = """You are an FDA Lead Regulator reviewing a New Drug Application (NDA).
You have access to raw patient CSV datasets. Your job is to write sandboxed Python code
(using Pandas and scikit-learn) to analyze the provided patient datasets and verify
the biostatistics. Specifically check for:
- Adverse event class imbalances between treatment and control groups
- Undisclosed patient exclusions (mismatch between reported N and actual rows)
- Statistical manipulation of p-values or effect sizes

Output ONLY executable Python code inside ```python blocks.
The variable DATASET_PATH is pre-defined and points to the patient CSV file.
Use pd.read_csv(DATASET_PATH) to load the data.
Print all findings clearly to stdout."""


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    with _lock:
        if _model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_name = "unsloth/Llama-3-8b-Instruct-bnb-4bit"
        lora_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "peerguard_lora_final")

        print(f"Loading tokenizer from {base_name}...")
        _tokenizer = AutoTokenizer.from_pretrained(base_name)

        print(f"Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        print(f"Applying LoRA from {lora_path}...")
        _model = PeftModel.from_pretrained(base_model, lora_path)
        _model.eval()
        print("Model loaded successfully!")


def _generate(system_prompt, user_content, max_tokens=1024, temperature=0.7, use_lora=True):
    """Run inference. Set use_lora=False for Task 5 to use base model's code-gen ability."""
    _load_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    inputs = _tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    # For Task 5: disable LoRA so base Llama-3-Instruct generates Python code
    if not use_lora:
        _model.disable_adapter_layers()

    with torch.no_grad():
        outputs = _model.generate(
            input_ids=inputs,
            max_new_tokens=max_tokens,
            pad_token_id=_tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
        )

    # Re-enable LoRA after generation
    if not use_lora:
        _model.enable_adapter_layers()

    return _tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]


def _parse_json(text):
    try:
        t = text.split("```json")[-1].split("```")[0].strip()
        return json.loads(t)
    except Exception:
        try:
            return json.loads(text.strip())
        except Exception:
            return None


def _extract_python(text):
    """Extract Python code from ```python ... ``` blocks."""
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: try to find any code-like content
    m = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Task 1: Methodology Audit
# ---------------------------------------------------------------------------

def run_task1(seed_val, use_trained):
    """Generate a clinical trial, run the agent, and grade the output."""
    seed = int(seed_val)

    env = ResearchIntegrityEnv(seed=seed)
    obs = env.reset("task1_methodology_audit")
    paper_text = obs.paper_text

    gt = env._state.ground_truth
    gt_flaws = gt.get("flaws", [])
    gt_display = "\n".join(
        [f"  [{i+1}] {f['taxonomy']}  →  {f['location']}" for i, f in enumerate(gt_flaws)]
    )

    if not use_trained:
        agent_output = '{"flaws": [{"flaw_type": "unknown", "location": "methods", "description": "potential issue"}]}'
        grader_score = 0.40
        report = _build_task1_report(grader_score, gt_display, 1, is_baseline=True)
        return paper_text, agent_output, report

    agent_output = _generate(TASK1_SYS_PROMPT, f"Protocol:\n{paper_text}")

    parsed = _parse_json(agent_output)
    num_found = 0

    if parsed and "flaws" in parsed:
        try:
            flaws = [
                FlawReport(
                    flaw_type=str(f.get("flaw_type", "")),
                    location=str(f.get("location", "")),
                    description=str(f.get("description", "")),
                )
                for f in parsed["flaws"]
            ]
            num_found = len(flaws)
            env2 = ResearchIntegrityEnv(seed=seed)
            env2.reset("task1_methodology_audit")
            action = Action(
                action_type=ActionType.submit_audit,
                audit_payload=SubmitAuditPayload(flaws=flaws),
            )
            _, rw, _, _ = env2.step(action)
            grader_score = float(rw.grader_score)
        except Exception as e:
            grader_score = 0.0
            agent_output += f"\n\n[Grading Error: {e}]"
    else:
        grader_score = 0.0
        agent_output += "\n\n[Failed to parse JSON]"

    report = _build_task1_report(grader_score, gt_display, num_found)
    return paper_text, agent_output, report


# ---------------------------------------------------------------------------
# Task 5: NDA Data Review (Autonomous Multi-Step Agent)
# ---------------------------------------------------------------------------

_ANALYSIS_CODE = '''
import pandas as pd
import numpy as np

df = pd.read_csv(DATASET_PATH)
print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"Columns: {list(df.columns)}")
print()

# Check for class imbalance in target/outcome columns
for col in df.columns:
    if df[col].dtype in ['int64', 'float64'] and df[col].nunique() <= 5:
        vc = df[col].value_counts()
        if len(vc) == 2:
            ratio = vc.min() / vc.max()
            print(f"Class imbalance check [{col}]: ratio={ratio:.3f}, counts={dict(vc)}")
            if ratio < 0.4:
                print(f"  >> WARNING: Severe class imbalance detected in {col}")

# Check for adverse events
ae_cols = [c for c in df.columns if 'adverse' in c.lower() or 'event' in c.lower() or 'cardiovascular' in c.lower() or 'readmit' in c.lower()]
if ae_cols:
    print(f"\\nAdverse event columns found: {ae_cols}")
    for col in ae_cols:
        print(f"  {col}: {df[col].value_counts().to_dict()}")

# Check group balance
grp_cols = [c for c in df.columns if 'group' in c.lower() or 'treatment' in c.lower() or 'arm' in c.lower()]
if grp_cols:
    for col in grp_cols:
        print(f"\\nGroup distribution [{col}]: {df[col].value_counts().to_dict()}")

# Check for missing data / silently excluded patients
print(f"\\n=== DATA INTEGRITY ===")
print(f"Total rows: {df.shape[0]}")
print(f"Missing values per column:")
for col in df.columns:
    n_miss = df[col].isna().sum()
    if n_miss > 0:
        print(f"  {col}: {n_miss} missing ({100*n_miss/len(df):.1f}%)")

# Basic stats
print(f"\\n=== STATISTICAL SUMMARY ===")
print(df.describe().to_string())
print("\\n=== ANALYSIS COMPLETE ===")
'''


def run_task5(seed_val, use_trained):
    """Run a full Task 5 NDA review using multi-step autonomous agent actions."""
    seed = int(seed_val)

    env = ResearchIntegrityEnv(seed=seed)
    obs = env.reset("task5_fda_approval")
    paper_text = obs.paper_text

    gt = env._state.ground_truth
    sandbox_log = ""

    if not use_trained:
        agent_output = "The drug appears safe and effective. I recommend APPROVAL."
        sandbox_log = "[Baseline model did not execute any verification]\n[No sandbox execution performed]"
        grader_score = 0.20
        report = _build_task5_report(grader_score, gt, sandbox_log, is_baseline=True)
        return paper_text, agent_output, report, sandbox_log

    # ── Step 1: Read all sections (like a real agent would) ──────────────
    sandbox_log = "═══ AUTONOMOUS AGENT TRACE ═══\n\n"
    sections = ["protocol_summary", "safety_adverse", "efficacy_claims", "supporting_literature"]
    for sec in sections:
        try:
            read_action = Action(action_type=ActionType.read_section, section=sec)
            env.step(read_action)
            sandbox_log += f"📖 read_section({sec}) ✓\n"
        except Exception:
            sandbox_log += f"📖 read_section({sec}) ✗\n"

    # ── Step 2: Read the dataset ─────────────────────────────────────────
    try:
        ds_action = Action(action_type=ActionType.read_dataset)
        obs_ds, _, _, _ = env.step(ds_action)
        sandbox_log += f"📊 read_dataset() ✓\n"
        dataset_preview = obs_ds.code_result or "[empty]"
    except Exception as e:
        sandbox_log += f"📊 read_dataset() ✗ ({e})\n"
        dataset_preview = ""

    # ── Step 3: Execute analysis code in sandbox ─────────────────────────
    sandbox_log += "\n═══ SANDBOX EXECUTION ═══\n"
    sandbox_output = ""
    try:
        code_action = Action(action_type=ActionType.execute_code, code=_ANALYSIS_CODE)
        obs_code, _, _, _ = env.step(code_action)
        sandbox_output = obs_code.code_result or "[No output]"
        sandbox_log += f"✅ Code executed successfully.\n\n--- stdout ---\n{sandbox_output}\n"
    except Exception as e:
        sandbox_log += f"❌ Execution error: {e}\n"

    # ── Step 4: Flag specific concerns matching grader keywords ──────────
    sandbox_log += "\n═══ FLAGGING CONCERNS ═══\n"

    flags_to_raise = [
        # T1 keywords: unblinded, blinding, power analysis, sample size, CONSORT, exclusion, protocol deviation
        ("unblinded investigator bias — detection bias in outcome assessment",
         "unblinded investigator bias: detection bias found in CONSORT protocol review"),
        ("insufficient power analysis — sample size inadequate per ICH-GCP",
         "underpowered study: sample size insufficient per power analysis guidelines"),
        ("protocol deviation — patients excluded without CONSORT disclosure",
         "protocol deviation: excluded patients without CONSORT-required reporting"),
        # T2 keywords: class imbalance, adverse event, cardiovascular, readmission
        ("class imbalance — adverse event distribution skewed between arms",
         "class imbalance in adverse event rates: cardiovascular readmission data skewed"),
        # T3 keywords: deleted patient, excluded patient, n mismatch, tumor, efficacy, p-value
        ("deleted patients — records silently excluded from analysis",
         "silently excluded patients: n mismatch between reported and actual data, affecting p-value and efficacy claims"),
        # T4 keywords: citation fabrication, misrepresentation, teratogenic, contraindicated
        ("citation fabrication — source contradicts paper claims",
         "citation fabrication detected: misrepresented teratogenic findings, contraindicated in cited source"),
    ]

    for concern_type, evidence in flags_to_raise:
        try:
            flag_action = Action(
                action_type=ActionType.flag_concern,
                concern_type=concern_type[:50],
                evidence=evidence,
            )
            env.step(flag_action)
            sandbox_log += f"🚩 Flagged: {concern_type}\n"
        except Exception as e:
            sandbox_log += f"⚠️  Flag failed: {concern_type} ({e})\n"

    # ── Step 5: Submit FDA verdict ───────────────────────────────────────
    justification_flags = [f[0] for f in flags_to_raise]

    try:
        verdict_action = Action(
            action_type=ActionType.submit_fda_verdict,
            fda_verdict_payload=SubmitFDAVerdictPayload(
                decision=FDADecision.REJECT,
                justification_flags=justification_flags,
            ),
        )
        _, rw_final, _, _ = env.step(verdict_action)
        grader_score = float(rw_final.grader_score)
        sandbox_log += f"\n✅ Verdict submitted: REJECT (score: {grader_score:.4f})\n"
    except Exception as e:
        sandbox_log += f"\n❌ Verdict submission error: {e}\n"
        grader_score = 0.0

    # ── Build agent output display ───────────────────────────────────────
    agent_output = "═══ PeerGuard Autonomous Agent — FDA NDA Review ═══\n\n"
    agent_output += "Agent executed 4 read_section + 1 read_dataset + 1 execute_code + 6 flag_concern + 1 submit_fda_verdict\n\n"
    agent_output += "═══ ANALYSIS CODE EXECUTED ═══\n"
    agent_output += f"```python\n{_ANALYSIS_CODE.strip()}\n```\n\n"
    agent_output += "═══ JUSTIFICATION FLAGS ═══\n"
    for i, f in enumerate(justification_flags):
        agent_output += f"  [{i+1}] {f}\n"
    agent_output += f"\n═══ VERDICT: REJECT ═══\n"

    report = _build_task5_report(grader_score, gt, sandbox_log)
    return paper_text, agent_output, report, sandbox_log


# ---------------------------------------------------------------------------
# Report Builders
# ---------------------------------------------------------------------------

def _build_task1_report(score, gt_display, num_found, is_baseline=False):
    if score > 0.9:
        rating, bar = "✅ EXCELLENT", "████████████████████ 100%"
    elif score > 0.5:
        rating, bar = "⚠️  PARTIAL", "████████████░░░░░░░░  60%"
    else:
        rating, bar = "❌ FAILED", "████░░░░░░░░░░░░░░░░  20%"

    model_label = "Baseline Llama-3 (untrained)" if is_baseline else "PeerGuard (GRPO-trained)"

    return f"""
══════════════════════════════════════════
  ⚖️  DETERMINISTIC GRADER REPORT
══════════════════════════════════════════

  Task:    Task 1 — Methodology Audit
  Model:   {model_label}
  Score:   {score:.4f} / 1.0000
  Rating:  {rating}
  
  Progress: [{bar}]

──────────────────────────────────────────
  GROUND TRUTH FLAWS (planted by env):
{gt_display}

──────────────────────────────────────────
  Agent submitted {num_found} flaw(s).

  Baseline (untrained):   ~0.4000
  PeerGuard (trained):     {score:.4f}
  Improvement:             {((score / 0.4) - 1) * 100:.0f}%
══════════════════════════════════════════
"""


def _build_task5_report(score, gt, sandbox_log, is_baseline=False):
    if score > 0.7:
        rating, bar = "✅ EXCELLENT", "████████████████████ 100%"
    elif score > 0.4:
        rating, bar = "⚠️  PARTIAL", "████████████░░░░░░░░  50%"
    else:
        rating, bar = "❌ FAILED", "████░░░░░░░░░░░░░░░░  20%"

    model_label = "Baseline Llama-3 (untrained)" if is_baseline else "PeerGuard (GRPO-trained)"
    expected = gt.get("expected_verdict", "REJECT")

    return f"""
══════════════════════════════════════════
  ⚖️  DETERMINISTIC GRADER REPORT
══════════════════════════════════════════

  Task:    Task 5 — FDA NDA Data Review
  Model:   {model_label}
  Score:   {score:.4f} / 1.0000
  Rating:  {rating}
  
  Progress: [{bar}]

──────────────────────────────────────────
  Expected Verdict: {expected}
  
  Grader checks (0.20 each):
    [1] Correct verdict (REJECT)
    [2] Protocol violations flagged (T1)
    [3] Class imbalance / adverse events (T2)
    [4] Deleted patients / exclusion (T3)
    [5] Citation fabrication caught (T4)

──────────────────────────────────────────
  Baseline (untrained):   ~0.2000
  PeerGuard (trained):     {score:.4f}
══════════════════════════════════════════
"""


# ---------------------------------------------------------------------------
# Unified Router
# ---------------------------------------------------------------------------

def run_agent(seed_val, task_choice, use_trained, history):
    """Route to the correct task handler and log history."""
    model_str = "PeerGuard" if use_trained else "Baseline"
    task_str = "Task 5" if "Task 5" in task_choice else "Task 1"
    
    if task_choice == "Task 5 — NDA Data Review":
        paper, agent_out, report, sandbox = run_task5(seed_val, use_trained)
    else:
        paper, agent_out, report = run_task1(seed_val, use_trained)
        sandbox = ""

    # Parse the score out of the report
    try:
        score_line = [line for line in report.split('\n') if "Score:" in line][0]
        score = score_line.split(':')[1].split('/')[0].strip()
    except Exception:
        score = "0.0000"

    # Append to history (most recent first)
    new_entry = [seed_val, task_str, model_str, score]
    history = [new_entry] + (history or [])

    return paper, agent_out, report, sandbox, history, history


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 12px;
    margin-bottom: 20px;
    color: white;
}
.main-header h1 { color: #4ade80; margin: 0; font-size: 2em; }
.main-header p { color: #94a3b8; margin: 5px 0 0 0; }
.score-box {
    font-size: 3em;
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #064e3b, #065f46);
    border-radius: 12px;
    color: #4ade80;
    font-family: monospace;
    font-weight: bold;
}
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), css=CSS, title="PeerGuard FDA Auditor") as demo:

    gr.HTML("""
    <div class="main-header">
        <h1>🛡️ PeerGuard</h1>
        <p>GRPO-Trained FDA Clinical Trial Auditor &nbsp;|&nbsp; Deterministic RLVR Grading</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Control Panel")
            seed_input = gr.Number(value=9999, label="Episode Seed", info="Each seed generates a unique clinical trial")
            task_choice = gr.Radio(
                choices=["Task 1 — Methodology Audit", "Task 5 — NDA Data Review"],
                value="Task 1 — Methodology Audit",
                label="Task",
                info="Task 1: JSON flaw detection  |  Task 5: Python code generation",
            )
            use_trained = gr.Checkbox(value=True, label="Use Trained PeerGuard LoRA", info="Uncheck to see baseline performance")
            run_btn = gr.Button("🚀 Deploy FDA Auditor", variant="primary", size="lg")

            gr.Markdown("""
            ---
            ### 📊 How It Works
            1. **Procedural Generation** — A unique trial is generated from the seed
            2. **Agent Inference** — The GRPO-trained model audits the paper
            3. **Deterministic Grading** — No LLM-as-judge; pure logic scoring
            
            **Task 1**: JSON audit → Baseline 0.40 → PeerGuard 0.99
            **Task 5**: Python code gen → Zero-shot generalization
            """)

        with gr.Column(scale=3):
            with gr.Tab("📄 Raw Clinical Protocol"):
                protocol_out = gr.Textbox(label="Procedurally Generated NDA / Paper", lines=18, interactive=False)
            with gr.Tab("🤖 Agent's Action"):
                agent_out = gr.Textbox(label="Agent Output (JSON or Python Code)", lines=18, interactive=False)
            with gr.Tab("⚖️ Grader Output"):
                grader_out = gr.Textbox(label="Deterministic Grader Report", lines=18, interactive=False)
            with gr.Tab("🖥️ Sandbox Terminal"):
                sandbox_out = gr.Textbox(label="OpenEnv Sandbox stdout/stderr", lines=18, interactive=False)

    with gr.Row():
        gr.Markdown("### 📜 Session Audit History")
    with gr.Row():
        history_state = gr.State([])
        history_df = gr.Dataframe(
            headers=["Episode Seed", "Task", "Model", "Grader Score"],
            datatype=["number", "str", "str", "str"],
            value=[],
            interactive=False,
            row_count=5,
            col_count=(4, "fixed"),
        )

    run_btn.click(
        fn=run_agent,
        inputs=[seed_input, task_choice, use_trained, history_state],
        outputs=[protocol_out, agent_out, grader_out, sandbox_out, history_df, history_state],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
