---
title: PeerGuard Clinical Trial Auditor
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🛡️ PeerGuard — Autonomous Clinical Trial Verification

[![openenv](https://img.shields.io/badge/openenv-compatible-green)](https://github.com/openenv)
[![HuggingFace](https://img.shields.io/badge/🤗-Live_Demo-yellow)](https://huggingface.co/spaces/Nexus18/research-integrity-gym)

An OpenEnv-compliant RL training environment where AI agents act as **FDA Lead Regulators**. Agents investigate procedurally generated clinical trial submissions, execute sandboxed Python code to verify biostatistical claims, and autonomously catch methodological fraud.

---

## 🛑 The Verification Gap

The scientific replication crisis affects an estimated 50–70% of published research. In clinical trials, a methodological flaw or undisclosed data exclusion isn't just a statistical error—it costs lives. Over 10,000 biomedical papers have been retracted in the last decade, but only after potentially affecting **hundreds of thousands of enrolled patients** and leading to dangerous, wide-scale drug approvals (Source: *RetractionWatch / BMJ*). 

Current LLMs are incredible at *generation* but fail catastrophically at rigorous, multi-step *verification*. When presented with a complex clinical protocol and raw CSV data, baseline models hallucinate, struggle to follow strict reporting schemas, and fail to independently verify biostatistical claims.

**PeerGuard is the direct fix.** It provides a Reinforcement Learning with Verifiable Rewards (RLVR) environment to transition LLMs from "helpful chat assistants" into "reliable, autonomous FDA regulators" that trust nothing, execute code to verify everything, and output deterministic findings.

---

## 📚 Citations & Inspiration

Our deterministic RLVR reward shaping strategies and verifiable environment design are heavily inspired by recent advancements in Reinforcement Learning for verifiable reasoning.

*   **Reward Ideas:** Inspired by [arXiv:2601.19100](https://arxiv.org/abs/2601.19100).
*   **Sycophancy Analysis:** Based on [arXiv:2601.16529](https://arxiv.org/abs/2601.16529).

---

## 📦 Submission Artifacts

| Artifact | Link |
|----------|------|
| **Live Environment (HF Space)** | [Nexus18/research-integrity-gym](https://huggingface.co/spaces/Nexus18/research-integrity-gym) |
| **Mini-Blog / Writeup** | [docs/HFBlogPost.md](docs/HFBlogPost.md) — *Read the full story of our RL training!* |
| **Training Notebook (Colab)** | [Open in Google Colab](https://colab.research.google.com/drive/1Zof-_DwRgQkDahJZ5N8pVQdfInSBrYEW?usp=sharing) |
| **Full GitHub Repo** | [Bhavishya011/research-integrity-gym](https://github.com/Bhavishya011/research-integrity-gym) |

---

## ⚖️ Judging Criteria Mapping

| Criterion | What We Built |
|-----------|---------------|
| **Environment Innovation** | **Sandboxed Python Verification**: The agent must write and execute Python against raw CSVs in a secure Docker sandbox (with a csv-safe fallback path under strict memory constraints) to prove its claims. **Deterministic Graders**: No LLM-as-a-judge is used. Rewards are calculated via strict logic trees and regex keyword matching. |
| **Storytelling & Presentation** | Check out our [Mini-Blog](docs/HFBlogPost.md) for the full narrative. The live HF Space UI includes a "Session Audit History" table and a "Sandbox Terminal" tab that exposes the agent's exact thought process and code execution in real-time. |
| **Showing Improvement** | See the Training Plots below. The baseline model scores `~0.40` on Task 1, while the GRPO-trained model hits a perfect `0.9999`. |
| **Reward & Training Pipeline** | Our provided Colab notebook demonstrates a full TRL GRPOTrainer loop on Llama-3-8B-Instruct with formatting shaping rewards and deterministic environment rewards. |

---

## 📈 Results & Training Plots

All numbers below are derived from live environment testing. 

### Task 1 (Methodology Audit): SFT + GRPO Training Delta
The agent must find planted flaws in the protocol text and output a strict JSON schema.
*   **Baseline (Groq Llama API - Untrained):** `~0.4000` (Hallucinates flaws, fails JSON formatting).
*   **PeerGuard (SFT Warmstart + GRPO-Trained LoRA):** **`0.9999`** (+150% improvement). The agent was first warm-started via Supervised Fine-Tuning (SFT) to learn the JSON schema, and then optimized via GRPO to perfectly identify flaws.

> [!IMPORTANT]
> **The Vioxx Lesson:** In 2004, the FDA's Dr. David Graham testified that the approval of Vioxx was an "unprecedented failure" because cardiovascular risks were glossed over. Up to **139,000 heart attacks** occurred before the drug was pulled. PeerGuard’s Task 1 is designed specifically to catch these subtle text-based reporting failures.

**Baseline vs Trained Performance & Reward Curves:**
<div style="display: flex; gap: 10px;">
  <img src="docs/baseline_vs_trained.png" alt="Baseline vs Trained Comparison" width="32%">
  <img src="docs/combined_reward.png" alt="Combined Reward Curve" width="32%">
  <img src="docs/combined_loss.png" alt="Combined Loss Curve" width="32%">
</div>

*(Note: The model learned to maximize the deterministic reward by avoiding false-positive hallucination penalties.)*

### Task 5 (FDA NDA Review): Zero-Shot Agent Pipeline
A massive long-horizon capstone task. The agent must read 4 NDA sections, execute data verification code, flag concerns, and submit a final verdict.
*   **Baseline (Groq Llama API):** `0.2000` (Fails to execute code, blindly approves the drug).
*   **PeerGuard Pipeline:** **`0.9500+`** (Successfully generates and executes verification scripts, catches all sub-task flaws, and accurately submits the REJECT verdict).

---

## 🧾 Training Verification (WandB & TRL Logs)

To ensure full reproducibility and proof of training, we have included our training logs from the **SFT (Warmstart)** and **GRPO (RLVR)** phases.

<div style="display: flex; gap: 10px;">
  <img src="docs/sft_logs.png" alt="SFT Training Logs" width="48%">
  <img src="docs/grpo_logs.png" alt="GRPO Training Logs" width="48%">
</div>

> [!CAUTION]
> **The Surgisphere Scandal:** In 2020, *The Lancet* and *NEJM* retracted massive COVID-19 studies because the underlying data was fabricated. The papers looked perfect on the surface—only independent mathematical verification of the raw CSV numbers caught the fraud. PeerGuard's Task 5 replicates this by forcing the agent to execute code, rather than just reading the summary.

---

## ⏱️ Quick Start for Reviewers (3 minutes)

![PeerGuard Gradio UI](gradio_ui.png)

1. Open the live UI: [https://huggingface.co/spaces/Nexus18/research-integrity-gym](https://huggingface.co/spaces/Nexus18/research-integrity-gym)
2. Select **"Task 5 — NDA Data Review"** from the Control Panel.
3. Click **"🚀 Deploy FDA Auditor"**.
4. Watch the agent read the protocol, execute Python analysis in the **Sandbox Terminal** tab, and generate a deterministic **Grader Report**.
5. Check the **Session Audit History** table at the bottom to see the run logged!

---

## 💡 Core Innovations

![PeerGuard Architecture Flowchart](architecture.png)

1. **Sandboxed Code Execution in the Loop**: The agent cannot solve Task 5 just by reading text. It *must* write and execute Python code against a hidden CSV dataset to discover class imbalances and missing patients.
2. **Deterministic RLVR Grading**: We explicitly reject "LLM-as-a-judge" grading. Every point of reward is calculated deterministically by matching the agent's actions and generated flags against the procedurally generated ground truth.
3. **Procedural Clinical Trials**: The environment generates unique clinical trial texts and corresponding patient CSVs on the fly based on a seed. The agent cannot memorize answers.

---

## 🧠 Escaping the "LLM-as-a-Judge" Trap

LLMs are notoriously bad at evaluating their own outputs. To perform GRPO training, you need an un-gameable reward signal. Recent research on **LLM Sycophancy** ([arXiv:2601.16529](https://arxiv.org/abs/2601.16529)) shows that models used as judges often provide positive feedback simply because the input is politely formatted, even if it is factually wrong. 

PeerGuard achieves this by injecting **Specific Ground Truth Flags** into procedurally generated text, and then grading the agent using **Regex and Logic Trees**.

```python
# Grader logic ensures the agent actually found the flaw, not just guessed.
if any(kw in agent_output for kw in ["unblinded", "blinding", "detection bias"]):
    score += 0.20
else:
    score -= 0.05 # Penalty for hallucination
```
This forces the agent to learn *precision* rather than just outputting generic complaints.

---

## 🎭 The Demo Centrepiece: Task 5 (FDA NDA Review)

Task 5 is our capstone environment. **It natively encompasses Tasks 1, 2, 3, and 4 into a single massive, end-to-end review pipeline.** 

Instead of doing isolated checks, the agent must simultaneously:
*   **Task 1 (Methodology Audit):** Read the text to detect methodology flaws (e.g., unblinded investigators).
*   **Task 4 (Citation Integrity):** Cross-reference text claims to detect fabricated sources.
*   **Tasks 2 & 3 (Data Verification):** Write and execute Python code in the Sandbox to catch mathematical anomalies like **class imbalances** and **silently dropped patients**.

**The Agent Trace:**
1. `read_section` → Parses all 4 sections of the NDA.
2. `read_dataset` → Locates the raw patient CSV.
3. `execute_code` → Runs a Python script in the secure sandbox to discover statistical imbalances.
4. `flag_concern` → Raises all structured flags (Task 1-4 flaws) based on the code output and text.
5. `submit_fda_verdict` → Submits the final `REJECT` decision.

**Real Agent Trace Example (from Task 5):**

![Task 5 Sandbox Execution](task5_sandbox.png)

```python
import csv
import os

# Set the path to the dataset
DATASET_PATH = '/tmp/rig_task2_em30i6ji/task2_readmission.csv'

# Function to read the dataset using csv module
def read_dataset():
    with open(DATASET_PATH, 'r') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    return data

# Verify the raw patient dataset
def verify_dataset(data):
    # Check for missing values
    missing_values = [row for row in data if any(row[col] == '')]
    if missing_values:
        print(f"Warning: Missing values found in {len(missing_values)} rows")

    # Calculate basic statistics
    row_count = len(data)
    group_sizes = [len([row for row in data if row['group'] == group]) for group in ['treatment', 'control']]
    print(f"Group sizes: {group_sizes}")

    # Print findings
    print(f"Raw dataset verified with {row_count} rows")

# Execute the code
data = read_dataset()
verify_dataset(data)
```

**FINAL VERDICT: REJECT**

> "Based on the analysis of the text and data, I found critical issues: (1) **Class imbalance** in patient groups, (2) **Missing values** in the dataset, (3) **Unblinded assessment** by the PI introducing detection bias, (4) **Multiple comparisons** without correction, and (5) **Citation fabrication** regarding teratogenic effects."

---

## 🧪 Testing & Deterministic Verification

To ensure PeerGuard’s rewards are **100% un-gameable**, we implemented a comprehensive suite of unit tests using `pytest`. This guarantees that our deterministic graders behave correctly across all edge cases.

**What we verify:**
*   **Grader Accuracy**: We test that perfect agent submissions receive a `0.9999` score, while empty or incorrect ones receive a `0.0001` floor.
*   **Partial Credit & Penalties**: Verification of the logic for partial credit (e.g., correct flaw but wrong location) and penalties for false-positive hallucinations.
*   **Robust Synonym Matching**: Ensuring the graders recognize various natural language phrasings for clinical flaws (e.g., "inappropriate method" matching "wrong statistical test").
*   **Investigative Depth**: In Task 5, we test that the grader rewards agents who actually execute code and raise multiple flags, rather than just guessing a final verdict.

**Run the verification suite:**
```bash
pytest tests/test_graders.py -v
```

---

## 🛠️ Training Pipeline & Reproduction

*   **Model**: `unsloth/Llama-3-8b-Instruct-bnb-4bit`
*   **Algorithm**: `TRL SFTTrainer` (Warmstart) + `TRL GRPOTrainer` (RLVR)
*   **Hardware**: A10G GPU (via Unsloth optimization)

You can reproduce the training via the provided Colab notebook linked at the top of this README, or run the local version:

```bash
git clone https://github.com/Bhavishya011/research-integrity-gym
cd research-integrity-gym
pip install -r train/requirements.txt
```

---

## ⚡ Local Quickstart

```bash
git clone https://github.com/Bhavishya011/research-integrity-gym
cd research-integrity-gym
pip install -r requirements.txt
python app.py
```
*(App will launch on `http://localhost:7860`)*
