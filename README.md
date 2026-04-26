# 🛡️ PeerGuard — Autonomous Clinical Trial Verification

[![openenv](https://img.shields.io/badge/openenv-compatible-green)](https://github.com/openenv)
[![HuggingFace](https://img.shields.io/badge/🤗-Live_Demo-yellow)](https://huggingface.co/spaces/Nexus18/research-integrity-gym)

An OpenEnv-compliant RL training environment where AI agents act as **FDA Lead Regulators**. Agents investigate procedurally generated clinical trial submissions, execute sandboxed Python code to verify biostatistical claims, and autonomously catch methodological fraud.

---

## 🛑 The Verification Gap

The scientific replication crisis affects an estimated 50–70% of published research. In clinical trials, a methodological flaw or undisclosed data exclusion isn't just a statistical error—it costs lives.

Current LLMs are incredible at *generation* but fail catastrophically at rigorous, multi-step *verification*. When presented with a complex clinical protocol and raw CSV data, baseline models hallucinate, struggle to follow strict reporting schemas, and fail to independently verify biostatistical claims.

**PeerGuard is the direct fix.** It provides a Reinforcement Learning with Verifiable Rewards (RLVR) environment to transition LLMs from "helpful chat assistants" into "reliable, autonomous FDA regulators" that trust nothing, execute code to verify everything, and output deterministic findings.

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
| **Environment Innovation** | **Sandboxed Python Verification**: The agent must write and execute pandas code against raw CSVs in a secure Docker sandbox to prove its claims. **Deterministic Graders**: No LLM-as-a-judge is used. Rewards are calculated via strict logic trees and regex keyword matching. |
| **Storytelling & Presentation** | Check out our [Mini-Blog](docs/HFBlogPost.md) for the full narrative. The live HF Space UI includes a "Session Audit History" table and a "Sandbox Terminal" tab that exposes the agent's exact thought process and code execution in real-time. |
| **Showing Improvement** | See the Training Plots below. The baseline model scores `~0.40` on Task 1, while the GRPO-trained model hits a perfect `0.9999`. |
| **Reward & Training Pipeline** | Our provided Colab notebook demonstrates a full TRL GRPOTrainer loop on Llama-3-8B-Instruct with formatting shaping rewards and deterministic environment rewards. |

---

## 📈 Results & Training Plots

All numbers below are derived from live environment testing. 

### Task 1 (Methodology Audit): GRPO Training Delta
The agent must find planted flaws in the protocol text and output a strict JSON schema.
*   **Baseline (Untrained Llama-3-8B):** `~0.4000` (Hallucinates flaws, fails JSON formatting).
*   **PeerGuard (GRPO-Trained LoRA):** **`0.9999`** (+150% improvement). The agent perfectly identifies flaws and strictly adheres to the schema.

**GRPO Reward Curve & Loss Curve:**
<div style="display: flex; gap: 10px;">
  <img src="docs/grpo_reward_curve.png" alt="GRPO Reward Curve" width="49%">
  <img src="docs/grpo_loss_curve.png" alt="SFT Loss Curve" width="49%">
</div>

*(Note: The model learned to maximize the deterministic reward by avoiding false-positive hallucination penalties.)*

### Task 5 (FDA NDA Review): Zero-Shot Agent Pipeline
A massive long-horizon capstone task. The agent must read 4 NDA sections, execute data verification code, flag concerns, and submit a final verdict.
*   **Baseline:** `0.2000` (Fails to execute code, blindly approves the drug).
*   **PeerGuard Pipeline:** **`0.9500+`** (Successfully generates and executes verification scripts, catches all sub-task flaws, and accurately submits the REJECT verdict).

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

LLMs are notoriously bad at evaluating their own outputs. To perform GRPO training, you need an un-gameable reward signal. 

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
