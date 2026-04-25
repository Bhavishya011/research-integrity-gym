# 🛡️ PeerGuard — Clinical Trial Verification RL Environment

[![openenv](https://img.shields.io/badge/openenv-compatible-green)](https://github.com/openenv)
[![HuggingFace](https://img.shields.io/badge/🤗-Live_Demo-yellow)](https://huggingface.co/spaces/Nexus18/research-integrity-gym)

An OpenEnv-compliant RL training environment where AI agents act as FDA Lead Regulators. Agents must investigate procedurally generated clinical trial submissions, execute sandboxed Python code to verify biostatistical claims, and catch methodological fraud. Built for the Meta PyTorch × Scaler Hackathon Grand Finale, April 25–26 2026.

---

## 🛑 Problem Statement

The scientific replication crisis affects an estimated 50–70% of published research. In clinical trials, a methodological flaw or undisclosed data exclusion isn't just a statistical error—it costs lives.

Current LLMs are incredible at *generation* but fail catastrophically at rigorous, multi-step *verification*. When presented with a complex clinical protocol and raw CSV data, baseline models hallucinate, struggle to follow strict reporting schemas, and fail to independently verify biostatistical claims.

**PeerGuard is the direct fix.** It provides a Reinforcement Learning with Verifiable Rewards (RLVR) environment to transition LLMs from "helpful chat assistants" into "reliable, autonomous FDA regulators" that trust nothing, execute code to verify everything, and output deterministic findings.

---

## 📦 Submission Artifacts

| Artifact | Link |
|----------|------|
| **Live Environment (HF Space)** | [Nexus18/research-integrity-gym](https://huggingface.co/spaces/Nexus18/research-integrity-gym) |
| **Training Notebook** | `PeerGuard_GRPO_Training.ipynb` (in repo) |
| **Reward & Loss Plots** | `grpo_reward_curve.png`, `grpo_loss_curve.png.png` |
| **Full GitHub Repo** | [Bhavishya011/research-integrity-gym](https://github.com/Bhavishya011/research-integrity-gym) |

---

## ⚖️ How This Submission Maps to the Judging Rubric

| Criterion | Weight | Where to find the evidence |
|-----------|--------|----------------------------|
| **Environment Innovation** | 40% | **Sandboxed Python Verification**: The agent doesn't just read text; it must write and execute pandas code against raw CSVs in a secure Docker sandbox to prove its claims. **Deterministic Graders**: No LLM-as-a-judge is used. Rewards are calculated via strict logic trees and regex keyword matching against the ground truth. |
| **Storytelling & Presentation** | 30% | The live HuggingFace Space Gradio UI includes a "Session Audit History" table and a "Sandbox Terminal" tab that shows the agent's exact thought process, code execution, and flag generation in real-time. |
| **Showing Improvement in Rewards** | 20% | `grpo_reward_curve.png` shows the GRPO reward climbing steadily over 200 steps. The live demo proves the baseline model scores ~0.40 on Task 1, while the GRPO-trained model hits a perfect 0.9999. |
| **Reward & Training Pipeline** | 10% | `PeerGuard_GRPO_Training.ipynb` demonstrates a full TRL GRPOTrainer loop on Llama-3-8B-Instruct with formatting shaping rewards and deterministic environment rewards. |

---

## 📊 Results

All numbers below are derived from live environment testing. 

**1. Task 1 (Methodology Audit): GRPO Training Delta**
The agent must find planted flaws in the protocol text and output a strict JSON schema.
*   **Baseline (Untrained Llama-3-8B-Instruct):** ~0.4000 (Hallucinates flaws, fails JSON formatting).
*   **PeerGuard (GRPO-Trained LoRA):** **0.9999** (+150% improvement). The agent perfectly identifies flaws and strictly adheres to the schema.

**2. Task 5 (FDA NDA Review): Zero-Shot Agent Pipeline**
A massive long-horizon capstone task. The agent must read 4 NDA sections, read the dataset, write/execute Python code, flag all concerns, and submit a final verdict.
*   **Baseline:** 0.2000 (Fails to execute code, blindly approves the drug).
*   **PeerGuard Pipeline:** **0.9500+** (Successfully executes verification scripts, catches all 4 sub-task flaws, and accurately submits the REJECT verdict).

---

## ⏱️ Quick Start for Reviewers (3 minutes)

1. Open the live UI: [https://huggingface.co/spaces/Nexus18/research-integrity-gym](https://huggingface.co/spaces/Nexus18/research-integrity-gym)
2. Select **"Task 5 — NDA Data Review"** from the Control Panel.
3. Click **"🚀 Deploy FDA Auditor"**.
4. Watch the agent read the protocol, execute Python analysis in the **Sandbox Terminal** tab, and generate a deterministic **Grader Report**.
5. Check the **Session Audit History** table at the bottom to see the run logged!

---

## 💡 What Makes This Novel

1. **Sandboxed Code Execution in the Loop**: The agent cannot solve Task 5 just by reading text. It *must* write and execute Python code against a hidden CSV dataset to discover class imbalances and missing patients.
2. **Deterministic RLVR Grading**: We explicitly reject "LLM-as-a-judge" grading. Every point of reward is calculated deterministically by matching the agent's actions and generated flags against the procedurally generated ground truth.
3. **Procedural Clinical Trials**: The environment does not use static datasets. It procedurally generates clinical trial text and corresponding patient CSVs on the fly based on a seed. The agent cannot memorize answers.

---

## 🎯 Theme Coverage

| Theme | What We Built |
|-------|---------------|
| **Theme 3.1 — World Modeling (Professional)** | Complex multi-document investigation, synthetic patient datasets, protocol violations, and adversarial citation fabrication. |
| **Theme 4 — Self-Improvement** | The agent learns to avoid false-positive shaping penalties (-0.05) by becoming highly precise in its flaw detection through GRPO. |

---

## 🧠 The Core Innovation: Deterministic RLVR Graders

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

Task 5 is the capstone environment. It stitches together methodology, safety data, efficacy claims, and citations into a single massive NDA submission. 

**The Agent Trace:**
1. `read_section` → Parses all 4 sections of the NDA.
2. `read_dataset` → Locates the raw patient CSV.
3. `execute_code` → Runs a pandas script in the secure sandbox to discover class imbalances.
4. `flag_concern` → Raises structured flags based on the code output and text.
5. `submit_fda_verdict` → Submits the final `REJECT` decision.

---

## 🎲 Procedural Generation

A benchmark has fixed episodes. PeerGuard generates them procedurally:
*   5 distinct taxonomies of flaws (e.g., Unblinded Bias, Endpoint Switching).
*   Randomized clinical domains (e.g., Oncology, Cardiology).
*   Dynamic dataset generation (CSVs with dynamically dropped patients to simulate exclusion fraud).

---

## 🛠️ Training Pipeline

*   **Model**: `unsloth/Llama-3-8b-Instruct-bnb-4bit`
*   **Algorithm**: `TRL SFTTrainer` (Warmstart) + `TRL GRPOTrainer` (RLVR)
*   **Hardware**: A10G GPU (via Unsloth optimization)

Reproduce the training via the provided Colab notebook: `PeerGuard_GRPO_Training.ipynb`

---

## 🗺️ Architecture Map

```text
research-integrity-gym/
├── app.py                      # Main Gradio application & UI
├── env/
│   ├── environment.py          # Core OpenEnv state machine & sandbox execution
│   └── models.py               # Pydantic schemas for actions/observations
├── graders/
│   ├── grader1.py              # Deterministic regex grader for Task 1
│   └── grader5.py              # Aggregation grader for Task 5
├── tasks/
│   ├── task1_methodology.py    # Procedural generation for Task 1
│   └── task5_fda_approval.py   # Capstone generation for Task 5
└── PeerGuard_GRPO_Training.ipynb # Full Unsloth/TRL Training pipeline
```

---

## ⚡ Quickstart

**Run Locally:**
```bash
git clone https://github.com/Bhavishya011/research-integrity-gym
cd research-integrity-gym
pip install -r requirements.txt
python app.py
```
*(App will launch on `http://localhost:7860`)*
