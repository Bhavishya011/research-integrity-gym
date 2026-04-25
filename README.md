---
title: PeerGuard Clinical Trial Auditor
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🛡️ PeerGuard: Clinical Trial Verification Agent

[![openenv](https://img.shields.io/badge/openenv-compatible-green)](https://github.com/openenv)
[![HuggingFace](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces/Nexus18/research-integrity-gym)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**PeerGuard** is an OpenEnv-based reinforcement learning environment and GRPO training pipeline that teaches LLM agents to act as **FDA Lead Regulators**. 

While most agent benchmarks focus on producing code or playing grid-world games, PeerGuard focuses on **high-stakes scientific verification**. The agent must autonomously audit synthetic clinical trial protocols, execute Python code to verify datasets, catch methodological flaws, and ultimately decide whether to approve or reject a drug for the market.

---

## 🌟 Why This Matters (The Problem)

The replication crisis affects an estimated 50–70% of published scientific findings. In clinical trials, a methodological flaw or undisclosed data exclusion isn't just a statistical error—it costs lives. 

LLMs are excellent at *generating* text, but they struggle with rigorous, multi-step *verification*. PeerGuard exists to teach LLMs something they currently cannot do well: **critically auditing complex scientific claims against hard evidence and strict CONSORT standards.** This is an underexplored domain in RL/LLM training that moves agents from "helpful assistants" to "reliable regulators."

---

## 🏗️ The Environment & Reward Signal

PeerGuard provides a rich, procedurally generated OpenEnv environment designed specifically for **Reinforcement Learning with Verifiable Rewards (RLVR)**.

### Key Innovations:
1. **Procedural Generation (No Data Leakage)**: Every episode generates a fresh clinical trial paper stub with random domains, sample sizes, and planted flaws. The agent cannot memorize answers; it must actually reason.
2. **Subprocess Jail**: The agent can write Python code to analyze trial datasets. This runs in a secure, resource-limited subprocess. `step()` never hangs.
3. **Deterministic Graders (No LLM-as-Judge)**: We do not use LLMs to score the agent. Rewards are calculated using strict, keyword-and-logic-based Python scripts (`graders/`). 
   - *Rich Signal*: The reward isn't just 0/1. The agent gets partial credit for finding the right flaw in the wrong section, and receives negative shaping penalties (capped) for hallucinating false positives. This makes the reward function **hard to game**.

### The 5-Task Curriculum
1. **Methodology Audit `[easy]`**: Identify 4 planted CONSORT violations (e.g., unblinded investigator bias, insufficient power).
2. **Experiment Replication `[medium]`**: Write code to analyze a dataset, handle class imbalance, and replicate the paper's reported AUC/F1 scores.
3. **Claim Verification `[hard]`**: Detect subtle data exclusions by re-analyzing the raw dataset and comparing the true p-value to the paper's claims.
4. **Citation Integrity `[medium-hard]`**: Detect fabricated citations (e.g., paper claims drug works on adults, but cited source was on mice).
5. **FDA Approval (Capstone) `[epic]`**: A 40-step master task combining Tasks 1-4. The agent must independently investigate a massive paper and submit a final FDA verdict.

---

## 📈 Real Training, End-to-End

We have provided a complete training pipeline (`PeerGuard_GRPO_Training.ipynb`) that runs directly on HuggingFace Spaces (A10G GPU).

We use **GRPO (Group Relative Policy Optimization)** combined with an **SFT Warmstart**. 
1. **SFT Warmstart**: We procedurally generate 50 episodes and extract the absolute ground-truth answers from the environment state to teach the model the strict JSON output format and flaw taxonomy.
2. **GRPO**: The agent generates 8 varying audit reports per paper. The environment grades them, and GRPO reinforces the internal reasoning paths that led to the highest deterministic scores.

### Results & Training Plots

*(Reviewers: The plots below demonstrate the agent's learning progression during our GRPO training run. Notice the reward climbing as the agent learns to avoid false-positive penalties and correctly identify procedural flaws.)*

![Training Reward Curve](grpo_reward_curve.png)
> **Figure 1**: GRPO Reward Curve. The environment's deterministic grader is the sole reward signal. The curve shows the agent moving from the baseline SFT capability toward a perfect 1.0 score.

![SFT Loss Curve](grpo_loss_curve.png)
> **Figure 2**: SFT Warmstart Loss. Demonstrates the model learning the JSON schema and ground-truth taxonomy before RL begins.

*(Note: Baseline Llama-3-8B-Instruct scores ~0.40 on Task 1. Post-GRPO, the agent consistently achieves >0.85).*

---

## 🚀 Setup & Usage

```bash
git clone https://huggingface.co/spaces/Nexus18/research-integrity-gym
cd research-integrity-gym
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run Locally (FastAPI)
```bash
uvicorn api.app:app --host 0.0.0.0 --port 7860 --reload
```

### Interactive Smoke Test
```bash
python verify_env.py
```

### Run the RL Training Pipeline
1. Open the HuggingFace Space.
2. Run `PeerGuard_GRPO_Training.ipynb`.
3. The notebook will automatically generate and save the training plots to the repository.

---

## 🧠 Action Space & APIs

The environment complies with the OpenEnv standard (`step`, `reset`, `state`).

| Action | Payload |
|--------|---------|
| `read_section` | `section: str` |
| `execute_code` | `code: str` |
| `submit_audit` | `flaws: [{flaw_type, location, description}]` |
| `submit_fda_verdict`| `decision: 'APPROVE' | 'REJECT' | 'REVISE'` |

---

## License
MIT
