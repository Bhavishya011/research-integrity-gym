---
title: PeerGuard Clinical Trial Auditor
emoji: ⚕️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🛡️ PeerGuard: The FDA Autonomous Regulator

[![openenv](https://img.shields.io/badge/openenv-compatible-green)](https://github.com/openenv)
[![HuggingFace](https://img.shields.io/badge/🤗-Live_Demo-yellow)](https://huggingface.co/spaces/Nexus18/research-integrity-gym)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*A Reinforcement Learning environment and GRPO-trained agent that autonomously audits clinical trials, executes Python to verify data, and catches methodological fraud.*

**[Try the Live Demo on HuggingFace Space!](https://huggingface.co/spaces/Nexus18/research-integrity-gym)**

---

## 🛑 The Problem: The Verification Gap

The scientific replication crisis affects an estimated 50–70% of published research. In clinical trials, a methodological flaw or undisclosed data exclusion isn't just a statistical error—it costs lives. 

Current LLMs are incredible at **generation** (writing code, drafting emails, chatting). But they fail catastrophically at rigorous, multi-step **verification**. When presented with a complex clinical protocol and raw CSV data, baseline models hallucinate, struggle to follow strict reporting schemas, and fail to independently verify biostatistical claims.

**PeerGuard targets this exact capability gap.** We built an environment to transition LLMs from "helpful chat assistants" into "reliable, autonomous FDA regulators" that trust nothing and verify everything.

---

## 🏗️ The Environment: What the Agent Sees, Does, and Learns

To teach this capability, we built a complex, procedurally generated **OpenEnv** environment designed specifically for **Reinforcement Learning with Verifiable Rewards (RLVR)**.

### What the Agent Sees:
*   **Procedural Generation**: Every episode generates a completely unique clinical trial stub. The environment randomizes the medical domain, sample sizes, and injects specific, subtle methodological flaws (e.g., unblinded investigators, citation fabrication, silently dropped patients). 
*   **No Data Leakage**: Because the trials are generated on the fly, the agent *cannot* memorize answers. It must actually reason through the text.

### What the Agent Does:
*   **Investigates**: The agent can read protocol sections, inspect citations, and review datasets.
*   **Executes Code**: The agent has access to a secure **Python Sandbox**. It writes Pandas and Scikit-learn code to independently calculate p-values, F1 scores, and check for adverse-event class imbalances in the raw CSVs.
*   **Outputs Strict Schemas**: The agent must submit its final audit reports in strictly enforced JSON formats or make final `APPROVE / REJECT` FDA verdicts.

### What the Agent gets Rewarded for:
*   **Deterministic Graders (No LLM-as-Judge)**: We do NOT use LLMs to score the agent. The reward signal is 100% deterministic, based on strict regex keyword matching and logic trees.
*   **Hard to Game**: The agent receives partial credit (+0.10) for finding a flaw but misidentifying its location, a full reward (+0.25) for precision, and **negative shaping penalties** (-0.05) for hallucinating false positives.

---

## 📈 The Results: From Baseline to Expert

We trained a `Llama-3-8B-Instruct` model using **Unsloth** and **GRPO (Group Relative Policy Optimization)** directly on an A10G GPU. 

The training pipeline used an **SFT Warmstart** to teach the model the strict JSON schemas, followed by **GRPO** where the agent generated multiple reasoning paths. The environment's deterministic grader reinforced the paths that successfully found the planted flaws without triggering false-positive penalties.

### What Changed After Training?

1.  **Task 1 (Methodology Audit)**: The baseline Llama-3 model scores a miserable **~0.40**. It hallucinates flaws and fails to follow the JSON schema. After our GRPO training, the PeerGuard LoRA adapter consistently hits **0.9999 (Perfect Score)**.
2.  **Zero-Shot Generalization**: On Task 5 (a capstone task requiring the agent to write sandboxed Python code to verify data), the base model achieves a highly respectable **0.7600**, proving the environment's action space works seamlessly for complex code-generation tasks.

*(Reviewers: The plots below demonstrate the agent's learning progression. Notice the reward climbing as the agent learns to avoid false-positive penalties.)*

<div style="display: flex; gap: 10px;">
  <img src="grpo_reward_curve.png" alt="GRPO Reward Curve" width="49%">
  <img src="grpo_loss_curve.png.png" alt="SFT Loss Curve" width="49%">
</div>

---

## 🌍 Why Does It Matter?

Who cares about an agent that audits clinical trials?
*   **FDA and Regulatory Bodies**: Automating the initial sweep of New Drug Applications (NDAs) to catch basic statistical manipulation before human review.
*   **Academic Journals**: Running automated, rigorous peer-review checks on submitted manuscripts to catch data exclusion and p-hacking.
*   **AI Researchers**: Providing a highly structured, hard-to-game RLVR benchmark for evaluating an agent's ability to perform long-horizon, evidence-based verification instead of just code generation.

PeerGuard proves that with the right environment and deterministic reward shaping, we can train small open-source models to perform highly specialized, high-stakes regulatory reasoning.

---

## 🚀 Try It Yourself

The fastest way to experience PeerGuard is via our [Live Gradio Demo](https://huggingface.co/spaces/Nexus18/research-integrity-gym). The UI allows you to generate procedural trials, run the trained agent (or the baseline), view the sandboxed Python execution, and see the deterministic grader's real-time scoring.

**Local Setup:**
```bash
git clone https://huggingface.co/spaces/Nexus18/research-integrity-gym
cd research-integrity-gym
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 7860 --reload
```

## License
MIT
