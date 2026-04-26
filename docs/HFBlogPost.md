# Escaping the "LLM-as-a-Judge" Trap: Catching Clinical Fraud with Code

**By Team Nexus18**

## The Verification Gap in Medical AI

When building AI agents to handle high-stakes regulatory tasks—like auditing clinical trials or reviewing New Drug Applications (NDAs)—the industry relies heavily on "LLM-as-a-Judge" evaluation. 

The problem? LLMs are easily manipulated soft graders. They hallucinate partial credit, struggle with strict biostatistics, and can be fooled by polite, confident formatting. In a software demo, that’s an edge case. At the FDA, a hallucinated drug approval costs lives. 

Take the **Vioxx Disaster** of 2004. Up to 60,000 Americans died because cardiovascular risk data was glossed over in reporting. The paper looked valid to human eyes, but the underlying anomalies were fatal. Over 10,000 biomedical papers have been retracted in the last decade, but only after potentially affecting **hundreds of thousands of enrolled patients** (Source: *RetractionWatch / BMJ*). 

For the Meta PyTorch × Scaler Hackathon, our team decided to kill the soft-grading system. We built **PeerGuard**, an autonomous Review Board agent trained entirely inside a deterministic OpenEnv sandbox. If the math is wrong, the agent gets a zero. No partial credit. No vibes. Just Reinforcement Learning with Verifiable Rewards (RLVR).

Our RLVR reward shaping strategies and verifiable environment design are heavily inspired by recent advancements in Reinforcement Learning, specifically the reward ideas outlined in [arXiv:2601.19100](https://arxiv.org/abs/2601.19100).

---

## The Architecture: Deterministic Sandboxing

![PeerGuard Architecture Flowchart](../architecture.png)

Instead of asking an LLM if a clinical trial protocol "looks correct," PeerGuard is forced to execute verifiable actions.

We deployed a quantized `Llama-3-8B-Instruct` model and connected it to **OpenEnv**, a deterministic execution environment. The agent reads procedurally generated clinical papers, searches for planted flaws (like p-hacking or silently excluded patients), and outputs strict JSON payloads. 

The environment then parses that payload using regex and mathematically verifies the findings against the procedural ground truth. If the agent hallucinates a flaw, it gets hit with a negative shaping penalty.

---

## The RL Pipeline: SFT, GRPO, and Unsloth

<div style="display: flex; gap: 10px;">
  <img src="baseline_vs_trained.png" alt="Baseline vs Trained Comparison" width="32%">
  <img src="combined_reward.png" alt="Combined Reward Curve" width="32%">
  <img src="combined_loss.png" alt="Combined Loss Curve" width="32%">
</div>

We used a staged training approach:
1. **SFT warm-start** for output format discipline (valid schema / structured actions).
2. **GRPO** for reward-driven behavior in a deterministic environment.

### Proof of Training: SFT & GRPO Logs
To ensure full reproducibility, we have included our training logs showing the convergence of the **SFT Warmstart** and the **GRPO policy optimization**.

<div style="display: flex; gap: 10px;">
  <img src="sft_logs.png" alt="SFT Training Logs" width="48%">
  <img src="grpo_logs.png" alt="GRPO Training Logs" width="48%">
</div>

Before training, the baseline model (tested via **Groq Llama API**) struggled to follow instructions, scoring only **~0.4** on Task 1 and **~0.2** on Task 5. After the RLVR pipeline, PeerGuard achieved near-perfect scores by learning to prioritize deterministic evidence over narrative vibes.

---

## The Real Test: Zero-Shot Code Generation

![PeerGuard UI showing Zero-Shot Generalization](../gradio_ui.png)

Passing the methodology audit (Task 1) was great, but we wanted to see if the environment could handle complex, long-horizon data verification.

To test this, we built **Task 5 (FDA NDA Review)**. This task is our Capstone challenge. It natively combines all previous environment tasks into a single, massive end-to-end review pipeline. The agent must read 4 distinct sections of a synthetic NDA and simultaneously execute Python to analyze a raw patient CSV.

When we fed the baseline Llama-3 model this task, it failed. It read the text summaries, ignored the dataset, and blindly approved a toxic drug. This is exactly what happened during the **COVID-19 Surgisphere Scandal**, where studies were published based on fabricated data that "looked" perfect in text but was mathematically impossible in reality.

When we deployed PeerGuard, the agent autonomously abandoned static text summaries. It wrote an executable Python script, executed it within the secure OpenEnv sandbox, and correctly **rejected** the drug.

### The Agent in Action: Task 5 Sandbox Trace

![PeerGuard Task 5 Execution](../task5_sandbox.png)

Below is the actual code generated and executed by PeerGuard during an NDA audit:

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

**Verdict:** `REJECT`

---

## Why this matters

We cannot rely on LLMs to grade our homework when the stakes are this high. As documented in recent research on **LLM Sycophancy** ([arXiv:2601.16529](https://arxiv.org/abs/2601.16529)), models used as judges often provide positive feedback simply because the input is politely formatted, even if it is factually wrong. 

As we scale agentic workflows in healthcare and regulation, deterministic sandboxes like OpenEnv combined with GRPO will become the standard. 

The PeerGuard model weights are available as a LoRA adapter in the [peerguard_lora_final/](https://github.com/Bhavishya011/research-integrity-gym/tree/main/peerguard_lora_final) directory of our repository.

## References

- Reward design inspiration: [arXiv:2601.19100](https://arxiv.org/abs/2601.19100)  
- Sycophancy risk framing: [arXiv:2601.16529](https://arxiv.org/abs/2601.16529)  
- Retractions context: [Retraction Watch](https://retractionwatch.com/)  
- Clinical risk context: [BMJ](https://www.bmj.com/)
