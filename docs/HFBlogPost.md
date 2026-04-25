# Escaping the "LLM-as-a-Judge" Trap: Catching Clinical Fraud with Code

**By Team Nexus18**

## The Verification Gap in Medical AI

When building AI agents to handle high-stakes regulatory tasks—like auditing clinical trials or reviewing New Drug Applications (NDAs)—the industry relies heavily on "LLM-as-a-Judge" evaluation. 

The problem? LLMs are easily manipulated soft graders. They hallucinate partial credit, struggle with strict biostatistics, and can be fooled by polite, confident formatting. In a software demo, that’s an edge case. At the FDA, a hallucinated drug approval costs lives.

For the Meta PyTorch × Scaler Hackathon, our team decided to kill the soft-grading system. We built **PeerGuard**, an autonomous Review Board agent trained entirely inside a deterministic OpenEnv sandbox. If the math is wrong, the agent gets a zero. No partial credit. No vibes. Just Reinforcement Learning with Verifiable Rewards (RLVR).

Here is how we built it, why our early training runs collapsed, and how the agent achieved zero-shot generalization on raw CSV data.

## The Architecture: Deterministic Sandboxing

![PeerGuard Architecture Flowchart](../architecture.png)

Instead of asking an LLM if a clinical trial protocol "looks correct," PeerGuard is forced to execute verifiable actions.

We deployed a quantized `Llama-3-8B-Instruct` model and connected it to **OpenEnv**, a deterministic execution environment. The agent reads procedurally generated clinical papers, searches for planted flaws (like p-hacking or silently excluded patients), and outputs strict JSON payloads. 

The environment then parses that payload using regex and mathematically verifies the findings against the procedural ground truth. If the agent hallucinates a flaw, it gets hit with a negative shaping penalty.

## The RL Pipeline: TRL, GRPO, and Unsloth

<div style="display: flex; gap: 10px;">
  <img src="grpo_reward_curve.png" alt="GRPO Reward Curve" width="49%">
  <img src="grpo_loss_curve.png" alt="SFT Loss Curve" width="49%">
</div>

We fine-tuned the model on an Nvidia A10G using Hugging Face’s `GRPOTrainer` and `Unsloth`. But getting the policy gradient to actually converge was a fight.

In our early runs, the loss was thrashing and rewards were flatlining at zero. The issue wasn't the model; it was our reward design. Our sandbox was too unforgiving. If the model missed a single comma in the JSON, we were hitting it with a heavy `-0.5` penalty. The model quickly learned that "trying and failing" was mathematically worse than just generating empty text, leading to total policy collapse.

**The Fix:** We redesigned the Markov Decision Process (MDP) to prioritize outcomes over process constraints:
1. We set the reward floor to exactly `0.0`. No heavy negative penalties for formatting.
2. We added a tiny `+0.1` format-shaping bonus to gently guide the model toward valid JSON.
3. We dropped the learning rate to a highly conservative `5e-6` to protect the model's baseline intelligence from policy shock.
4. We increased the generation batch size to `8` to stabilize the advantage estimations.

**The Result:** The agent went from a `~40%` baseline success rate to scoring a perfect **0.9999** on the deterministic grader. It wasn't just guessing; it learned exact precision to calculate patient dropouts and flag unblinded investigators.

## The Real Test: Zero-Shot Code Generation

![PeerGuard UI showing Zero-Shot Generalization](../gradio_ui.png)

Passing the methodology audit (Task 1) was great, but we wanted to see if the environment could handle complex, long-horizon data verification.

To test this, we built **Task 5 (NDA Data Review)**. This task is entirely different. Instead of just reading text protocols, the agent is handed raw patient CSV files.

When we fed the baseline Llama-3 model this task, it failed. It read the text summaries, ignored the dataset, and blindly approved a toxic drug.

When we deployed the full PeerGuard pipeline, the agent autonomously abandoned static text summaries. It wrote an executable Python script using Pandas, executed it within the secure OpenEnv sandbox to mathematically check the CSV for adverse event class imbalances, and extracted its findings into a final FDA Verdict JSON. The sandbox executed the script, found the toxic anomalies, and correctly **rejected** the drug.

The rigorous reasoning we instilled during the text audits successfully transferred to a pure code-generation environment.

## What's Next

Building applied AI systems requires moving beyond simple prompt engineering and into rigorous, deterministic optimization. As we scale agentic workflows in healthcare and regulation, deterministic sandboxes like OpenEnv combined with GRPO will become the standard. 

We cannot rely on LLMs to grade our homework when the stakes are this high.
