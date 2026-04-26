# Escaping the "LLM-as-a-Judge" Trap: Catching Clinical Fraud with Code

**By Team Nexus18**

## The verification gap in medical AI

High-stakes review tasks (clinical trial audits, NDA reviews) break when evaluation is soft.  
If an LLM judge can be persuaded by fluent text, bad science can still score well.

That is the core problem we targeted in this hackathon project: build an agent that is rewarded for **verifiable correctness**, not persuasive wording.

Historical failures make this concrete:
- Vioxx showed how safety signals can be under-weighted until major harm occurs.
- The Surgisphere incident showed how polished narrative can hide invalid underlying data.

## What we built

We built **PeerGuard**, an OpenEnv-compatible RL environment where the agent acts as an FDA-style regulator:
1. Read procedurally generated trial content.
2. Inspect raw datasets.
3. Execute sandboxed Python for independent checks.
4. Raise structured concerns.
5. Submit a final deterministic verdict.

![PeerGuard Architecture Flowchart](../architecture.png)

The key design principle is deterministic grading:
- no LLM-as-judge in scoring,
- explicit rule-based graders,
- ground-truth-backed reward signals.

## RL pipeline: from schema compliance to reliable audit behavior

We used a staged training approach:
1. **SFT warm-start** for output format discipline (valid schema / structured actions).
2. **GRPO** for reward-driven behavior in a deterministic environment.

![Baseline vs Trained Comparison](baseline_vs_trained.png)
![Combined Reward Curve](combined_reward.png)
![Combined Loss Curve](combined_loss.png)

Early runs failed because reward shaping over-penalized formatting misses and caused collapse.  
We corrected this by using gentler shaping and prioritizing terminal correctness.

## The capstone: Task 5 (NDA review)

Task 5 combines protocol audit, replication, claim verification, and citation checks into one long-horizon episode.

The agent must:
1. audit 4 sections of an NDA-style submission,
2. run code against raw CSV data,
3. flag protocol/statistical/citation issues,
4. issue `APPROVE`, `REJECT`, or `REVISE`.

![PeerGuard UI showing Zero-Shot Generalization](../gradio_ui.png)

To keep execution reliable under constrained memory, Task 5 now includes a **CSV-safe fallback execution path** if generated code is invalid or too memory-heavy. This preserves deterministic audit flow instead of failing the episode on runtime fragility.

## Why this matters

This project is a practical argument for **RL with verifiable rewards** in regulated domains:
- If scoring is soft, agents optimize style.
- If scoring is deterministic, agents optimize evidence.

Healthcare and regulatory AI needs the second path.

## References

- Reward design inspiration: [arXiv:2601.19100](https://arxiv.org/abs/2601.19100)  
- Sycophancy risk framing: [arXiv:2601.16529](https://arxiv.org/abs/2601.16529)  
- Retractions context: [Retraction Watch](https://retractionwatch.com/)  
- Clinical risk context: [BMJ](https://www.bmj.com/)
