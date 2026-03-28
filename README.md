---
title: Research Integrity Gym
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Research Integrity Gym

An [OpenEnv](https://github.com/openenv) environment where AI agents evaluate **scientific research integrity** — the verification half of science that no existing benchmark covers.

> Every existing agent benchmark trains models to *produce* scientific outputs.  
> This environment trains agents to *verify* them.

---

## Motivation

The replication crisis affects an estimated 50–70% of published scientific findings. Detecting methodological flaws, replicating experiments, and catching statistical manipulation are critical skills — for human reviewers today, and for AI systems tomorrow. This environment provides a structured, fully automated benchmark for these capabilities.

---

## Environment Description

Agents interact with synthetic paper stubs via a step/reset/state API. Each episode presents a research integrity challenge at one of three difficulty levels.

**Key design decisions:**
- **Procedural generation** — paper stubs are generated fresh each episode using templates with random flaw injection. No two episodes share the same surface text, preventing memorisation.
- **Deterministic graders** — all scoring is pure code (numeric comparisons, keyword matching, taxonomy lookup). Zero LLM-as-judge, zero subjectivity.
- **Sandboxed code execution** — agent-supplied Python runs in a subprocess with a 5-second hard timeout and restricted builtins. `step()` never hangs.

---

## Tasks

### Task 1 — Methodology Audit `[easy]`
Agent reads a paper stub containing **4 planted methodological flaws** and must identify them.

| Flaw type | Example |
|-----------|---------|
| `wrong_statistical_test` | Chi-square applied to continuous outcome |
| `underpowered_sample` | n=22 per group, no power analysis reported |
| `undisclosed_exclusion` | Results report n=141 but 150 were recruited |
| `p_value_manipulation` | Multiple outcomes tested, only significant one reported |

Grader: each correct flaw = 0.25 pts; false positives = −0.05 each.

### Task 2 — Experiment Replication `[medium]`
Agent receives a methods section and must replicate a logistic regression experiment.

**Key challenge:** the dataset has ~20% class imbalance. A naive model without `stratify=y` and `class_weight='balanced'` scores ~0.71 AUC — outside the tolerance window.

Grader: AUC within ±0.01 = 0.45 pts, F1 within ±0.01 = 0.35 pts, interpretation quality = 0.20 pts.

### Task 3 — Claim Verification `[hard]`
A paper claims a treatment is significantly effective (p<0.05). The claim is **subtly wrong** — authors silently excluded outliers, changing the true result to p>0.05 (not significant).

Agent must independently re-analyse the raw dataset, detect the discrepancy, and submit the correct verdict.

Grader: correct verdict = 0.35, correct effect size = 0.20, p-value direction = 0.15, detected undisclosed exclusion = 0.20, justification keywords = 0.10.

---

## Action Space

| Action | Fields | Available in |
|--------|--------|-------------|
| `read_section` | `section: str` | All tasks |
| `read_dataset` | — | All tasks |
| `execute_code` | `code: str` | All tasks |
| `flag_flaw` | `flaw_type, location, description` | Task 1 |
| `flag_concern` | `concern_type, evidence` | Task 3 |
| `submit_audit` | `audit_payload: {flaws: [...]}` | Task 1 |
| `submit_results` | `results_payload: {auc, f1, interpretation}` | Task 2 |
| `submit_verdict` | `verdict_payload: {verdict, effect_size, p_value, justification}` | Task 3 |

## Observation Space

```json
{
  "task_id":          "string",
  "step":             "int",
  "paper_text":       "string — full paper visible to agent",
  "dataset_summary":  "string | null",
  "code_result":      "string | null — stdout from last execute_code",
  "last_reward":      "float",
  "flags_raised":     ["string"],
  "available_actions": ["string"],
  "done":             "bool"
}
```

---

## Baseline Scores

Evaluated with `llama-3.3-70b-versatile` (via Groq), temperature=0, seed=42.

| Task | Grader Score | Steps |
|------|-------------|-------|
| Task 1 — Methodology Audit | 0.40 | 8 |
| Task 2 — Experiment Replication | 0.80 | 6 |
| Task 3 — Claim Verification | 0.50 | 10 |
| **Average** | **0.57** | — |

*Scores show interesting performance: strong on ML replication (0.80), moderate on claim verification (0.50), but struggles with methodology audit (0.40). A perfect agent would score 1.0 on all tasks.*

---

## Setup & Usage

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/research-integrity-gym
cd research-integrity-gym
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run locally
```bash
uvicorn api.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run with Docker
```bash
docker build -t research-integrity-gym .
docker run -p 7860:7860 research-integrity-gym
```

### Run baseline
```bash
export GROQ_API_KEY=gsk_...    # Get free key at console.groq.com
python baseline.py
```

### Quick API test
```bash
# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_methodology_audit"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "read_section", "section": "methods"}}'
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Execute action |
| `GET` | `/state` | Current episode state |
| `GET` | `/tasks` | List tasks + schemas |
| `POST` | `/grader` | Run grader externally |
| `POST` | `/baseline` | Trigger baseline script |

Full interactive docs: `http://localhost:7860/docs`

---

## Reward Function

Mid-episode signals (scaled to ≤30% of total):

| Signal | Value |
|--------|-------|
| Read a relevant section | +0.02 |
| First dataset read | +0.03 |
| Code executes without error | +0.05 |
| Correctly flags a flaw (mid-episode) | +0.08 |
| Intermediate metric close to GT | +0.05 |
| Code throws exception | −0.03 |
| Repeat identical action | −0.05 |
| False positive flag | −0.05 (cap −0.20) |
| Step budget exceeded | −0.10 |

Terminal reward: `grader_score × 0.80 + mid_episode_total × 0.20`

---

## License

MIT
