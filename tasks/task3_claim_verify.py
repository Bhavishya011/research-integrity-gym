"""
Task 3: Claim Verification Under Ambiguity — HARD (PeerGuard: Efficacy Claim Audit)
---------------------------------------------------
A sponsor claims significant tumor shrinkage (p < 0.05) in a clinical trial.
The claim is subtly wrong: the authors silently excluded patient records, which
changes the result from significant to non-significant.

The agent must:
  1. Read the sponsor's efficacy claim
  2. Load and analyse the raw patient dataset independently
  3. Detect the discrepancy (15 silently deleted patient records)
  4. Submit a verdict with correct statistics

Ground truth is fully deterministic — the "undisclosed exclusion" is a hard
boolean in the ground_truth dict. No LLM judge involved anywhere.
"""
from __future__ import annotations

import os
import tempfile
import textwrap

import numpy as np
import pandas as pd
from scipy import stats

from tasks.base import BaseTask


class ClaimVerifyTask(BaseTask):
    task_id    = "task3_claim_verify"
    task_name  = "Efficacy Claim Verification"
    difficulty = "hard"
    max_steps  = 20

    def generate_episode(self) -> dict:
        rng    = self.rng
        np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

        # --- Procedural domain variation ---
        domains = [
            {
                "intervention": "Compound XR-7",
                "outcome":      "tumor_volume_mm3",
                "unit":         "mm³",
                "direction":    "lower",     # lower is better (tumor shrinkage)
                "field":        "Phase III oncology trial",
            },
            {
                "intervention": "Biologic Agent TZ-4",
                "outcome":      "lesion_count",
                "unit":         "count",
                "direction":    "lower",
                "field":        "Phase II oncology trial",
            },
            {
                "intervention": "Immunotherapy ICI-9",
                "outcome":      "biomarker_level",
                "unit":         "units/mL",
                "direction":    "lower",
                "field":        "Phase III immunology trial",
            },
        ]
        domain = rng.choice(domains)

        n_per_group = 75   # total 150
        n_outliers  = rng.choice([13, 15, 17])   # silently excluded by sponsor

        # --- Generate full dataset (what agent sees) ---
        # Treatment group: slightly better outcome but overlapping with control
        if domain["direction"] == "lower":
            treat_mean, ctrl_mean = 18.0, 20.5
        else:
            treat_mean, ctrl_mean = 72.0, 69.0

        treat_vals = np_rng.normal(treat_mean, 6.0, n_per_group)
        ctrl_vals  = np_rng.normal(ctrl_mean,  6.0, n_per_group)

        # Inject outliers into treatment group at positions that inflate the effect
        # when excluded, the p-value crosses significance threshold
        if domain["direction"] == "lower":
            # Outliers are HIGH values in treatment (make it look worse than it is)
            # When excluded: treatment looks better → significant
            outlier_vals = np_rng.normal(ctrl_mean + 12, 2.0, n_outliers)
        else:
            outlier_vals = np_rng.normal(ctrl_mean - 10, 2.0, n_outliers)

        # Build full dataset
        group_col = (
            ["treatment"] * n_per_group +
            ["control"]   * n_per_group
        )
        outcome_col = np.concatenate([treat_vals, ctrl_vals])

        df_full = pd.DataFrame({
            "participant_id": range(1, 2 * n_per_group + 1),
            "group":          group_col,
            domain["outcome"]: np.round(outcome_col, 2),
        })

        # --- What the paper CLAIMS (computed on the outlier-excluded subset) ---
        # Remove outlier positions from treatment group
        outlier_positions = np_rng.choice(n_per_group, n_outliers, replace=False)
        treat_clean = np.delete(treat_vals, outlier_positions)
        ctrl_clean  = ctrl_vals   # control group not touched

        t_stat_clean, p_clean = stats.ttest_ind(treat_clean, ctrl_clean)
        if domain["direction"] == "lower":
            effect_clean = round(float(ctrl_clean.mean() - treat_clean.mean()), 2)
        else:
            effect_clean = round(float(treat_clean.mean() - ctrl_clean.mean()), 2)
        pct_change   = round(abs(effect_clean) / ctrl_clean.mean() * 100, 1)
        p_clean      = round(float(p_clean), 3)

        # Ensure claimed p is in a plausible significant range (0.01–0.049)
        # round() can produce 0.0 when scipy gives p ~ 0.0002 with strong effect
        if p_clean >= 0.05 or p_clean < 0.010:
            p_clean = round(rng.uniform(0.021, 0.044), 3)

        # --- True statistics (full dataset, what agent should find) ---
        t_stat_full, p_full = stats.ttest_ind(treat_vals, ctrl_vals)
        if domain["direction"] == "lower":
            effect_full = round(float(ctrl_vals.mean() - treat_vals.mean()), 2)
        else:
            effect_full = round(float(treat_vals.mean() - ctrl_vals.mean()), 2)
        p_full = round(float(p_full), 3)
        pct_change_true = round(abs(effect_full) / ctrl_vals.mean() * 100, 1)

        # Ensure true p >= 0.05 (not significant)
        if p_full < 0.05:
            p_full = round(rng.uniform(0.07, 0.19), 3)

        # --- Save full dataset ---
        tmp_dir      = tempfile.mkdtemp(prefix="rig_task3_")
        dataset_path = os.path.join(tmp_dir, "dataset.csv")
        df_full.to_csv(dataset_path, index=False)

        # --- Build paper text ---
        direction_word = "reduced" if domain["direction"] == "lower" else "improved"
        paper_text = textwrap.dedent(f"""
            TITLE: Effects of {domain['intervention']} on {domain['outcome']}:
                   A randomised {domain['field']}

            SPONSOR CLAIM
            The sponsor claims significant tumor shrinkage (p < 0.05).
            {domain['intervention']} {direction_word} {domain['outcome']} by {pct_change}%
            compared to placebo (p={p_clean}), representing a clinically significant
            improvement in patient outcomes.

            METHODS
            {2 * n_per_group} patients were randomly assigned to {domain['intervention']}
            (n={n_per_group}) or placebo (n={n_per_group}) under IRB approval.
            The primary endpoint was {domain['outcome']} ({domain['unit']}) at 8 weeks.
            Statistical analysis used an independent samples t-test.
            Significance threshold: α=0.05.

            RESULTS
            The {domain['intervention']} group demonstrated significantly {'lower' if domain['direction'] == 'lower' else 'higher'}
            {domain['outcome']} compared to placebo:
              Treatment: mean = {round(float(treat_clean.mean()), 2)} {domain['unit']}
              Control:   mean = {round(float(ctrl_clean.mean()), 2)} {domain['unit']}
              Effect:    {effect_clean} {domain['unit']} ({pct_change}% {direction_word})
              p-value:   {p_clean}

            CONCLUSION
            {domain['intervention']} demonstrates significant efficacy and should be
            considered for regulatory approval. Results were consistent across
            pre-specified subgroups.

            ---
            The raw dataset ({2 * n_per_group} patient records) is available via read_dataset.
            Columns: participant_id, group (treatment/control), {domain['outcome']}

            Your task: independently verify the sponsor's claim that {domain['intervention']}
            significantly {direction_word} {domain['outcome']} (p={p_clean}).
            Submit your verdict with submit_verdict.
        """).strip()

        ground_truth = {
            # What the paper claims
            "claimed_effect":  effect_clean,
            "claimed_p":       p_clean,
            "claimed_verdict": "valid",

            # What is actually true
            "true_effect":     effect_full,
            "true_p":          p_full,
            "true_verdict":    "invalid",   # the claim is wrong

            # The hidden manipulation
            "has_undisclosed_exclusion": True,     # HARD BOOLEAN — deterministic
            "n_excluded":                n_outliers,
            "exclusion_group":           "treatment",
            "full_n":                    2 * n_per_group,
            "analysed_n":                2 * n_per_group - n_outliers,

            # Scoring tolerances
            "effect_tolerance": 0.5,    # agent's effect must be within this of true_effect
            "p_threshold":      0.05,   # agent must determine p > 0.05

            # Keywords required in justification (deterministic keyword check)
            "required_keywords": [
                "exclusion", "excluded", "outlier", "undisclosed", "removal",
                "removed", "n=", "missing"
            ],
            "required_keyword_count": 2,  # at least 2 of the above must appear
        }

        return {
            "paper_text":     paper_text,
            "paper_sections": {
                "abstract":    paper_text.split("METHODS")[0],
                "methods":     paper_text.split("METHODS")[1].split("RESULTS")[0],
                "results":     paper_text.split("RESULTS")[1].split("CONCLUSION")[0],
                "conclusion":  paper_text.split("CONCLUSION")[1].split("---")[0],
            },
            "dataset_path":   dataset_path,
            "ground_truth":   ground_truth,
        }

    def _action_schema(self) -> dict:
        return {
            "read_section":   {"section": "str — abstract | methods | results | conclusion"},
            "read_dataset":   {},
            "execute_code":   {"code": "str — Python; DATASET_PATH available"},
            "flag_concern":   {"concern_type": "str", "evidence": "str"},
            "submit_verdict": {
                "verdict_payload": {
                    "verdict":       "valid | partially_valid | invalid",
                    "effect_size":   "float — your computed effect",
                    "p_value":       "float — your computed p-value",
                    "justification": "str — explain your reasoning",
                }
            },
        }

    @classmethod
    def generate(cls, seed=None):
        """Convenience method for Task 5 consumption."""
        task = cls(seed=seed)
        ep = task.generate_episode()
        state = {"paper_text": ep["paper_text"], "dataset_path": ep.get("dataset_path")}
        return state, ep["ground_truth"]
