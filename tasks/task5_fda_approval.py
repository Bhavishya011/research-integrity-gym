"""
Task 5: FDA Approval — LONG-HORIZON CAPSTONE (PeerGuard Grand Finale)
----------------------------------------------------------------------
This task does NOT invent new data. It calls the generators from Tasks 1-4
and stitches them into a unified "NDA Submission" that the agent must audit
end-to-end before issuing an FDA verdict.

The agent has access to all investigatory actions (read_section, read_dataset,
execute_code, flag_flaw, flag_concern, check_citation, flag_fabrication) but
NO sub-task terminal actions. The ONLY way to end this episode is
submit_fda_verdict with a decision (APPROVE / REJECT / REVISE) and
justification flags.

Expected verdict: REJECT (because of the planted flaws across all sub-tasks).
"""
from __future__ import annotations

import textwrap
from typing import Optional

from tasks.base import BaseTask


class FDAApprovalTask(BaseTask):
    task_id    = "task5_fda_approval"
    task_name  = "FDA New Drug Application Review"
    difficulty = "long-horizon"
    max_steps  = 40   # Extended budget for multi-section audit

    def generate_episode(self) -> dict:
        rng = self.rng
        seed = self.seed

        # ---------------------------------------------------------------
        # 1. Generate sub-tasks using their generate() classmethods
        # ---------------------------------------------------------------
        from tasks.task1_methodology_audit import MethodologyAuditTask
        from tasks.task2_replication import ReplicationTask
        from tasks.task3_claim_verify import ClaimVerifyTask
        from tasks.task4_citation_check import CitationCheckTask

        t1_state, t1_truth = MethodologyAuditTask.generate(seed=seed)
        t2_state, t2_truth = ReplicationTask.generate(seed=seed)
        t3_state, t3_truth = ClaimVerifyTask.generate(seed=seed)
        t4_state, t4_truth = CitationCheckTask.generate(seed=seed)

        # ---------------------------------------------------------------
        # 2. Stitch into Master NDA Memo
        # ---------------------------------------------------------------
        t2_dataset_path = t2_state.get("dataset_path", "")
        t3_dataset_path = t3_state.get("dataset_path", "")

        master_memo = textwrap.dedent(f"""
            ================================================================
            FDA NEW DRUG APPLICATION (NDA) REVIEW FILE
            Submission ID: NDA-{rng.randint(100000, 999999)}
            Sponsor: PharmaGen Therapeutics Inc.
            Date: 2024-Q{rng.randint(1, 4)}
            ================================================================

            ## 1. Clinical Protocol Summary (CONSORT Audit Required)
            {t1_state['paper_text']}

            ================================================================

            ## 2. Safety & Adverse Events (Replication Required)
            Patient dataset available: {t2_dataset_path}
            Sponsor Claim: No significant adverse cardiovascular events detected
            in the treatment cohort during the 12-week trial period.

            {t2_state['paper_text']}

            ================================================================

            ## 3. Efficacy Claims (Independent Verification Required)
            Patient dataset available: {t3_dataset_path}
            {t3_state['paper_text']}

            ================================================================

            ## 4. Supporting Literature (Citation Audit Required)
            {t4_state['paper_text']}

            ================================================================

            INSTRUCTIONS FOR FDA LEAD REGULATOR:
            - Audit ALL four sections above using available tools
            - Execute Python code to verify raw patient datasets
            - Flag any protocol violations, statistical manipulation,
              missing data, or citation fabrication
            - Issue your final verdict using submit_fda_verdict with:
              * decision: APPROVE | REJECT | REVISE
              * justification_flags: list of specific findings

            Available tools: read_section, read_dataset, execute_code,
            flag_flaw, flag_concern, check_citation, flag_fabrication,
            submit_fda_verdict
        """).strip()

        # ---------------------------------------------------------------
        # 3. Build paper sections for read_section navigation
        # ---------------------------------------------------------------
        sections = {
            "protocol_summary":     t1_state["paper_text"],
            "safety_adverse":       t2_state["paper_text"],
            "efficacy_claims":      t3_state["paper_text"],
            "supporting_literature": t4_state["paper_text"],
            "overview":             master_memo,
        }

        # ---------------------------------------------------------------
        # 4. Combine ground truth from all sub-tasks
        # ---------------------------------------------------------------
        combined_truth = {
            "t1_truth":         t1_truth,
            "t2_truth":         t2_truth,
            "t3_truth":         t3_truth,
            "t4_truth":         t4_truth,
            "expected_verdict": "REJECT",   # Because of the planted flaws
            # Flatten key facts for grader5 to check against flags_raised
            "t1_flaw_taxonomies": [f["taxonomy"] for f in t1_truth.get("flaws", [])],
            "t2_has_imbalance":   True,
            "t3_has_exclusion":   t3_truth.get("has_undisclosed_exclusion", True),
            "t4_fabrication_type": t4_truth.get("fabrication_type", "directional"),
        }

        # Use t2's dataset path as the primary (agent can also find t3's in the memo)
        # Both paths are embedded in the master memo text
        dataset_path = t2_dataset_path or t3_dataset_path

        return {
            "paper_text":     master_memo,
            "paper_sections": sections,
            "dataset_path":   dataset_path,
            "ground_truth":   combined_truth,
        }

    def _action_schema(self) -> dict:
        return {
            "read_section":     {"section": "str — protocol_summary | safety_adverse | efficacy_claims | supporting_literature | overview"},
            "read_dataset":     {},
            "execute_code":     {"code": "str — Python code; DATASET_PATH constant available"},
            "flag_flaw":        {"flaw_type": "str", "location": "str", "description": "str"},
            "flag_concern":     {"concern_type": "str", "evidence": "str"},
            "check_citation":   {"citation_id": "int"},
            "flag_fabrication": {"citation_id": "int", "fabrication_type": "str", "evidence": "str"},
            "submit_fda_verdict": {
                "fda_verdict_payload": {
                    "decision":            "APPROVE | REJECT | REVISE",
                    "justification_flags": "[str] — list of specific findings",
                }
            },
        }
