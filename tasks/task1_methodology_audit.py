"""
Task 1: Methodology Audit — EASY (PeerGuard: CONSORT Protocol Violation Audit)
---------------------------------
Agent reads a synthetic clinical trial paper stub and must identify 4 planted
CONSORT protocol violations. Paper is procedurally generated each episode
to prevent memorisation.

Flaw injection system:
  - Templates have slots: [STATISTICAL_TEST], [GROUP_A], [GROUP_B], etc.
  - At generation time, a valid combination is picked, then one or more
    flaws are injected by replacing the valid choice with an incorrect one.
  - Ground truth records which taxonomy applies and where.
"""
from __future__ import annotations

import random
import textwrap
from typing import Optional

from tasks.base import BaseTask


# ---------------------------------------------------------------------------
# Vocabulary pools — the raw material for procedural generation
# ---------------------------------------------------------------------------

DOMAINS = [
    {
        "field": "clinical trial",
        "intervention": ["Drug A", "Treatment B", "Compound X", "Therapy Z"],
        "outcome": ["recovery time", "symptom severity", "blood pressure", "pain score"],
        "group_type": "patients",
    },
    {
        "field": "psychology study",
        "intervention": ["Mindfulness training", "Cognitive therapy", "Group therapy", "App-based intervention"],
        "outcome": ["anxiety score", "depression score", "stress levels", "cognitive performance"],
        "group_type": "participants",
    },
    {
        "field": "educational study",
        "intervention": ["Active learning", "Flipped classroom", "Peer tutoring", "Digital tools"],
        "outcome": ["exam scores", "knowledge retention", "engagement", "completion rate"],
        "group_type": "students",
    },
]

# (test_name, valid_data_types, invalid_for)
STAT_TESTS = {
    "independent samples t-test":   {"valid": ["continuous", "normally distributed"],     "invalid_for": ["categorical", "ordinal", "non-normal"]},
    "chi-square test":               {"valid": ["categorical", "frequency data"],           "invalid_for": ["continuous", "time-series"]},
    "Mann-Whitney U test":           {"valid": ["ordinal", "non-normal continuous"],        "invalid_for": ["normally distributed small samples"]},
    "one-way ANOVA":                 {"valid": ["continuous", "3+ groups", "normal"],       "invalid_for": ["binary outcome", "two groups"]},
    "Pearson correlation":           {"valid": ["continuous linear relationship"],          "invalid_for": ["ordinal", "non-linear"]},
    "logistic regression":           {"valid": ["binary outcome", "categorical"],           "invalid_for": ["continuous outcome", "ordinal multi-class"]},
}

# Flaw templates: (flaw_taxonomy, description_template, section_hint)
FLAW_TEMPLATES = [
    {
        "taxonomy": "unblinded_investigator_bias",
        "inject":   lambda ctx, rng: _inject_unblinded_bias(ctx, rng),
        "section":  "statistical_analysis",
    },
    {
        "taxonomy": "insufficient_power_analysis",
        "inject":   lambda ctx, rng: _inject_insufficient_power(ctx, rng),
        "section":  "participants",
    },
    {
        "taxonomy": "protocol_deviation_unreported",
        "inject":   lambda ctx, rng: _inject_protocol_deviation(ctx, rng),
        "section":  "results",
    },
    {
        "taxonomy": "endpoint_switching",
        "inject":   lambda ctx, rng: _inject_endpoint_switching(ctx, rng),
        "section":  "results",
    },
]


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class MethodologyAuditTask(BaseTask):
    task_id    = "task1_methodology_audit"
    task_name  = "CONSORT Protocol Violation Audit"
    difficulty = "easy"
    max_steps  = 20

    def generate_episode(self) -> dict:
        rng = self.rng
        domain = rng.choice(DOMAINS)

        ctx = {
            "field":         domain["field"],
            "intervention":  rng.choice(domain["intervention"]),
            "outcome":       rng.choice(domain["outcome"]),
            "group_type":    domain["group_type"],
            "n_per_group":   rng.choice([18, 22, 24, 26, 28]),   # small → underpowered
            "total_n":       0,
            "alpha":         0.05,
            "rng":           rng,
            "flaws_text":    {},   # section -> flaw sentence injected
            "flaw_notes":    [],   # ground truth list
        }
        ctx["total_n"] = ctx["n_per_group"] * 2

        # Inject all 4 flaws
        flaw_ids = []
        for i, ft in enumerate(FLAW_TEMPLATES):
            flaw_id = f"flaw_{i+1}"
            flaw_sentence, flaw_note = ft["inject"](ctx, rng)
            ctx["flaws_text"][ft["section"]] = ctx["flaws_text"].get(ft["section"], "") + " " + flaw_sentence
            flaw_note.update({
                "id":       flaw_id,
                "taxonomy": ft["taxonomy"],
                "location": ft["section"],
            })
            ctx["flaw_notes"].append(flaw_note)
            flaw_ids.append(flaw_id)

        sections = _build_sections(ctx)
        paper_text = _build_paper_text(ctx, sections)

        ground_truth = {
            "flaws":         ctx["flaw_notes"],
            "flaw_sections": list({f["location"] for f in ctx["flaw_notes"]}),
            "flaw_ids":      flaw_ids,
            "n_flaws":       4,
        }

        return {
            "paper_text":     paper_text,
            "paper_sections": sections,
            "dataset_path":   None,
            "ground_truth":   ground_truth,
        }

    def _action_schema(self) -> dict:
        return {
            "read_section":  {"section": "str — e.g. 'abstract', 'methods', 'statistical_analysis', 'results'"},
            "flag_flaw":     {"flaw_type": "str", "location": "str", "description": "str"},
            "submit_audit":  {"audit_payload": {"flaws": "[{flaw_type, location, description}]"}},
        }

    @classmethod
    def generate(cls, seed=None):
        """Convenience method for Task 5 consumption."""
        task = cls(seed=seed)
        ep = task.generate_episode()
        state = {"paper_text": ep["paper_text"], "dataset_path": ep.get("dataset_path")}
        return state, ep["ground_truth"]


# ---------------------------------------------------------------------------
# Flaw injectors — CONSORT protocol violations
# ---------------------------------------------------------------------------

def _inject_unblinded_bias(ctx: dict, rng: random.Random) -> tuple[str, dict]:
    """Inject unblinded investigator bias (CONSORT violation)."""
    outcome = ctx["outcome"]
    sentence = (
        f"The principal investigator, who was unblinded to group allocation, "
        f"personally assessed {outcome} across all {ctx['group_type']}. "
        f"No independent blinded assessor was used for outcome evaluation."
    )
    note = {
        "description": f"Unblinded investigator assessed primary outcome ({outcome}); "
                       f"CONSORT requires blinded outcome assessment to prevent detection bias.",
        "hint_keywords": ["unblinded", "investigator bias", "detection bias", "blinding", "assessor"],
    }
    return sentence, note


def _inject_insufficient_power(ctx: dict, rng: random.Random) -> tuple[str, dict]:
    """Inject an insufficient power analysis (CONSORT violation)."""
    n = ctx["n_per_group"]
    sentence = (
        f"A total of {ctx['total_n']} {ctx['group_type']} were recruited "
        f"({n} per group). No a priori power analysis was conducted "
        f"and no sample size justification was provided per ICH-GCP guidelines."
    )
    note = {
        "description": f"Sample of {n} per group is underpowered for detecting medium effects "
                       f"(typically requires n≥64 per group at 80% power, α=0.05). "
                       f"No power analysis reported — CONSORT Item 7a violation.",
        "hint_keywords": ["underpowered", "sample size", "power analysis", "insufficient power", "ICH-GCP"],
    }
    return sentence, note


def _inject_protocol_deviation(ctx: dict, rng: random.Random) -> tuple[str, dict]:
    """Inject unreported protocol deviation (CONSORT violation)."""
    n_excluded = rng.choice([5, 7, 9, 11])
    original_n = ctx["total_n"]
    reported_n = original_n - n_excluded
    outcome = ctx["outcome"]
    p_val = round(rng.uniform(0.02, 0.048), 3)
    sentence = (
        f"After per-protocol analysis, the {ctx['intervention']} group showed significantly "
        f"improved {outcome} compared to placebo (p={p_val}, n={reported_n}), "
        f"representing a clinically meaningful improvement. Protocol deviations "
        f"were not separately reported in the CONSORT flow diagram."
    )
    note = {
        "description": f"Results reported for n={reported_n} but {original_n} were recruited. "
                       f"{n_excluded} participants excluded from analysis without disclosure "
                       f"in the CONSORT flow diagram — protocol deviation unreported.",
        "hint_keywords": ["protocol deviation", "exclusion", "excluded", "undisclosed", "CONSORT flow"],
        "excluded_n": n_excluded,
        "reported_n": reported_n,
        "original_n": original_n,
    }
    return sentence, note


def _inject_endpoint_switching(ctx: dict, rng: random.Random) -> tuple[str, dict]:
    """Inject endpoint switching (CONSORT violation)."""
    outcomes_tested = [ctx["outcome"], "secondary biomarker", "quality of life score",
                       "adverse event rate", "dropout rate"]
    rng.shuffle(outcomes_tested)
    sentence = (
        f"The pre-registered primary endpoint was {outcomes_tested[1]}, however "
        f"multiple secondary outcomes were assessed including "
        f"{', '.join(outcomes_tested[:3])}. "
        f"Only {ctx['outcome']} reached statistical significance (p=0.043) "
        f"and is reported as the primary outcome in the final analysis."
    )
    note = {
        "description": f"Primary endpoint was switched post-hoc. Multiple outcomes tested "
                       f"({len(outcomes_tested[:3])}) without correction for multiple comparisons. "
                       f"Original primary endpoint did not reach significance — CONSORT violation.",
        "hint_keywords": ["endpoint switching", "primary endpoint", "outcome switching", 
                         "multiple comparison", "selective reporting"],
    }
    return sentence, note


# ---------------------------------------------------------------------------
# Paper builder
# ---------------------------------------------------------------------------

def _build_sections(ctx: dict) -> dict:
    field        = ctx["field"]
    intervention = ctx["intervention"]
    outcome      = ctx["outcome"]
    group_type   = ctx["group_type"]
    total_n      = ctx["total_n"]
    n_per_group  = ctx["n_per_group"]

    abstract = textwrap.dedent(f"""
        Background: This {field} evaluated the efficacy of {intervention} on {outcome}
        in accordance with ICH-GCP guidelines and CONSORT reporting standards.
        We hypothesised that {group_type} receiving {intervention} would demonstrate
        significantly better {outcome} compared to those receiving placebo.
        Methods: A randomised controlled trial design was employed per IRB approval.
        Results: {intervention} produced statistically significant improvements.
        Conclusion: These findings support adoption of {intervention} in clinical practice.
    """).strip()

    participants = textwrap.dedent(f"""
        {total_n} {group_type} were enrolled from three clinical sites between 2021 and 2023
        under IRB protocol #2021-CT-{ctx['rng'].randint(100,999)}.
        Inclusion criteria: aged 18–65, no prior treatment exposure, written informed
        consent obtained per Declaration of Helsinki.
        Exclusion criteria: severe comorbidities, inability to complete clinical assessments.
        {ctx['flaws_text'].get('participants', '')}
        Participants were randomly assigned to {intervention} (n={n_per_group})
        or placebo (n={n_per_group}) using block randomisation (block size=4)
        per the CONSORT-compliant allocation sequence.
    """).strip()

    statistical_analysis = textwrap.dedent(f"""
        All analyses were performed using SAS v9.4 and Python 3.10 per the
        pre-registered Statistical Analysis Plan (SAP).
        The primary outcome ({outcome}) was compared between groups at 12 weeks.
        {ctx['flaws_text'].get('statistical_analysis', '')}
        Significance threshold was set at α=0.05. Missing data handled via last
        observation carried forward (LOCF). No corrections for multiple comparisons
        were pre-specified in the protocol or SAP.
    """).strip()

    results = textwrap.dedent(f"""
        {ctx['flaws_text'].get('results', '')}
        Secondary analysis of subgroups by age and sex showed consistent direction
        of effect. Adverse events were minor and balanced across arms (p=0.71).
        Full CONSORT flow diagram available in supplementary materials.
    """).strip()

    discussion = textwrap.dedent(f"""
        The present study demonstrates that {intervention} significantly improves
        {outcome} in {group_type}. These results are consistent with prior
        mechanistic studies and meet the bar for regulatory consideration.
        Limitations include the single-blind design and
        relatively short follow-up period of 12 weeks. Future work should
        examine long-term durability of effects and dose-response relationships
        per FDA post-marketing surveillance requirements.
        Generalisability may be limited to populations similar to those studied.
    """).strip()

    return {
        "abstract":              abstract,
        "participants":          participants,
        "statistical_analysis":  statistical_analysis,
        "results":               results,
        "discussion":            discussion,
    }


def _build_paper_text(ctx: dict, sections: dict) -> str:
    intervention = ctx["intervention"]
    outcome      = ctx["outcome"]
    field        = ctx["field"]
    return textwrap.dedent(f"""
        TITLE: Efficacy of {intervention} on {outcome}: A CONSORT-compliant randomised controlled {field}

        ABSTRACT
        {sections['abstract']}

        1. PARTICIPANTS (CONSORT Items 3-5)
        {sections['participants']}

        2. STATISTICAL ANALYSIS (CONSORT Item 12)
        {sections['statistical_analysis']}

        3. RESULTS (CONSORT Items 13-19)
        {sections['results']}

        4. DISCUSSION (CONSORT Items 20-22)
        {sections['discussion']}

        ---
        Available sections for read_section: abstract, participants,
        statistical_analysis, results, discussion
    """).strip()
