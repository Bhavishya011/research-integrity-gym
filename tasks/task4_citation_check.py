"""
Task 4: Citation Integrity Check — MEDIUM-HARD
-----------------------------------------------
Agent reads a research paper that cites 3-4 sources. ONE citation is fabricated —
the paper's claim doesn't match what the cited source actually says.

Agent must cross-reference claims against provided citation excerpts and flag
the misrepresentation.

This addresses a critical LLM weakness: citation hallucination.
"""
from __future__ import annotations

import random
import textwrap
from typing import Optional

from tasks.base import BaseTask


# ---------------------------------------------------------------------------
# Domain vocabularies for procedural generation
# ---------------------------------------------------------------------------

DOMAINS = [
    {
        "field": "education",
        "intervention": ["gamified apps", "video-based learning", "peer tutoring", "flipped classroom"],
        "outcome": ["student engagement", "test scores", "retention rates", "dropout rates"],
        "population": ["elementary students", "middle school students", "high school students", "college students"],
    },
    {
        "field": "clinical psychology",
        "intervention": ["cognitive behavioral therapy", "mindfulness training", "group therapy", "digital interventions"],
        "outcome": ["anxiety levels", "depression scores", "quality of life", "symptom severity"],
        "population": ["adults", "adolescents", "elderly patients", "college students"],
    },
    {
        "field": "public health",
        "intervention": ["vaccination campaigns", "nutrition programs", "exercise interventions", "smoking cessation"],
        "outcome": ["disease incidence", "mortality rates", "BMI", "health outcomes"],
        "population": ["adults", "children", "elderly", "at-risk populations"],
    },
]


FABRICATION_TYPES = [
    "directional",    # increases vs decreases
    "magnitude",      # 25% vs 2.5%
    "population",     # children vs adults
    "significance",   # p<0.05 vs p>0.05
    "absent",         # claim not in citation
]


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

class CitationCheckTask(BaseTask):
    task_id    = "task4_citation_check"
    task_name  = "Citation Integrity Check"
    difficulty = "medium-hard"
    max_steps  = 20

    def generate_episode(self) -> dict:
        rng = self.rng
        domain = rng.choice(DOMAINS)

        # Generate 3 citations
        n_citations = 3
        citations = []
        for i in range(n_citations):
            citations.append(self._generate_citation(domain, i + 1, rng))

        # Pick one to fabricate
        fabricated_idx = rng.randint(0, n_citations - 1)
        fabrication_type = rng.choice(FABRICATION_TYPES)
        
        # Apply fabrication
        fabricated_cite = citations[fabricated_idx]
        self._apply_fabrication(fabricated_cite, fabrication_type, domain, rng)

        # Build paper text
        paper_text = self._build_paper(domain, citations, rng)

        # Build citations-only section for read_section("citations")
        citations_section = "\n\n".join(
            f"[{c['id']}] {c['author']} ({c['year']}): \"{c['excerpt']}\""
            for c in citations
        )

        ground_truth = {
            "fabricated_id": fabricated_idx + 1,
            "fabrication_type": fabrication_type,
            "correct_citation_ids": [i + 1 for i in range(n_citations) if i != fabricated_idx],
            "paper_claim": fabricated_cite["paper_claim"],
            "actual_excerpt": fabricated_cite["excerpt"],
            "excerpt_keywords": fabricated_cite["excerpt_keywords"],
            "all_citations": citations,
        }

        return {
            "paper_text": paper_text,
            "paper_sections": {
                "introduction": paper_text,
                "citations": citations_section,
            },
            "dataset_path": None,
            "ground_truth": ground_truth,
        }

    def _generate_citation(self, domain: dict, citation_id: int, rng: random.Random) -> dict:
        """Generate a realistic citation with author, year, and excerpt."""
        authors = rng.choice([
            ["Smith", "Jones"],
            ["Martinez", "Chen"],
            ["Wang", "Lee", "Park"],
            ["Williams"],
            ["Johnson", "Brown"],
        ])
        
        if len(authors) == 1:
            author_str = f"{authors[0]} et al."
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        else:
            author_str = f"{authors[0]} et al."

        year = rng.randint(2018, 2023)
        
        intervention = rng.choice(domain["intervention"])
        outcome = rng.choice(domain["outcome"])
        population = rng.choice(domain["population"])
        
        # Generate a realistic finding
        effect_size = rng.choice([12, 15, 18, 22, 25, 28, 34, 40])
        direction = rng.choice(["increased", "decreased", "improved", "reduced"])
        p_value = round(rng.uniform(0.001, 0.048), 3)
        
        excerpt = (
            f"{intervention} {direction} {outcome} by {effect_size}% "
            f"in {population} (95% CI: {effect_size-8}-{effect_size+10}%, p={p_value})"
        )

        return {
            "id": citation_id,
            "author": author_str,
            "year": year,
            "excerpt": excerpt,
            "excerpt_keywords": [direction, str(effect_size), outcome],
            "is_fabricated": False,
            "paper_claim": f"{author_str} ({year}) found {intervention} {direction} {outcome} by {effect_size}%",
            "intervention": intervention,
            "outcome": outcome,
            "effect_size": effect_size,
            "direction": direction,
            "p_value": p_value,
            "population": population,
        }

    def _apply_fabrication(self, citation: dict, fab_type: str, domain: dict, rng: random.Random):
        """Apply a specific type of fabrication to a citation."""
        citation["is_fabricated"] = True
        citation["fabrication_type"] = fab_type

        if fab_type == "directional":
            # Reverse the direction in the paper claim
            direction_map = {
                "increased": "decreased",
                "decreased": "increased",
                "improved": "worsened",
                "reduced": "increased",
            }
            wrong_direction = direction_map.get(citation["direction"], "decreased")
            citation["paper_claim"] = (
                f"{citation['author']} ({citation['year']}) found "
                f"{citation['intervention']} {wrong_direction} "
                f"{citation['outcome']} by {citation['effect_size']}%"
            )
            citation["excerpt_keywords"].append("directional")

        elif fab_type == "magnitude":
            # Change magnitude by factor of 10
            wrong_size = citation["effect_size"] // 10 if citation["effect_size"] >= 20 else citation["effect_size"] * 10
            citation["paper_claim"] = (
                f"{citation['author']} ({citation['year']}) found "
                f"{citation['intervention']} {citation['direction']} "
                f"{citation['outcome']} by {wrong_size}%"
            )
            citation["excerpt_keywords"].append("magnitude")

        elif fab_type == "population":
            # Change population
            populations = domain["population"]
            wrong_pop = rng.choice([p for p in populations if p != citation["population"]])
            citation["paper_claim"] = (
                f"{citation['author']} ({citation['year']}) found "
                f"{citation['intervention']} {citation['direction']} "
                f"{citation['outcome']} by {citation['effect_size']}% in {wrong_pop}"
            )
            citation["excerpt_keywords"].append(citation["population"])

        elif fab_type == "significance":
            # Claim significance when there is none
            citation["paper_claim"] = (
                f"{citation['author']} ({citation['year']}) found "
                f"{citation['intervention']} significantly {citation['direction']} "
                f"{citation['outcome']} (p<0.05)"
            )
            # Change excerpt to non-significant
            citation["excerpt"] = (
                f"{citation['intervention']} showed no significant effect on "
                f"{citation['outcome']} in {citation['population']} (p=0.18)"
            )
            citation["excerpt_keywords"] = ["no significant", "p=0.18"]

        elif fab_type == "absent":
            # Claim finding that's not in the excerpt
            fake_outcome = rng.choice([o for o in domain["outcome"] if o != citation["outcome"]])
            citation["paper_claim"] = (
                f"{citation['author']} ({citation['year']}) found "
                f"{citation['intervention']} {citation['direction']} "
                f"{fake_outcome} by {citation['effect_size']}%"
            )
            # Don't add "absent" as a keyword - add the fake_outcome instead
            citation["excerpt_keywords"] = [fake_outcome, "not mentioned", citation["outcome"]]

    def _build_paper(self, domain: dict, citations: list, rng: random.Random) -> str:
        """Build the full paper text with citations."""
        field = domain["field"]
        
        citation_texts = []
        for cite in citations:
            citation_texts.append(f"  - {cite['paper_claim']}")

        citation_excerpts = []
        for cite in citations:
            citation_excerpts.append(
                f"[{cite['id']}] {cite['author']} ({cite['year']}): \"{cite['excerpt']}\""
            )

        paper = textwrap.dedent(f"""
            TITLE: Meta-analysis of interventions in {field}

            INTRODUCTION
            Recent research has established important findings in {field}:
            {chr(10).join(citation_texts)}

            We conducted a comprehensive meta-analysis to synthesize these findings
            and identify patterns across studies.

            METHODS
            We searched PubMed, PsycINFO, and Google Scholar for relevant studies
            published between 2018-2023. Inclusion criteria required randomized
            controlled trials with validated outcome measures.

            CITATION EXCERPTS PROVIDED:
            {chr(10).join(citation_excerpts)}

            ---
            Your task: Cross-reference the claims in the introduction against
            the citation excerpts. One citation is misrepresented. Identify which
            citation is fabricated and explain the discrepancy.

            Available actions:
              - read_section
              - check_citation (to review specific citations)
              - flag_fabrication (when you find the misrepresentation)
              - submit_report (your final verdict)
        """).strip()

        return paper

    def _action_schema(self) -> dict:
        return {
            "read_section": {"section": "str — introduction | methods | citations"},
            "check_citation": {"citation_id": "int — 1, 2, or 3"},
            "flag_fabrication": {
                "citation_id": "int",
                "fabrication_type": "str — directional | magnitude | population | significance | absent",
                "evidence": "str — quote discrepancy"
            },
            "submit_report": {
                "report_payload": {
                    "fabricated_citation_id": "int | null — which citation is fake",
                    "fabrication_type": "str — type of error",
                    "verified_correct_citations": "[int] — which citations are accurate",
                    "evidence": "str — specific quote showing the mismatch"
                }
            },
        }
