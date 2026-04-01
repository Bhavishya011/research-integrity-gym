"""
Grader 4: Citation Integrity Check scorer
Scores 0.0-1.0. 100% deterministic text matching. No LLM judge.

Scoring:
  Identified correct fabricated citation    0.40
  Correct fabrication type description      0.30
  Verified other citations are accurate     0.15
  Cited specific evidence from excerpt      0.15
"""
from __future__ import annotations


# Fabrication type synonyms for matching
_FABRICATION_KEYWORDS = {
    "directional": [
        "reverse", "opposite", "wrong direction", "flipped", "increases vs decreases",
        "decreases vs increases", "improved vs worsened", "directional",
        "contradiction", "contradicts", "says opposite",
    ],
    "magnitude": [
        "magnitude", "amount", "factor", "off by", "exaggerated", "understated",
        "10x", "tenfold", "wrong amount", "wrong percentage", "wrong number",
        "2.5% vs 25%", "25% vs 2.5%",
    ],
    "population": [
        "population", "demographic", "age group", "generalization", "different group",
        "adults vs children", "children vs adults", "elderly", "adolescents",
        "wrong population", "different subjects",
    ],
    "significance": [
        "p-value", "significance", "significant", "statistical", "p<0.05", "p>0.05",
        "not significant", "no significant", "claimed significant", "false significance",
    ],
    "absent": [
        "not mentioned", "doesn't say", "absent", "missing", "not in citation",
        "citation doesn't mention", "never states", "not found", "fabricated",
        "completely made up",
    ],
}


def _type_matches(described_type: str, ground_truth_type: str) -> bool:
    """Check if the agent's description matches the ground truth fabrication type."""
    desc = described_type.lower().strip()
    keywords = _FABRICATION_KEYWORDS.get(ground_truth_type, [])
    return any(kw in desc for kw in keywords)


def grade_citation_report(payload, ground_truth: dict) -> float:
    """
    payload: object with attributes:
      - fabricated_citation_id: int | None
      - fabrication_type: str
      - verified_correct_citations: list[int]
      - evidence: str
    
    ground_truth: dict with:
      - fabricated_id: int
      - fabrication_type: str
      - correct_citation_ids: list[int]
      - excerpt_keywords: list[str]
    """
    gt_fabricated_id = ground_truth.get("fabricated_id")
    gt_type = ground_truth.get("fabrication_type")
    gt_correct_ids = set(ground_truth.get("correct_citation_ids", []))
    gt_keywords = ground_truth.get("excerpt_keywords", [])

    # Extract agent response
    agent_id = payload.fabricated_citation_id
    agent_type = (payload.fabrication_type or "").strip()
    agent_verified = set(payload.verified_correct_citations or [])
    agent_evidence = (payload.evidence or "").lower()

    score = 0.0

    # 1. Identified correct fabricated citation (0.40)
    if agent_id == gt_fabricated_id:
        score += 0.40

    # 2. Correct fabrication type (0.30)
    if _type_matches(agent_type, gt_type):
        score += 0.30
    elif agent_type:
        # Partial credit if they at least tried to describe the problem
        if len(agent_type) > 10:  # Substantial description
            score += 0.10

    # 3. Verified other citations are accurate (0.15)
    if agent_verified == gt_correct_ids:
        score += 0.15
    elif agent_verified and len(agent_verified & gt_correct_ids) > 0:
        # Partial credit if they verified at least some correct ones
        overlap_ratio = len(agent_verified & gt_correct_ids) / len(gt_correct_ids)
        score += 0.15 * overlap_ratio

    # 4. Cited specific evidence (0.15)
    # Check if agent quoted relevant keywords from the excerpt
    keyword_hits = sum(1 for kw in gt_keywords if kw.lower() in agent_evidence)
    if keyword_hits >= 2:
        score += 0.15
    elif keyword_hits == 1:
        score += 0.08

    return round(max(0.0, min(1.0, score)), 4)
