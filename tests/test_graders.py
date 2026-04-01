"""
Unit tests for graders — deterministic scoring verification.

Run with: pytest tests/test_graders.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Mock classes to avoid importing pydantic models in tests
# ---------------------------------------------------------------------------

@dataclass
class MockFlawReport:
    flaw_type: str
    location: str
    description: str


@dataclass
class MockAuditPayload:
    flaws: List[MockFlawReport]


@dataclass
class MockResultsPayload:
    auc: float
    f1: float
    interpretation: str


@dataclass
class MockVerdictPayload:
    verdict: str
    effect_size: float
    p_value: float
    justification: str
    
    @property 
    def verdict_value(self):
        """Mock the .verdict.value pattern used by Pydantic enum."""
        return self.verdict


# Wrapper to handle enum vs string
class VerdictWrapper:
    def __init__(self, value: str):
        self.value = value


@dataclass  
class MockVerdictPayloadWithEnum:
    """Payload that mimics the real Pydantic model with enum."""
    effect_size: float
    p_value: float
    justification: str
    _verdict: str = ""
    
    def __init__(self, verdict: str, effect_size: float, p_value: float, justification: str):
        self._verdict = verdict
        self.effect_size = effect_size
        self.p_value = p_value
        self.justification = justification
    
    @property
    def verdict(self):
        return VerdictWrapper(self._verdict)


# ---------------------------------------------------------------------------
# Import graders
# ---------------------------------------------------------------------------

from graders.grader1 import grade_audit
from graders.grader2 import grade_results
from graders.grader3 import grade_verdict


# ===========================================================================
# Grader 1 Tests: Methodology Audit
# ===========================================================================

class TestGrader1:
    """Tests for the methodology audit grader."""

    def test_perfect_score_all_four_flaws(self):
        """Agent identifies all 4 flaws correctly → 1.0"""
        ground_truth = {
            "flaws": [
                {"id": "flaw_1", "taxonomy": "wrong_statistical_test", "location": "statistical_analysis"},
                {"id": "flaw_2", "taxonomy": "underpowered_sample", "location": "participants"},
                {"id": "flaw_3", "taxonomy": "undisclosed_exclusion", "location": "results"},
                {"id": "flaw_4", "taxonomy": "p_value_manipulation", "location": "results"},
            ]
        }
        payload = MockAuditPayload(flaws=[
            MockFlawReport("wrong statistical test", "statistical_analysis", "Chi-square on continuous"),
            MockFlawReport("underpowered sample", "participants", "n=26 is too small"),
            MockFlawReport("undisclosed exclusion", "results", "52 recruited but only 45 analyzed"),
            MockFlawReport("p-value manipulation", "results", "Multiple outcomes tested"),
        ])
        
        score = grade_audit(payload, ground_truth)
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_zero_score_no_flaws(self):
        """Agent submits empty flaws → 0.0"""
        ground_truth = {
            "flaws": [
                {"id": "flaw_1", "taxonomy": "wrong_statistical_test", "location": "statistical_analysis"},
            ]
        }
        payload = MockAuditPayload(flaws=[])
        
        score = grade_audit(payload, ground_truth)
        assert score == 0.0

    def test_partial_credit_two_flaws(self):
        """Agent identifies 2 of 4 flaws → 0.5"""
        ground_truth = {
            "flaws": [
                {"id": "flaw_1", "taxonomy": "wrong_statistical_test", "location": "statistical_analysis"},
                {"id": "flaw_2", "taxonomy": "underpowered_sample", "location": "participants"},
                {"id": "flaw_3", "taxonomy": "undisclosed_exclusion", "location": "results"},
                {"id": "flaw_4", "taxonomy": "p_value_manipulation", "location": "results"},
            ]
        }
        payload = MockAuditPayload(flaws=[
            MockFlawReport("wrong test", "statistical_analysis", "Chi-square used incorrectly"),
            MockFlawReport("small sample size", "participants", "Only 26 per group"),
        ])
        
        score = grade_audit(payload, ground_truth)
        assert score == 0.5, f"Expected 0.5, got {score}"

    def test_false_positive_penalty(self):
        """False positives reduce score (capped at -0.20)"""
        ground_truth = {
            "flaws": [
                {"id": "flaw_1", "taxonomy": "wrong_statistical_test", "location": "statistical_analysis"},
            ]
        }
        # Agent identifies the real flaw (0.25) but adds 5 false positives
        payload = MockAuditPayload(flaws=[
            MockFlawReport("chi-square wrong", "statistical_analysis", "correct"),
            MockFlawReport("fake flaw 1", "abstract", "not real"),
            MockFlawReport("fake flaw 2", "abstract", "not real"),
            MockFlawReport("fake flaw 3", "abstract", "not real"),
            MockFlawReport("fake flaw 4", "abstract", "not real"),
            MockFlawReport("fake flaw 5", "abstract", "not real"),
        ])
        
        score = grade_audit(payload, ground_truth)
        # 0.25 (correct) - 0.20 (capped FP penalty) = 0.05
        assert score == 0.05, f"Expected 0.05, got {score}"

    def test_synonym_matching_inappropriate_method(self):
        """Synonym 'inappropriate method' should match wrong_statistical_test"""
        ground_truth = {
            "flaws": [
                {"id": "flaw_1", "taxonomy": "wrong_statistical_test", "location": "statistical_analysis"},
            ]
        }
        payload = MockAuditPayload(flaws=[
            MockFlawReport("inappropriate statistical method", "statistical_analysis", "wrong approach"),
        ])
        
        score = grade_audit(payload, ground_truth)
        assert score == 0.25, f"Expected 0.25, got {score}"

    def test_partial_credit_wrong_location(self):
        """Right flaw type, wrong location → 0.10 partial credit"""
        ground_truth = {
            "flaws": [
                {"id": "flaw_1", "taxonomy": "wrong_statistical_test", "location": "statistical_analysis"},
            ]
        }
        payload = MockAuditPayload(flaws=[
            MockFlawReport("wrong test", "methods", "chi-square issue"),  # wrong location
        ])
        
        score = grade_audit(payload, ground_truth)
        assert score == 0.10, f"Expected 0.10, got {score}"


# ===========================================================================
# Grader 2 Tests: Experiment Replication
# ===========================================================================

class TestGrader2:
    """Tests for the replication grader."""

    def test_perfect_replication(self):
        """Exact match of AUC and F1 with keywords → 1.0"""
        ground_truth = {
            "auc": 0.85,
            "f1": 0.72,
        }
        # Need interpretation keywords for full score
        payload = MockResultsPayload(
            auc=0.85, f1=0.72, 
            interpretation="The model handles class imbalance well with balanced weights"
        )
        
        score = grade_results(payload, ground_truth)
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_within_tight_tolerance(self):
        """AUC and F1 within ±0.01 → 0.45 + 0.35 = 0.80 (no interp keywords)"""
        ground_truth = {
            "auc": 0.85,
            "f1": 0.72,
        }
        payload = MockResultsPayload(auc=0.85, f1=0.72, interpretation="Good model")
        
        score = grade_results(payload, ground_truth)
        # 0.45 (AUC) + 0.35 (F1) + 0.0 (no keywords) = 0.80
        assert score == 0.8, f"Expected 0.8, got {score}"

    def test_way_off_metrics(self):
        """Completely wrong metrics → low score"""
        ground_truth = {
            "auc": 0.85,
            "f1": 0.72,
        }
        payload = MockResultsPayload(auc=0.50, f1=0.30, interpretation="Bad model")
        
        score = grade_results(payload, ground_truth)
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_interpretation_keywords_add_points(self):
        """Interpretation with 2+ keywords adds 0.20"""
        ground_truth = {
            "auc": 0.85,
            "f1": 0.72,
        }
        payload = MockResultsPayload(
            auc=0.85, f1=0.72, 
            interpretation="Class imbalance handled with stratified sampling"
        )
        
        score = grade_results(payload, ground_truth)
        # 0.45 + 0.35 + 0.20 = 1.0
        assert score == 1.0, f"Expected 1.0, got {score}"


# ===========================================================================
# Grader 3 Tests: Claim Verification
# ===========================================================================

class TestGrader3:
    """Tests for the claim verification grader."""

    def test_perfect_verdict_invalid(self):
        """Correctly identify invalid claim with evidence → high score"""
        ground_truth = {
            "true_verdict": "invalid",
            "true_effect": 2.5,
            "true_p": 0.12,
            "effect_tolerance": 0.5,
            "p_threshold": 0.05,
            "has_undisclosed_exclusion": True,
            "required_keywords": ["exclusion", "excluded", "outlier", "undisclosed"],
            "required_keyword_count": 2,
        }
        # Long justification to pass coherence check (20+ words, 40%+ unique)
        payload = MockVerdictPayloadWithEnum(
            verdict="invalid",
            effect_size=2.4,
            p_value=0.11,
            justification="The data shows clear evidence of undisclosed exclusion of outliers from the treatment group which completely invalidates the claimed statistically significant result. The authors removed participants without proper documentation or justification in the methods section."
        )
        
        score = grade_verdict(payload, ground_truth)
        assert score >= 0.9, f"Expected >= 0.9, got {score}"

    def test_wrong_verdict(self):
        """Agent says valid when it's invalid → lose verdict points"""
        ground_truth = {
            "true_verdict": "invalid",
            "true_effect": 2.5,
            "true_p": 0.12,
            "effect_tolerance": 0.5,
            "p_threshold": 0.05,
            "has_undisclosed_exclusion": True,
            "required_keywords": ["exclusion", "excluded", "outlier", "undisclosed"],
            "required_keyword_count": 2,
        }
        payload = MockVerdictPayloadWithEnum(
            verdict="valid",  # WRONG
            effect_size=2.5,
            p_value=0.03,  # Also wrong - claims significant
            justification="The study methodology appears sound and the statistical analysis is appropriate for the research design presented."
        )
        
        score = grade_verdict(payload, ground_truth)
        # Wrong verdict, wrong p-value direction, no exclusion keywords
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_correct_verdict_no_keywords(self):
        """Correct verdict but missing exclusion keywords → partial score"""
        ground_truth = {
            "true_verdict": "invalid",
            "true_effect": 2.5,
            "true_p": 0.12,
            "effect_tolerance": 0.5,
            "p_threshold": 0.05,
            "has_undisclosed_exclusion": True,
            "required_keywords": ["exclusion", "excluded", "outlier", "undisclosed"],
            "required_keyword_count": 2,
        }
        payload = MockVerdictPayloadWithEnum(
            verdict="invalid",
            effect_size=2.5,
            p_value=0.12,
            justification="The statistics do not support the claim because the p-value is clearly above the significance threshold of point zero five which means we cannot reject the null hypothesis."
        )
        
        score = grade_verdict(payload, ground_truth)
        # Correct verdict (0.35) + effect (0.20) + p-value (0.15) = 0.70
        # But missing exclusion keywords so no exclusion bonus
        assert 0.6 <= score <= 0.75, f"Expected 0.6-0.75, got {score}"


# ---------------------------------------------------------------------------
# Grader 4 — Citation Integrity Check
# ---------------------------------------------------------------------------

class MockCitationReportPayload:
    def __init__(self, fabricated_citation_id, fabrication_type, 
                 verified_correct_citations, evidence):
        self.fabricated_citation_id = fabricated_citation_id
        self.fabrication_type = fabrication_type
        self.verified_correct_citations = verified_correct_citations
        self.evidence = evidence


class TestGrader4:
    """Test suite for Task 4: Citation Integrity Check grader"""

    def test_perfect_identification(self):
        """All components correct → 1.0 score"""
        ground_truth = {
            "fabricated_id": 2,
            "fabrication_type": "directional",
            "correct_citation_ids": [1, 3],
            "excerpt_keywords": ["decreased", "performance", "control group"],
        }
        payload = MockCitationReportPayload(
            fabricated_citation_id=2,
            fabrication_type="directional reversal - paper claims increased but source says decreased",
            verified_correct_citations=[1, 3],
            evidence="The excerpt clearly states 'decreased performance in the control group' but the paper claims increased performance"
        )

        from graders.grader4 import grade_citation_report
        score = grade_citation_report(payload, ground_truth)
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_wrong_citation_id(self):
        """Identified wrong citation → lose 0.40 points"""
        ground_truth = {
            "fabricated_id": 2,
            "fabrication_type": "magnitude",
            "correct_citation_ids": [1, 3],
            "excerpt_keywords": ["2.5%", "improvement"],
        }
        payload = MockCitationReportPayload(
            fabricated_citation_id=1,  # WRONG
            fabrication_type="magnitude error",
            verified_correct_citations=[2, 3],
            evidence="The excerpt shows 2.5% improvement"
        )

        from graders.grader4 import grade_citation_report
        score = grade_citation_report(payload, ground_truth)
        # Lost 0.40 for wrong ID, got 0.30 for type, got 0.08 for partial verification, got 0.15 for evidence
        # Should be around 0.50-0.60
        assert 0.40 <= score <= 0.65, f"Expected 0.40-0.65, got {score}"

    def test_type_synonym_matching(self):
        """Various ways to describe the same fabrication type should match"""
        ground_truth = {
            "fabricated_id": 1,
            "fabrication_type": "population",
            "correct_citation_ids": [2, 3],
            "excerpt_keywords": ["adults", "age 25-65"],
        }
        
        # Test different phrasings that should all match "population"
        test_descriptions = [
            "wrong population - generalized to children",
            "population mismatch between study and claim",
            "demographic issue - different age group",
            "applied adults study to adolescents",
        ]

        from graders.grader4 import grade_citation_report
        for desc in test_descriptions:
            payload = MockCitationReportPayload(
                fabricated_citation_id=1,
                fabrication_type=desc,
                verified_correct_citations=[2, 3],
                evidence="Source studied adults age 25-65 only"
            )
            score = grade_citation_report(payload, ground_truth)
            # Should get: 0.40 (ID) + 0.30 (type) + 0.15 (verification) + 0.15 (evidence) = 1.0
            assert score >= 0.95, f"Description '{desc}' should match 'population', got {score}"

    def test_partial_evidence(self):
        """Evidence with only 1 keyword → partial credit"""
        ground_truth = {
            "fabricated_id": 3,
            "fabrication_type": "significance",
            "correct_citation_ids": [1, 2],
            "excerpt_keywords": ["p-value", "not significant", "p>0.05"],
        }
        payload = MockCitationReportPayload(
            fabricated_citation_id=3,
            fabrication_type="significance flip",
            verified_correct_citations=[1, 2],
            evidence="The source shows p-value above threshold"  # Has "p-value" but not the others
        )

        from graders.grader4 import grade_citation_report
        score = grade_citation_report(payload, ground_truth)
        # 0.40 + 0.30 + 0.15 + 0.08 (partial evidence) = 0.93
        assert 0.88 <= score <= 0.95, f"Expected 0.88-0.95, got {score}"

    def test_no_evidence(self):
        """Empty or minimal evidence → no evidence points"""
        ground_truth = {
            "fabricated_id": 1,
            "fabrication_type": "absent",
            "correct_citation_ids": [2, 3],
            "excerpt_keywords": ["never mentioned", "not in source"],
        }
        payload = MockCitationReportPayload(
            fabricated_citation_id=1,
            fabrication_type="finding absent from citation",
            verified_correct_citations=[2, 3],
            evidence="Bad"  # Too short, no keywords
        )

        from graders.grader4 import grade_citation_report
        score = grade_citation_report(payload, ground_truth)
        # 0.40 + 0.30 + 0.15 + 0.0 = 0.85
        assert 0.83 <= score <= 0.87, f"Expected 0.83-0.87, got {score}"



# ===========================================================================
# Run tests
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
