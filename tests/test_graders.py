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


# ---------------------------------------------------------------------------
# Import graders
# ---------------------------------------------------------------------------

from graders.grader1 import grade_audit
from graders.grader2 import grade_replication
from graders.grader3 import grade_verification


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
        """Exact match of AUC and F1 → 1.0"""
        ground_truth = {
            "auc": 0.85,
            "f1": 0.72,
        }
        payload = MockResultsPayload(auc=0.85, f1=0.72, interpretation="Good model")
        
        score = grade_replication(payload, ground_truth)
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_within_tolerance(self):
        """AUC and F1 within ±0.03 tolerance → full credit for those components"""
        ground_truth = {
            "auc": 0.85,
            "f1": 0.72,
        }
        payload = MockResultsPayload(auc=0.87, f1=0.70, interpretation="Good model")
        
        score = grade_replication(payload, ground_truth)
        # Both within tolerance: 0.45 (AUC) + 0.35 (F1) + 0.20 (interpretation) = 1.0
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_way_off_metrics(self):
        """Completely wrong metrics → low score"""
        ground_truth = {
            "auc": 0.85,
            "f1": 0.72,
        }
        payload = MockResultsPayload(auc=0.50, f1=0.30, interpretation="Bad model")
        
        score = grade_replication(payload, ground_truth)
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_no_interpretation_penalty(self):
        """Empty interpretation → lose interpretation points"""
        ground_truth = {
            "auc": 0.85,
            "f1": 0.72,
        }
        payload = MockResultsPayload(auc=0.85, f1=0.72, interpretation="")
        
        score = grade_replication(payload, ground_truth)
        # Perfect metrics but no interpretation: 0.45 + 0.35 + 0.0 = 0.80
        assert score == 0.8, f"Expected 0.8, got {score}"


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
        payload = MockVerdictPayload(
            verdict="invalid",
            effect_size=2.4,
            p_value=0.11,
            justification="The data shows undisclosed exclusion of outliers which invalidates the claimed significant result."
        )
        
        score = grade_verification(payload, ground_truth)
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
        payload = MockVerdictPayload(
            verdict="valid",  # WRONG
            effect_size=2.5,
            p_value=0.03,
            justification="The study looks fine to me."
        )
        
        score = grade_verification(payload, ground_truth)
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_justification_keywords_matter(self):
        """Missing required keywords → lose justification points"""
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
        payload = MockVerdictPayload(
            verdict="invalid",
            effect_size=2.5,
            p_value=0.12,
            justification="The statistics do not support the claim. The p-value is above threshold."  # No exclusion keywords
        )
        
        score = grade_verification(payload, ground_truth)
        # Should score lower due to missing keywords about exclusion
        assert score < 0.9, f"Expected < 0.9 due to missing keywords, got {score}"


# ===========================================================================
# Run tests
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
