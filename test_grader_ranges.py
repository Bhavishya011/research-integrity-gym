#!/usr/bin/env python3
"""Test that all graders return scores strictly between 0 and 1."""
import sys
sys.path.insert(0, ".")

from graders.grader1 import grade_audit
from graders.grader2 import grade_results
from graders.grader3 import grade_verdict
from graders.grader4 import grade_citation_report
from env.models import (
    SubmitAuditPayload, FlawReport,
    SubmitResultsPayload,
    SubmitVerdictPayload, Verdict,
    SubmitCitationReportPayload,
)

print("Testing grader score ranges...\n")

# Test 1: Perfect score (should be < 1.0)
print("Test 1: Perfect scores (should be 0.9999)")
payload1 = SubmitAuditPayload(flaws=[
    FlawReport(flaw_type="wrong_statistical_test", location="statistical_analysis", description="Used t-test"),
    FlawReport(flaw_type="underpowered_sample", location="participants", description="n=15"),
    FlawReport(flaw_type="undisclosed_exclusion", location="results", description="Excluded outliers"),
    FlawReport(flaw_type="p_value_manipulation", location="results", description="p=0.049"),
])
gt1 = {
    "flaws": [
        {"id": "f1", "taxonomy": "wrong_statistical_test", "location": "statistical_analysis"},
        {"id": "f2", "taxonomy": "underpowered_sample", "location": "participants"},
        {"id": "f3", "taxonomy": "undisclosed_exclusion", "location": "results"},
        {"id": "f4", "taxonomy": "p_value_manipulation", "location": "results"},
    ]
}
score1 = grade_audit(payload1, gt1)
print(f"  Grader 1 (perfect): {score1} {'✅' if 0 < score1 < 1 else '❌'}")

# Test 2: Zero score (should be > 0.0)
print("\nTest 2: Zero scores (should be 0.0001)")
payload2 = SubmitAuditPayload(flaws=[])
score2 = grade_audit(payload2, gt1)
print(f"  Grader 1 (empty): {score2} {'✅' if 0 < score2 < 1 else '❌'}")

payload3 = SubmitResultsPayload(auc=0.5, f1=0.5, interpretation="wrong")
gt3 = {"auc_ground_truth": 0.85, "f1_ground_truth": 0.72}
score3 = grade_results(payload3, gt3)
print(f"  Grader 2 (wrong): {score3} {'✅' if 0 < score3 < 1 else '❌'}")

payload4 = SubmitVerdictPayload(
    verdict=Verdict.invalid,
    effect_size=0.0,
    p_value=0.5,
    justification="The statistical analysis is incorrect. " * 10  # 100+ chars
)
gt4 = {"claim_is_valid": True, "effect_size": 0.45, "p_value": 0.003}
score4 = grade_verdict(payload4, gt4)
print(f"  Grader 3 (wrong): {score4} {'✅' if 0 < score4 < 1 else '❌'}")

payload5 = SubmitCitationReportPayload(
    fabricated_citation_id=None,
    fabrication_type="",
    verified_correct_citations=[],
    evidence="No evidence provided at all." * 5  # 20+ chars
)
gt5 = {
    "fabricated_id": 1,
    "fabrication_type": "magnitude",
    "correct_citation_ids": [2, 3],
    "excerpt_keywords": ["18%", "children"]
}
score5 = grade_citation_report(payload5, gt5)
print(f"  Grader 4 (empty): {score5} {'✅' if 0 < score5 < 1 else '❌'}")

# Verify all scores are in valid range
all_valid = all(0 < s < 1 for s in [score1, score2, score3, score4, score5])

print(f"\n{'='*50}")
if all_valid:
    print("✅ PASS: All graders return scores strictly between 0 and 1")
    sys.exit(0)
else:
    print("❌ FAIL: Some graders return 0.0 or 1.0")
    sys.exit(1)
