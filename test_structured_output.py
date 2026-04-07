#!/usr/bin/env python3
"""Test that inference.py outputs the required structured format."""
import subprocess
import sys
import os

# Set a dummy API key
env = os.environ.copy()
env["HF_TOKEN"] = "dummy_token_for_testing"
env["API_BASE_URL"] = "http://localhost:9999"  # Non-existent endpoint

print("Testing inference.py structured output format...")
print("(Will fail at LLM call, but should print [START]/[STEP]/[END])\n")

result = subprocess.run(
    [sys.executable, "inference.py"],
    capture_output=True,
    text=True,
    timeout=10,
    env=env
)

stdout = result.stdout
stderr = result.stderr

print("=== STDOUT ===")
print(stdout)
print("\n=== STDERR ===")
print(stderr)

# Check for required structured output
has_start = "[START]" in stdout
has_step = "[STEP]" in stdout
has_end = "[END]" in stdout

print("\n=== VALIDATION ===")
print(f"✅ [START] found: {has_start}" if has_start else "❌ [START] missing")
print(f"✅ [STEP] found: {has_step}" if has_step else "❌ [STEP] missing")
print(f"✅ [END] found: {has_end}" if has_end else "❌ [END] missing")

if has_start and has_end:
    print("\n✅ PASS: Structured output format is correct!")
    sys.exit(0)
else:
    print("\n❌ FAIL: Missing required structured output blocks")
    sys.exit(1)
