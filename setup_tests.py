"""Helper to create tests directory."""
import os

base = r"c:\Users\rockb\OneDrive\Desktop\Projects\Meta Hack\research-integrity-gym"

# Create tests directory
tests_dir = os.path.join(base, "tests")
os.makedirs(tests_dir, exist_ok=True)
print(f"Created: {tests_dir}")

# Create __init__.py
init_file = os.path.join(tests_dir, "__init__.py")
with open(init_file, "w") as f:
    f.write('"""Test suite for Research Integrity Gym."""\n')
print(f"Created: {init_file}")

print("\nDone! Now run: python setup_tests.py")
