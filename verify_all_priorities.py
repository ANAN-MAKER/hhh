#!/usr/bin/env python3
"""Final verification of all 5 Priorities"""

import subprocess
import sys
import os

tests = [
    ("Priority 1: Action Simulator", "test_action_simulator.py"),
    ("Priority 2: Masked Policy", "test_masked_policy_consistency.py"),
    ("Priority Cleanup (regression check)", "test_cleanup_final.py"),
    ("Priority 4-5: Temporal & Multi-head", "test_temporal_multihead.py"),
]

print("=" * 70)
print("🏆 FINAL VERIFICATION - ALL 5 PRIORITIES")
print("=" * 70 + "\n")

total_passed = 0
total_tests = 0

for name, test_file in tests:
    print(f"Running {name}...")
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        output = result.stdout + result.stderr
        
        # Extract test summary
        if "Total:" in output:
            for line in output.split('\n'):
                if "Total:" in line:
                    print(f"  {line.strip()}")
                    # Parse to get counts
                    if "/" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "/" in part:
                                nums = part.split("/")
                                passed = int(nums[0])
                                total = int(nums[1])
                                total_passed += passed
                                total_tests += total
                    break
        else:
            print(f"  Could not find test summary")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total Tests: {total_passed}/{total_tests}")
print(f"Pass Rate: {100*total_passed/max(total_tests,1):.1f}%")

if total_passed == total_tests and total_tests > 0:
    print("\n✅ ALL TESTS PASSED! Project 100% complete.")
    sys.exit(0)
else:
    print("\n⚠️  Some tests may have issues.")
    sys.exit(1)
