#!/usr/bin/env python3
"""Test for Priority 2: Masked Policy Consistency"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/workspace/code')

from agent_ppo.feature.mask_utils import softmax_numpy, softmax_torch, verify_consistency


def test_numpy_torch_consistency():
    """Test that numpy and torch versions produce identical results."""
    print("=" * 70)
    print("TEST 1: Numpy vs Torch consistency")
    print("=" * 70)
    
    # Test case 1: Simple case
    logits_np = np.array([0.5, 1.0, 0.2, -0.5], dtype=np.float32)
    legal_np = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)  # action 2 is illegal
    
    # Numpy version
    prob_np = softmax_numpy(logits_np, legal_np)
    
    # Torch version (batch size 1)
    logits_torch = torch.from_numpy(logits_np).unsqueeze(0).float()
    legal_torch = torch.from_numpy(legal_np).unsqueeze(0).float()
    prob_torch = softmax_torch(logits_torch, legal_torch)[0].cpu().numpy()
    
    # Compare
    max_diff = np.max(np.abs(prob_np - prob_torch))
    is_close = np.allclose(prob_np, prob_torch, atol=1e-6)
    
    print(f"  Logits: {logits_np}")
    print(f"  Legal:  {legal_np}")
    print(f"  Numpy prob:  {prob_np}")
    print(f"  Torch prob:  {prob_torch}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Result: {'PASS' if is_close else 'FAIL'}")
    
    return is_close


def test_illegal_actions_zero_prob():
    """Test that illegal actions get exactly zero probability."""
    print("\n" + "=" * 70)
    print("TEST 2: Illegal actions have zero probability")
    print("=" * 70)
    
    # All actions illegal except action 0
    logits = np.random.randn(16).astype(np.float32) * 10  # Large noise
    legal = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    
    prob = softmax_numpy(logits, legal)
    
    # Check: only action 0 should be non-zero
    expected_prob_at_0_nonzero = prob[0] > 0.99  # Should be ≈1
    all_others_zero = np.allclose(prob[1:], 0, atol=1e-7)
    
    print(f"  Probability of action 0: {prob[0]:.6f}")
    print(f"  Probabilities of other actions: {prob[1:]}")
    print(f"  Action 0 near 1.0: {expected_prob_at_0_nonzero}")
    print(f"  Other actions near 0: {all_others_zero}")
    print(f"  Result: {'PASS' if (expected_prob_at_0_nonzero and all_others_zero) else 'FAIL'}")
    
    return expected_prob_at_0_nonzero and all_others_zero


def test_probability_sum_is_one():
    """Test that probabilities sum to 1."""
    print("\n" + "=" * 70)
    print("TEST 3: Probabilities sum to 1")
    print("=" * 70)
    
    test_cases = [
        # (logits, legal, description)
        (np.array([0.0] * 16, dtype=np.float32), np.ones(16, dtype=np.float32), "all legal, uniform logits"),
        (np.random.randn(16).astype(np.float32), np.ones(16, dtype=np.float32), "all legal, random logits"),
        (np.random.randn(16).astype(np.float32), np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0], dtype=np.float32), "half legal"),
    ]
    
    all_pass = True
    for logits, legal, desc in test_cases:
        prob_np = softmax_numpy(logits, legal)
        prob_sum = np.sum(prob_np)
        is_sum_one = np.isclose(prob_sum, 1.0, atol=1e-6)
        
        print(f"  {desc}: sum={prob_sum:.8f}, pass={is_sum_one}")
        all_pass = all_pass and is_sum_one
    
    print(f"  Result: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_batch_consistency():
    """Test that batch processing is consistent with single samples."""
    print("\n" + "=" * 70)
    print("TEST 4: Batch processing consistency")
    print("=" * 70)
    
    batch_size = 4
    action_dim = 16
    
    # Create random batch
    logits_batch = np.random.randn(batch_size, action_dim).astype(np.float32)
    legal_batch = np.random.randint(0, 2, (batch_size, action_dim)).astype(np.float32)
    # Ensure each sample has at least one legal action
    for i in range(batch_size):
        if np.sum(legal_batch[i]) == 0:
            legal_batch[i, np.random.randint(0, action_dim)] = 1.0
    
    # Torch batch version
    logits_torch = torch.from_numpy(logits_batch).float()
    legal_torch = torch.from_numpy(legal_batch).float()
    prob_batch = softmax_torch(logits_torch, legal_torch).cpu().numpy()
    
    # Compare with single sample numpy versions
    all_consistent = True
    for i in range(batch_size):
        prob_single = softmax_numpy(logits_batch[i], legal_batch[i])
        is_consistent = np.allclose(prob_batch[i], prob_single, atol=1e-6)
        all_consistent = all_consistent and is_consistent
        
        if not is_consistent:
            print(f"  Sample {i}: INCONSISTENT")
            print(f"    Batch result: {prob_batch[i]}")
            print(f"    Single result: {prob_single}")
    
    print(f"  All batch samples consistent with single processing: {all_consistent}")
    print(f"  Result: {'PASS' if all_consistent else 'FAIL'}")
    return all_consistent


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("\n" + "=" * 70)
    print("TEST 5: Numerical stability with extreme logits")
    print("=" * 70)
    
    test_cases = [
        # (logits, legal, description)
        (np.array([100, 101, 102, 103, -100, -101, -102, -103, 0, 1, 2, 3, 50, 51, 52, 53], dtype=np.float32),
         np.ones(16, dtype=np.float32),
         "extreme range logits"),
        
        (np.full(16, 1e6, dtype=np.float32),
         np.ones(16, dtype=np.float32),
         "very large uniform logits"),
         
        (np.full(16, -1e6, dtype=np.float32),
         np.ones(16, dtype=np.float32),
         "very small uniform logits"),
    ]
    
    all_stable = True
    for logits, legal, desc in test_cases:
        try:
            prob_np = softmax_numpy(logits, legal)
            logits_torch = torch.from_numpy(logits).unsqueeze(0).float()
            legal_torch = torch.from_numpy(legal).unsqueeze(0).float()
            prob_torch = softmax_torch(logits_torch, legal_torch)[0].cpu().numpy()
            
            has_nan_np = np.any(np.isnan(prob_np))
            has_nan_torch = np.any(np.isnan(prob_torch))
            is_consistent = np.allclose(prob_np, prob_torch, atol=1e-5, equal_nan=True)
            
            stability_ok = not (has_nan_np or has_nan_torch) and is_consistent
            print(f"  {desc}: stable={stability_ok}, nan_np={has_nan_np}, nan_torch={has_nan_torch}")
            all_stable = all_stable and stability_ok
        except Exception as e:
            print(f"  {desc}: EXCEPTION - {e}")
            all_stable = False
    
    print(f"  Result: {'PASS' if all_stable else 'FAIL'}")
    return all_stable


def test_verify_consistency_function():
    """Test the verify_consistency helper function."""
    print("\n" + "=" * 70)
    print("TEST 6: verify_consistency() helper works correctly")
    print("=" * 70)
    
    logits = np.random.randn(16).astype(np.float32)
    legal = np.random.randint(0, 2, 16).astype(np.float32)
    legal[0] = 1.0  # Ensure at least one legal action
    
    is_consistent = verify_consistency(logits, legal)
    
    print(f"  Random test case: consistent={is_consistent}")
    print(f"  Result: {'PASS' if is_consistent else 'FAIL'}")
    
    return is_consistent


def main():
    print("\n" + "=" * 70)
    print("PRIORITY 2: MASKED POLICY UNIFICATION VERIFICATION")
    print("=" * 70 + "\n")
    
    results = {
        "numpy_torch_consistency": test_numpy_torch_consistency(),
        "illegal_actions_zero": test_illegal_actions_zero_prob(),
        "prob_sum_one": test_probability_sum_is_one(),
        "batch_consistency": test_batch_consistency(),
        "numerical_stability": test_numerical_stability(),
        "verify_consistency": test_verify_consistency_function(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All Priority 2 tests passed! Masked policy is unified and consistent.")
    else:
        print("\n✗ Some tests failed. Please review the failures above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
