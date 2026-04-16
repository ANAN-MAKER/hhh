#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
验证第三轮清理的完整性和正确性。
"""

import sys
import numpy as np
import torch

def test_imports():
    """验证所有导入正常."""
    print("\n" + "="*70)
    print("TEST 1: Verify Imports")
    print("="*70)
    
    try:
        from agent_ppo.feature.preprocessor import Preprocessor
        from agent_ppo.feature.spatial_utils import evaluate_direction_quality
        from agent_ppo.feature.action_quality import ActionQualityEvaluator
        from agent_ppo.model.model import Model
        from agent_ppo.model.model_v2 import ModelV2
        from agent_ppo.conf.conf import Config
        
        print("✓ Preprocessor imported")
        print("✓ evaluate_direction_quality imported (shared function)")
        print("✓ ActionQualityEvaluator imported")
        print("✓ Model imported")
        print("✓ ModelV2 imported")
        print("✓ Config imported")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_shared_function():
    """验证共享的方向质量评估函数."""
    print("\n" + "="*70)
    print("TEST 2: Verify Shared Direction Quality Function")
    print("="*70)
    
    from agent_ppo.feature.spatial_utils import evaluate_direction_quality
    from agent_ppo.feature.preprocessor import MAP_SIZE
    
    # Test with empty monsters/treasures
    result = evaluate_direction_quality(
        action_id=0,
        hero_pos={'x': 64.0, 'z': 64.0},
        monsters=[],
        treasures=[],
        local_map=None,
        map_size=MAP_SIZE,
    )
    
    expected_keys = {'safety', 'treasure_reward', 'openness', 'obstacle_risk', 'direction_quality'}
    actual_keys = set(result.keys())
    
    if expected_keys == actual_keys:
        print(f"✓ Return keys match: {actual_keys}")
    else:
        print(f"✗ Key mismatch. Expected: {expected_keys}, Got: {actual_keys}")
        return False
    
    # Verify all values are in [0, 1]
    all_valid = True
    for key, value in result.items():
        if not (0.0 <= value <= 1.0):
            print(f"✗ {key} = {value} is outside [0, 1]")
            all_valid = False
    
    if all_valid:
        print("✓ All values in valid range [0, 1]")
        return True
    else:
        return False


def test_model_naming():
    """验证model.py中的变量名改名."""
    print("\n" + "="*70)
    print("TEST 3: Verify Model Naming Updates")
    print("="*70)
    
    from agent_ppo.model.model import Model
    
    model = Model()
    
    # Check encoder names
    encoder_names = [name for name, _ in model.named_modules()]
    
    expected_names = [
        'self_decision_aux_encoder',  # was self_progress_encoder
        'temporal_memory_encoder',     # was plan_encoder
    ]
    
    found_count = 0
    for expected_name in expected_names:
        if any(expected_name in name for name in encoder_names):
            print(f"✓ Found '{expected_name}'")
            found_count += 1
        else:
            print(f"✗ Missing '{expected_name}'")
    
    # Check that old names don't exist
    old_names = ['self_progress_encoder', 'plan_encoder']
    old_found = False
    for old_name in old_names:
        if any(old_name in name for name in encoder_names):
            print(f"✗ Old name '{old_name}' still exists")
            old_found = True
    
    if not old_found:
        print("✓ Old encoder names successfully removed")
    
    return found_count == len(expected_names) and not old_found


def test_preprocessor_imports():
    """验证preprocessor.py去除了无用的导入."""
    print("\n" + "="*70)
    print("TEST 4: Verify Preprocessor Cleanup")
    print("="*70)
    
    import agent_ppo.feature.preprocessor as preproc_module
    
    # Check that spatial_utils functions are available
    required_funcs = ['apply_action_to_pos', 'extract_safe_pos', 'evaluate_direction_quality']
    all_present = True
    
    for func_name in required_funcs:
        if hasattr(preproc_module, func_name):
            print(f"✓ {func_name} available")
        else:
            print(f"✗ {func_name} NOT available")
            all_present = False
    
    # Test that Preprocessor can still work
    try:
        p = preproc_module.Preprocessor()
        print("✓ Preprocessor instantiated successfully")
    except Exception as e:
        print(f"✗ Preprocessor instantiation failed: {e}")
        all_present = False
    
    return all_present


def test_model_forward():
    """验证模型前向传播还工作正常."""
    print("\n" + "="*70)
    print("TEST 5: Verify Model Forward Pass")
    print("="*70)
    
    from agent_ppo.model.model import Model
    from agent_ppo.conf.conf import Config
    
    model = Model()
    
    batch_size = 2
    obs = torch.randn(batch_size, Config.FEATURE_LEN)
    
    try:
        logits, value = model.forward(obs)
        
        expected_logits_shape = (batch_size, 16)
        expected_value_shape = (batch_size, 1)
        
        if logits.shape == expected_logits_shape:
            print(f"✓ Logits shape correct: {logits.shape}")
        else:
            print(f"✗ Logits shape wrong. Expected {expected_logits_shape}, got {logits.shape}")
            return False
        
        if value.shape == expected_value_shape:
            print(f"✓ Value shape correct: {value.shape}")
        else:
            print(f"✗ Value shape wrong. Expected {expected_value_shape}, got {value.shape}")
            return False
        
        print("✓ Forward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """验证config.py的正确性."""
    print("\n" + "="*70)
    print("TEST 6: Verify Config")
    print("="*70)
    
    from agent_ppo.conf.conf import Config
    
    # Check FEATURE_SPLIT_SHAPE
    expected_shape = [24, 7, 7, 12, 8, 20, 35, 30, 16, 96]
    if Config.FEATURE_SPLIT_SHAPE == expected_shape:
        print(f"✓ FEATURE_SPLIT_SHAPE correct: {Config.FEATURE_SPLIT_SHAPE}")
    else:
        print(f"✗ FEATURE_SPLIT_SHAPE mismatch. Expected {expected_shape}, got {Config.FEATURE_SPLIT_SHAPE}")
        return False
    
    # Check FEATURE_LEN
    expected_len = sum(expected_shape)
    if Config.FEATURE_LEN == expected_len:
        print(f"✓ FEATURE_LEN correct: {Config.FEATURE_LEN}")
    else:
        print(f"✗ FEATURE_LEN mismatch. Expected {expected_len}, got {Config.FEATURE_LEN}")
        return False
    
    # Verify the mapping explanation
    print("✓ Config structure:")
    print(f"  [A] Self features: 24D")
    print(f"  [B1-B4] Entities: 7+7+12+8=34D  ")
    print(f"  [E] Decision auxiliary: 20D")
    print(f"  [C] Map and paths: 35D")
    print(f"  [D] Temporal memory: 30D")
    print(f"  [Legal] Action mask: 16D")
    print(f"  [Quality] Action quality: 96D")
    print(f"  Total: {Config.FEATURE_LEN}D")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("FINAL CLEANUP VERIFICATION - test_cleanup_final.py")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Shared Function", test_shared_function),
        ("Model Naming", test_model_naming),
        ("Preprocessor Cleanup", test_preprocessor_imports),
        ("Model Forward", test_model_forward),
        ("Config", test_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Final cleanup is complete and correct.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
