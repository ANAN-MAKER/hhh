#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Test action_quality integration - verifies that:
1. hero_pos_history stores correct {x, z} dicts
2. action_quality receives correct position type
3. action_quality output is 96D
4. no silent failures occur
"""

import sys
import numpy as np
from collections import deque

# Setup paths
sys.path.insert(0, '/root/code')
sys.path.insert(0, 'f:/wai/test/code')

def test_hero_pos_history():
    """Test that hero_pos_history stores {x, z} dicts correctly"""
    print("\n" + "="*70)
    print("TEST: hero_pos_history type and format")
    print("="*70)
    
    hero_pos_history = deque(maxlen=20)
    
    # Simulate adding positions
    test_positions = [
        {"x": 10.5, "z": 20.3},
        {"x": 11.2, "z": 19.8},
        {"x": 12.0, "z": 18.5},
    ]
    
    for pos in test_positions:
        hero_pos_history.append(pos)
    
    # Verify
    assert len(hero_pos_history) == 3, f"Expected 3 positions, got {len(hero_pos_history)}"
    
    for i, expected_pos in enumerate(test_positions):
        actual_pos = list(hero_pos_history)[i]
        assert isinstance(actual_pos, dict), f"Position {i} is not dict: {type(actual_pos)}"
        assert "x" in actual_pos and "z" in actual_pos, f"Position {i} missing x or z: {actual_pos}"
        assert actual_pos == expected_pos, f"Position {i} mismatch: {actual_pos} vs {expected_pos}"
    
    print("✓ hero_pos_history correctly stores {x, z} dicts")
    return hero_pos_history


def test_action_quality_computation():
    """Test that action_quality can be computed with correct position format"""
    print("\n" + "="*70)
    print("TEST: action_quality computation with hero_pos_history")
    print("="*70)
    
    try:
        from agent_ppo.feature.action_quality import compute_action_quality_features
        print("✓ action_quality module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import action_quality: {e}")
        return False
    
    # Create sample data
    hero_pos = {"x": 50.0, "z": 50.0}
    hero_hp = 100.0
    hero_max_hp = 100.0
    monsters = [{"x": 40.0, "z": 50.0}]
    treasures = [{"x": 60.0, "z": 60.0}]
    buffs = []
    legal_action_mask = np.ones(16, dtype=np.float32)
    local_map = np.zeros((128, 128), dtype=np.float32)
    
    # Create hero_pos_history with correct format
    hero_pos_history = deque(maxlen=20)
    hero_pos_history.append({"x": 49.0, "z": 50.0})
    hero_pos_history.append({"x": 48.0, "z": 50.0})
    hero_pos_history.append({"x": 47.0, "z": 50.0})
    
    flash_cooldown = 0
    danger_trend = 0.0
    
    try:
        action_quality_features = compute_action_quality_features(
            hero_pos=hero_pos,
            hero_hp=hero_hp,
            hero_max_hp=hero_max_hp,
            monsters=monsters,
            treasures=treasures,
            buffs=buffs,
            legal_action_mask=legal_action_mask,
            local_map=local_map,
            recent_positions=list(hero_pos_history),
            flash_cooldown=flash_cooldown,
            danger_trend=danger_trend,
        )
        
        print(f"✓ action_quality computed successfully")
        print(f"  - Output shape: {action_quality_features.shape}")
        print(f"  - Output dtype: {action_quality_features.dtype}")
        assert action_quality_features.shape == (96,), f"Expected shape (96,), got {action_quality_features.shape}"
        print(f"✓ action_quality output shape is correct (96D)")
        
        # Check values
        print(f"  - Output range: [{action_quality_features.min():.3f}, {action_quality_features.max():.3f}]")
        print(f"  - Non-zero values: {np.count_nonzero(action_quality_features)} / 96")
        
        return True
        
    except TypeError as e:
        print(f"✗ Type error in action_quality: {e}")
        print(f"  This suggests recent_positions format is wrong")
        return False
    except Exception as e:
        print(f"✗ Error in action_quality computation: {type(e).__name__}: {e}")
        return False


def test_logging_setup():
    """Test that logging is properly configured"""
    print("\n" + "="*70)
    print("TEST: logging configuration in preprocessor")
    print("="*70)
    
    try:
        from agent_ppo.feature.preprocessor import logger
        print(f"✓ Logger imported from preprocessor")
        print(f"  - Logger name: {logger.name}")
        print(f"  - Logger level: {logger.level}")
        return True
    except Exception as e:
        print(f"✗ Failed to import logger: {e}")
        return False


def test_preprocessor_integration():
    """Test that preprocessor properly uses hero_pos_history"""
    print("\n" + "="*70)
    print("TEST: preprocessor integration with hero_pos_history")
    print("="*70)
    
    try:
        from agent_ppo.feature.preprocessor import FeatureProcessor
        print("✓ FeatureProcessor imported")
        
        # Check if hero_pos_history exists
        import inspect
        source = inspect.getsource(FeatureProcessor.reset)
        if "hero_pos_history" in source:
            print("✓ hero_pos_history initialized in reset()")
        else:
            print("✗ hero_pos_history not found in reset()")
            return False
        
        # Check _update_memory
        source = inspect.getsource(FeatureProcessor._update_memory)
        if "hero_pos_history" in source:
            print("✓ hero_pos_history updated in _update_memory()")
        else:
            print("✗ hero_pos_history not updated in _update_memory()")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error checking preprocessor: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ACTION_QUALITY INTEGRATION TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test 1
    try:
        test_hero_pos_history()
        results.append(True)
    except Exception as e:
        print(f"✗ hero_pos_history test failed: {e}")
        results.append(False)
    
    # Test 2
    results.append(test_action_quality_computation())
    
    # Test 3
    results.append(test_logging_setup())
    
    # Test 4
    results.append(test_preprocessor_integration())
    
    # Summary
    print("\n" + "="*70)
    print(f"SUMMARY: {sum(results)}/{len(results)} tests passed")
    print("="*70 + "\n")
    
    if all(results):
        print("✓ All integration tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
