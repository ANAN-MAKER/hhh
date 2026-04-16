#!/usr/bin/env python3
"""Quick test for list/numpy array compatibility in get_direction_ahead_in_local_map."""

import numpy as np
import sys

# Add workspace to path
sys.path.insert(0, '/workspace/code')

from agent_ppo.feature.spatial_utils import get_direction_ahead_in_local_map

def test_local_map_as_list():
    """Test that get_direction_ahead_in_local_map works with Python list."""
    print("=" * 60)
    print("TEST: local_map as Python list")
    print("=" * 60)
    
    # Create a simple 21x21 local map as list (what the runtime is actually passing)
    local_map_list = [[1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    local_map_list = local_map_list * 21  # Simple way to make 21 rows
    
    print(f"local_map type: {type(local_map_list)}")
    print(f"local_map shape: {len(local_map_list)}x{len(local_map_list[0])}")
    
    # Test different action IDs
    for action_id in [0, 1, 8, 15]:
        try:
            result = get_direction_ahead_in_local_map(action_id, local_map_list, depth=3)
            print(f"✓ action_id={action_id}: {result}")
        except Exception as e:
            print(f"✗ action_id={action_id}: ERROR - {e}")
            return False
    
    return True

def test_local_map_as_numpy():
    """Test that get_direction_ahead_in_local_map still works with numpy array."""
    print("\n" + "=" * 60)
    print("TEST: local_map as numpy array")
    print("=" * 60)
    
    # Create a 21x21 local map as numpy array
    local_map_array = np.ones((21, 21), dtype=np.float32)
    local_map_array[10, 10] = 0  # Add a blocked cell
    
    print(f"local_map type: {type(local_map_array)}")
    print(f"local_map shape: {local_map_array.shape}")
    
    # Test different action IDs
    for action_id in [0, 1, 8, 15]:
        try:
            result = get_direction_ahead_in_local_map(action_id, local_map_array, depth=3)
            print(f"✓ action_id={action_id}: {result}")
        except Exception as e:
            print(f"✗ action_id={action_id}: ERROR - {e}")
            return False
    
    return True

def test_local_map_none():
    """Test that get_direction_ahead_in_local_map works with None."""
    print("\n" + "=" * 60)
    print("TEST: local_map as None")
    print("=" * 60)
    
    # Test with None (should use default values)
    for action_id in [0, 1, 8, 15]:
        try:
            result = get_direction_ahead_in_local_map(action_id, None, depth=3)
            print(f"✓ action_id={action_id}: {result}")
        except Exception as e:
            print(f"✗ action_id={action_id}: ERROR - {e}")
            return False
    
    return True

if __name__ == "__main__":
    all_pass = True
    
    all_pass &= test_local_map_as_list()
    all_pass &= test_local_map_as_numpy()
    all_pass &= test_local_map_none()
    
    print("\n" + "=" * 60)
    if all_pass:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)
