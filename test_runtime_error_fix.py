#!/usr/bin/env python3
"""
Minimal test to verify the runtime fix: get_direction_ahead_in_local_map can handle list type local_map.

This directly reproduces the error scenario from the log:
  Error: AttributeError: 'list' object has no attribute 'size'
  Location: spatial_utils.py, line 377, in get_direction_ahead_in_local_map
"""

import sys
import numpy as np

sys.path.insert(0, '/workspace/code')

from agent_ppo.feature.spatial_utils import get_direction_ahead_in_local_map, evaluate_direction_quality

def main():
    print("=" * 70)
    print("RUNTIME ERROR FIX VERIFICATION")
    print("Testing: AttributeError: 'list' object has no attribute 'size'")
    print("=" * 70)
    
    # Create a local_map as list (as passed by the runtime system)
    local_map_list = [[1 if np.random.random() > 0.2 else 0 for _ in range(21)] for _ in range(21)]
    
    print(f"\nTest 1: Direct get_direction_ahead_in_local_map call")
    print(f"  Input type: {type(local_map_list)} (list)")
    print(f"  Input shape: {len(local_map_list)}x{len(local_map_list[0])}")
    
    try:
        # This is the exact call from the error traceback: line 747 in evaluate_direction_quality
        result = get_direction_ahead_in_local_map(0, local_map_list, depth=3)
        print(f"  ✓ SUCCESS: {result}")
    except AttributeError as e:
        print(f"  ✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"  ✗ FAILED (other error): {type(e).__name__}: {e}")
        return False
    
    print(f"\nTest 2: Full evaluate_direction_quality call")
    print(f"  Input type: {type(local_map_list)}")
    
    try:
        # Create dummy data for evaluate_direction_quality
        hero_pos = {"x": 64.0, "z": 64.0}
        monsters = [{"x": 50.0, "z": 50.0}]
        treasures = [{"x": 70.0, "z": 70.0}]
        
        # This is the exact call from the error traceback: line 747
        result = evaluate_direction_quality(0, hero_pos, monsters, treasures, local_map_list)
        print(f"  ✓ SUCCESS: {result}")
    except AttributeError as e:
        print(f"  ✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"  ✗ FAILED (other error): {type(e).__name__}: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ RUNTIME ERROR HAS BEEN FIXED!")
    print("The code can now handle list-type local_map without AttributeError.")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
