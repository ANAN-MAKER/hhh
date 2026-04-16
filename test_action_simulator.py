#!/usr/bin/env python3
"""Test for Priority 1: Action Simulator"""

import sys
import numpy as np

sys.path.insert(0, '/workspace/code')

from agent_ppo.feature.action_simulator import (
    simulate_action,
    simulate_move,
    simulate_flash,
    is_diag_move_legal,
    ACTION_DELTA,
)

def test_all_actions_simulate():
    """Test that all 16 actions can be simulated."""
    print("=" * 70)
    print("TEST 1: All 16 actions can be simulated")
    print("=" * 70)
    
    hero_pos = {"x": 64.0, "z": 64.0}
    local_map = [[1] * 21 for _ in range(21)]  # All passable
    
    success_count = 0
    for action_id in range(16):
        try:
            result = simulate_action(hero_pos, action_id, local_map)
            is_flash = "flash_pos" in result
            print(f"  Action {action_id:2d}: {'Flash' if is_flash else 'Move ':5s} -> "
                  f"dist={result.get('movement_distance', result.get('flash_distance', 0.0)):5.1f}, "
                  f"blocked={result.get('blocked', False)}")
            success_count += 1
        except Exception as e:
            print(f"  Action {action_id:2d}: ERROR - {e}")
    
    print(f"\nResult: {success_count}/16 actions simulated")
    return success_count == 16

def test_diagonal_movement():
    """Test diagonal movement constraint (wall-cutting prevention)."""
    print("\n" + "=" * 70)
    print("TEST 2: Diagonal movement respects wall-cutting rules")
    print("=" * 70)
    
    hero_pos = {"x": 64.0, "z": 64.0}
    
    # Create a map with walls that should block diagonal move
    # For diagonal move from (64,64) to (65,65):
    # - Need (11,11) passable
    # - Need EITHER (11,10) OR (10,11) passable
    # We block both to force failure
    local_map = [[1] * 21 for _ in range(21)]
    local_map[11][10] = 0  # Block one adjacent cell
    local_map[10][11] = 0  # Block other adjacent cell
    
    # Try diagonal movement to (65, 65) which should be blocked
    result = simulate_move(hero_pos, 7, local_map, buff_active=False)  # action 7 = southeast diagonal
    
    blocked = result.get("blocked", True)
    print(f"  Diagonal move to (65, 65) with both adjacent walls: blocked={blocked}")
    print(f"  Next pos: {result.get('next_pos')}")
    
    return blocked  # Should be blocked

def test_flash_distance():
    """Test flash distances: linear=10, diagonal=8."""
    print("\n" + "=" * 70)
    print("TEST 3: Flash distances are correct")
    print("=" * 70)
    
    hero_pos = {"x": 64.0, "z": 64.0}
    local_map = [[1] * 21 for _ in range(21)]  # All passable
    
    # Test linear flash (action 8 = right)
    result_linear = simulate_flash(hero_pos, 8, local_map)
    linear_dist = result_linear.get("flash_distance", 0.0)
    print(f"  Linear flash (right): distance={linear_dist:.1f} (expect ~10.0)")
    
    # Test diagonal flash (action 9 = right-up)  
    result_diag = simulate_flash(hero_pos, 9, local_map)
    diag_dist = result_diag.get("flash_distance", 0.0)
    print(f"  Diagonal flash (right-up): distance={diag_dist:.1f} (expect ~8.0)")
    
    linear_ok = 9.5 < linear_dist <= 10.5
    diag_ok = 7.5 < diag_dist <= 8.5
    
    print(f"\n  Validation: linear_ok={linear_ok}, diag_ok={diag_ok}")
    return linear_ok and diag_ok

def test_action_results_structure():
    """Test that simulate_action returns complete result structure."""
    print("\n" + "=" * 70)
    print("TEST 4: Action results have all required fields")
    print("=" * 70)
    
    hero_pos = {"x": 64.0, "z": 64.0}
    local_map = [[1] * 21 for _ in range(21)]
    treasures = [{"x": 65.0, "z": 64.0}]  # One treasure to the right
    
    # Test normal move
    result_move = simulate_action(hero_pos, 0, local_map, treasures=treasures)
    
    required_fields_move = [
        "next_pos", "movement_distance", "path_cells", "blocked", 
        "stay_same_cell", "collected_treasures", "collected_buffs", "is_flash"
    ]
    
    move_ok = all(field in result_move for field in required_fields_move)
    
    print(f"  Move result fields: {move_ok}")
    for field in required_fields_move:
        if field in result_move:
            print(f"    ✓ {field}: {result_move[field]}")
        else:
            print(f"    ✗ {field}: MISSING")
    
    # Test flash
    result_flash = simulate_action(hero_pos, 8, local_map)
    flash_ok = "flash_pos" in result_flash and result_flash.get("is_flash", False)
    
    print(f"\n  Flash result is flash: {flash_ok}")
    
    return move_ok and flash_ok

def test_no_wall_cutting_for_speed_2():
    """Test that buff speed=2 movement doesn't cut walls."""
    print("\n" + "=" * 70)
    print("TEST 5: Buff speed=2 respects diagonal cut constraints")
    print("=" * 70)
    
    hero_pos = {"x": 64.0, "z": 64.0}
    
    # Create map with corner blocked
    local_map = [[1] * 21 for _ in range(21)]
    local_map[11][11] = 0  # Diagonal target blocked
    local_map[10][11] = 1  # One adjacent passable
    local_map[11][10] = 0  # Other adjacent blocked
    
    # Try diagonal move with buff (speed=2)
    result = simulate_move(hero_pos, 7, local_map, buff_active=True)  # action 7 = southeast
    
    blocked = result.get("blocked", True)
    print(f"  Diagonal move to (66, 66) with buff (speed=2): blocked={blocked}")
    print(f"  Movement distance: {result.get('movement_distance', 0.0):.1f}")
    
    return blocked

def main():
    print("\n" + "=" * 70)
    print("PRIORITY 1: ACTION SIMULATOR VERIFICATION")
    print("=" * 70 + "\n")
    
    tests = [
        ("All 16 Actions", test_all_actions_simulate),
        ("Diagonal Movement", test_diagonal_movement),
        ("Flash Distance", test_flash_distance),
        ("Result Structure", test_action_results_structure),
        ("Speed=2 Wall Cut", test_no_wall_cutting_for_speed_2),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = passed
        except Exception as e:
            print(f"\n✗ {name} EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total_pass = sum(1 for p in results.values() if p)
    print(f"\nTotal: {total_pass}/{len(results)} tests passed")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
