#!/usr/bin/env python3
"""Integration test for preprocessor.feature_process with real-world data."""

import sys
import numpy as np

sys.path.insert(0, '/workspace/code')

from agent_ppo.feature.preprocessor import Preprocessor

def create_mock_observation():
    """Create a mock observation to pass to feature_process."""
    # Note: feature_process expects env_obs["observation"] to exist
    return {
        "observation": {
            'step_no': 0,
            'frame_state': {
                'heroes': {
                    'buff_remaining_time': 0,
                    'flash_cooldown': 0,
                    'hp': 100.0,
                    'hero_hp': 100.0,
                    'max_hp': 100.0,
                    'speed': 0.0,
                    'status': 1,
                    'x': 64.0,
                    'z': 64.0,
                },
                'monsters': [
                    {
                        'speed': 2.0,
                        'hero_l2_distance': 30.0,
                        'x': 50.0,
                        'z': 50.0,
                    },
                    {
                        'speed': 1.5,
                        'hero_l2_distance': 40.0,
                        'x': 80.0,
                        'z': 80.0,
                    }
                ],
                'organs': [
                    # Treasures (sub_type=1)
                    {'sub_type': 1, 'status': 1, 'x': 70.0, 'z': 70.0},
                    {'sub_type': 1, 'status': 1, 'x': 60.0, 'z': 60.0},
                    # Buffs (sub_type=2)
                    {'sub_type': 2, 'status': 1, 'x': 75.0, 'z': 75.0},
                ],
            },
            'legal_action': [1] * 16,
            'env_info': {
                'treasures_collected': 0,
                'total_treasure': 5,
                'collected_buff': 0,
                'total_buff': 2,
                'max_step': 200,
            },
            # Local map: 21x21 grid as LIST (this is what triggers the bug)
            'map_info': [[1 if np.random.random() > 0.2 else 0 for _ in range(21)] for _ in range(21)],
        }
    }

def test_preprocessor_with_list_local_map():
    """Test that preprocessor.feature_process works with list-type local_map."""
    print("=" * 70)
    print("INTEGRATION TEST: Preprocessor.feature_process with list local_map")
    print("=" * 70)
    
    preprocessor = Preprocessor()
    
    # Create a mock observation
    obs = create_mock_observation()
    
    print(f"\nObservation structure:")
    print(f"  - step_no: {obs['observation']['step_no']}")
    print(f"  - heroes hp: {obs['observation']['frame_state']['heroes']['hp']}")
    print(f"  - monsters count: {len(obs['observation']['frame_state']['monsters'])}")
    print(f"  - map_info type: {type(obs['observation']['map_info'])}")
    print(f"  - map_info shape: {len(obs['observation']['map_info'])}x{len(obs['observation']['map_info'][0])}")
    
    try:
        # This is where the error was happening
        feature, legal_action, reward, reward_info = preprocessor.feature_process(
            obs,
            -1
        )
        
        print(f"\n✓ feature_process succeeded!")
        print(f"  - feature shape: {feature.shape}")
        print(f"  - feature dtype: {feature.dtype}")
        print(f"  - legal_action: {legal_action[:8]}... (showing first 8)")
        print(f"  - reward value: {reward}")
        print(f"  - reward_info keys: {list(reward_info.keys())}")
        
        # Validate feature dimensions
        expected_feature_len = 255  # As per FEATURE_SPLIT_SHAPE
        if feature.shape[0] != expected_feature_len:
            print(f"✗ Feature dimension mismatch! Expected {expected_feature_len}, got {feature.shape[0]}")
            return False
        
        # Validate legal_action
        if len(legal_action) != 16:
            print(f"✗ Legal action length mismatch! Expected 16, got {len(legal_action)}")
            return False
        
        # Validate reward
        if not isinstance(reward, (int, float)):
            print(f"✗ Reward type error! Got {type(reward)}")
            return False
        
        print(f"\n✓ All validations passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ feature_process failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_frames():
    """Test multiple consecutive frames to ensure state is properly maintained."""
    print("\n" + "=" * 70)
    print("SEQUENTIAL TEST: Multiple frames")
    print("=" * 70)
    
    preprocessor = Preprocessor()
    
    for frame_no in range(3):
        obs = create_mock_observation()
        obs['observation']['map_info'] = [[1 if np.random.random() > 0.2 else 0 for _ in range(21)] for _ in range(21)]
        
        try:
            feature, legal_action, reward, reward_info = preprocessor.feature_process(
                obs,
                -1
            )
            print(f"✓ Frame {frame_no}: feature_process succeeded (reward={reward:.4f})")
        except Exception as e:
            print(f"✗ Frame {frame_no}: feature_process failed - {e}")
            return False
    
    print(f"\n✓ All frames processed successfully!")
    return True

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNTIME ERROR FIX VERIFICATION")
    print("Testing fix for: AttributeError: 'list' object has no attribute 'size'")
    print("=" * 70)
    
    all_pass = True
    
    try:
        all_pass &= test_preprocessor_with_list_local_map()
        all_pass &= test_multiple_frames()
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False
    
    print("\n" + "=" * 70)
    if all_pass:
        print("🎉 INTEGRATION TEST PASSED - Fix verified!")
        print("The runtime error has been resolved.")
        sys.exit(0)
    else:
        print("❌ INTEGRATION TEST FAILED!")
        sys.exit(1)
