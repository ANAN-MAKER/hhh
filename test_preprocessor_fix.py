#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Preprocessor feature extraction test - verify the fix
"""

import sys
import numpy as np
from collections import deque

def test_preprocessor():
    try:
        print("=" * 70)
        print("PREPROCESSOR FEATURE EXTRACTION TEST")
        print("=" * 70)
        
        from agent_ppo.feature.preprocessor import Preprocessor
        from agent_ppo.conf.conf import Config
        
        print("\n[Testing] Creating Preprocessor...")
        preprocessor = Preprocessor()
        preprocessor.reset()
        print(f"  [OK] Preprocessor created")
        
        # Create mock environment observation
        print("\n[Testing] Creating mock environment observation...")
        env_obs = {
            'env_id': 'test_env',
            'frame_no': 0,
            'observation': {
                'step_no': 0,
                'frame_state': {
                    'heroes': {
                        'x': 64.0,
                        'z': 64.0,
                        'hp': 100.0,
                        'flash_cooldown': 0,
                        'buff_remaining_time': 0.0,
                    },
                    'monsters': [
                        {
                            'x': 32.0,
                            'z': 32.0,
                            'hp': 50.0,
                        }
                    ],
                    'treasures': [
                        {
                            'x': 96.0,
                            'z': 96.0,
                        }
                    ],
                    'buffs': [],
                },
                'map_info': np.zeros((128, 128), dtype=np.float32),
                'env_info': {
                    'max_step': 200,
                    'total_treasure': 5,
                    'treasures_collected': 0,
                    'total_buff': 2,
                    'collected_buff': 0,
                    'flash_count': 0,
                    'total_score': 0.0,
                },
            }
        }
        
        print("  [OK] Mock observation created")
        
        # Test feature_process
        print("\n[Testing] Running feature_process...")
        try:
            feature, legal_action, reward, reward_info = preprocessor.feature_process(env_obs, last_action=-1)
            
            print(f"  [OK] feature_process completed successfully")
            print(f"    - Feature shape: {feature.shape}")
            print(f"    - Feature dtype: {feature.dtype}")
            print(f"    - Expected shape: ({Config.FEATURE_LEN},)")
            print(f"    - Expected: 255D")
            
            # Verify dimensions
            if feature.shape[0] == Config.FEATURE_LEN:
                print(f"  [OK] Feature dimension correct: {feature.shape[0]}D")
            else:
                print(f"  [ERROR] Feature dimension mismatch: {feature.shape[0]} != {Config.FEATURE_LEN}")
                return 1
            
            print(f"    - Legal action: {len(legal_action)} actions")
            print(f"    - Reward: {reward}")
            print(f"    - Reward keys: {list(reward_info.keys())[:5]}...")
            
            return 0
            
        except NameError as e:
            print(f"  [ERROR] NameError: {e}")
            import traceback
            traceback.print_exc()
            return 1
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
    except Exception as e:
        print(f"\n[CRITICAL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    result = test_preprocessor()
    print("\n" + "=" * 70)
    if result == 0:
        print("RESULT: [OK] Preprocessor test passed!")
    else:
        print("RESULT: [FAILED] Preprocessor test failed!")
    print("=" * 70)
    sys.exit(result)
