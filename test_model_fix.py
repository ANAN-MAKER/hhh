#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
测试模型修复 - 验证两个模型类都能处理255D输入
"""

import sys
import torch
import numpy as np

def test_models():
    try:
        from agent_ppo.model.model import Model as OldModel
        from agent_ppo.model.model_v2 import ModelV2
        from agent_ppo.conf.conf import Config
        
        print("=" * 70)
        print("MODEL DIMENSION FIX VERIFICATION")
        print("=" * 70)
        
        print(f"\nConfig.FEATURE_SPLIT_SHAPE: {Config.FEATURE_SPLIT_SHAPE}")
        print(f"Config.FEATURE_LEN: {Config.FEATURE_LEN}")
        print(f"Number of dimensions: {len(Config.FEATURE_SPLIT_SHAPE)}")
        
        # Test old Model class
        print("\n[Testing] Old Model class (model.py)...")
        try:
            model_v1 = OldModel(device='cpu')
            test_input_v1 = np.random.randn(2, Config.FEATURE_LEN).astype(np.float32)
            test_tensor = torch.from_numpy(test_input_v1)
            logits, value = model_v1(test_tensor)
            print(f"  [OK] Input shape: {test_tensor.shape}")
            print(f"  [OK] Logits shape: {logits.shape}")
            print(f"  [OK] Value shape: {value.shape}")
            model_v1_ok = True
        except Exception as e:
            print(f"  [FAILED] {type(e).__name__}: {e}")
            model_v1_ok = False
        
        # Test ModelV2
        print("\n[Testing] ModelV2 class (model_v2.py)...")
        try:
            model_v2 = ModelV2(device='cpu')
            test_input_v2 = np.random.randn(2, Config.FEATURE_LEN).astype(np.float32)
            test_tensor_v2 = torch.from_numpy(test_input_v2)
            logits2, value2 = model_v2(test_tensor_v2)
            print(f"  [OK] Input shape: {test_tensor_v2.shape}")
            print(f"  [OK] Logits shape: {logits2.shape}")
            print(f"  [OK] Value shape: {value2.shape}")
            model_v2_ok = True
        except Exception as e:
            print(f"  [FAILED] {type(e).__name__}: {e}")
            model_v2_ok = False
        
        print("\n" + "=" * 70)
        if model_v1_ok and model_v2_ok:
            print("RESULT: [OK] Both models handle 255D input correctly!")
            return 0
        else:
            print("RESULT: [FAILED] One or more models failed")
            if not model_v1_ok:
                print("  - Old Model (model.py) failed")
            if not model_v2_ok:
                print("  - ModelV2 (model_v2.py) failed")
            return 1
            
    except Exception as e:
        print(f"Critical error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_models())
