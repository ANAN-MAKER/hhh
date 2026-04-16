#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
测试Agent类创建 - 验证修复后Agent能正常初始化
"""

import sys
import torch

def test_agent_creation():
    try:
        print("=" * 70)
        print("AGENT CREATION TEST")
        print("=" * 70)
        
        from agent_ppo.agent import Agent
        from agent_ppo.conf.conf import Config
        
        print("\n[Testing] Creating Agent instance...")
        agent = Agent(agent_type="player", device=torch.device('cpu'))
        print(f"  [OK] Agent created successfully")
        print(f"  - Model type: {type(agent.model).__name__}")
        print(f"  - Device: {agent.device}")
        print(f"  - Preprocessor: {type(agent.preprocessor).__name__}")
        
        print("\n[Testing] Agent properties...")
        print(f"  - Config.FEATURE_LEN: {Config.FEATURE_LEN}")
        print(f"  - Config.ACTION_NUM: {Config.ACTION_NUM}")
        print(f"  - Config.FEATURE_SPLIT_SHAPE sum: {sum(Config.FEATURE_SPLIT_SHAPE)}")
        
        print("\n" + "=" * 70)
        print("RESULT: [OK] Agent creation successful!")
        return 0
        
    except Exception as e:
        print(f"\n[FAILED] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("RESULT: [FAILED] Agent creation failed")
        return 1

if __name__ == "__main__":
    sys.exit(test_agent_creation())
