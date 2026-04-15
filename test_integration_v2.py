#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
完整的集成测试：验证 Preprocessor 与新奖励系统的完整集成
Integration test: Verify complete pipeline of Preprocessor + new reward system
"""

import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置 stdout 编码为 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.conf.conf import Config
import numpy as np


def test_preprocessor_integration():
    """测试 Preprocessor 与新奖励系统的集成"""
    print("=" * 70)
    print("集成测试：Preprocessor + 新三层奖励系统")
    print("=" * 70)
    
    # 初始化 Preprocessor
    p = Preprocessor()
    print("[PASS] Preprocessor 初始化成功")
    print(f"  - FEATURES 配置: {Config.FEATURES}")
    print(f"  - DIM_OF_OBSERVATION: {Config.DIM_OF_OBSERVATION}")
    
    # 模拟一个简单的游戏状态
    print("\n" + "-" * 70)
    print("模拟游戏状态 1: 收集宝藏，怪物靠近")
    print("-" * 70)
    
    hero_pos = {"x": 64.0, "z": 64.0}
    
    env_info = {
        "treasures": [
            {"name": "treasure1", "pos": {"x": 70.0, "z": 70.0}},
            {"name": "treasure2", "pos": {"x": 60.0, "z": 60.0}},
            {"name": "treasure3", "pos": {"x": 80.0, "z": 80.0}},
        ],
        "monsters": [
            {"name": "monster1", "pos": {"x": 50.0, "z": 50.0}, "hp": 10},
            {"name": "monster2", "pos": {"x": 70.0, "z": 75.0}, "hp": 8},
        ],
        "buffs": [
            {"name": "shield", "pos": {"x": 55.0, "z": 55.0}},
        ],
        "skills": {
            "flash_count": 2,
            "flash_cd": 0,
            "speed_buff_remaining": 0.0,
        },
        "hero": {
            "pos": hero_pos,
            "speed": 1.0,
            "hp": 1.0,
        },
        "treasures_collected": 2,
        "total_treasure": 15,
        "collected_buff": 0,
        "total_buff": 3,
        "buff_remaining_time": 0.0,
        "flash_count": 2,
        "total_score": 40.0,
        "is_dead": False,
    }
    
    legal_action = np.ones(16, dtype=np.int32)
    last_action = 0
    
    # 调用 feature_process
    try:
        feature, legal_action_out, reward_list, reward_info = p.feature_process(
            hero_pos=hero_pos,
            env_info=env_info,
            action_mask=legal_action,
            last_action=last_action,
        )
        
        print(f"[PASS] feature_process 执行成功")
        print(f"  - 特征维度: {len(feature)}")
        print(f"  - 奖励: {reward_list[0]:.4f}")
        print(f"\n奖励详情:")
        print(f"  Layer A (主目标): {reward_info.get('layer_a', 0):.4f}")
        print(f"  Layer B (引导):   {reward_info.get('layer_b', 0):.4f}")
        print(f"  Layer C (约束):   {reward_info.get('layer_c', 0):.4f}")
        
        if 'survival_reward' in reward_info:
            print(f"\n新系统的详细拆分:")
            print(f"  - Survival: {reward_info['survival_reward']:.4f}")
            print(f"  - Score: {reward_info['score_objective']:.4f}")
            print(f"  - Distance: {reward_info['distance_shaping']:.4f}")
            print(f"  - Danger: {reward_info['danger_zone_penalty']:.4f}")
            print(f"  - Buff: {reward_info['buff_bonus']:.4f}")
            print(f"  - Flash: {reward_info['flash_escape_bonus']:.4f}")
            print(f"  - Movement: {reward_info['movement_penalty']:.4f}")
            print(f"  - Loop: {reward_info['loop_penalty']:.4f}")
            print(f"  - Wasteful Flash: {reward_info['wasteful_flash_penalty']:.4f}")
        
        assert -15 <= reward_list[0] <= 15, f"奖励应该在 [-15, 15] 范围内"
        print(f"\n[PASS] 奖励值在合理范围内")
        
    except Exception as e:
        print(f"✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试状态2：死亡
    print("\n" + "-" * 70)
    print("模拟游戏状态 2: 英雄死亡")
    print("-" * 70)
    
    env_info["is_dead"] = True
    hero_pos = {"x": 50.0, "z": 50.0}
    
    try:
        feature, legal_action_out, reward_list, reward_info = p.feature_process(
            hero_pos=hero_pos,
            env_info=env_info,
            action_mask=legal_action,
            last_action=1,
        )
        
        print(f"[PASS] 死亡时的处理正确")
        print(f"  - 奖励: {reward_list[0]:.4f}")
        assert reward_list[0] < 0, "死亡应该产生负奖励"
        print(f"[PASS] 死亡确实产生了负奖励")
        
    except Exception as e:
        print(f"✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("[PASS] 所有集成测试通过!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_preprocessor_integration()
    sys.exit(0 if success else 1)
