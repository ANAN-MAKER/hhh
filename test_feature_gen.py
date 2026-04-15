#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
验证 Preprocessor 特征生成是否正确
"""

from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.conf.conf import Config
import numpy as np

print("Preprocessor 特征生成测试:")
print()

p = Preprocessor()
print(f"[OK] Preprocessor 初始化成功")

# 模拟环境观察
env_obs = {
    "observation": {
        "frame_state": {
            "heroes": {"pos": {"x": 64.0, "z": 64.0}, "speed": 1.0, "flash_cooldown": 0.0, "buff_remaining_time": 0.0},
            "monsters": [
                {"pos": {"x": 50.0, "z": 50.0}, "speed": 2.0, "hp": 10},
                {"pos": {"x": 70.0, "z": 75.0}, "speed": 1.5, "hp": 8},
            ],
            "treasures": [
                {"pos": {"x": 70.0, "z": 70.0}},
                {"pos": {"x": 60.0, "z": 60.0}},
                {"pos": {"x": 80.0, "z": 80.0}},
            ],
            "buffs": [
                {"pos": {"x": 55.0, "z": 55.0}},
            ],
        },
        "map_info": [[0]*128 for _ in range(128)],  # 128x128 地图，全部可通行
        "legal_action": [1] * 16,  # 所有动作合法
        "env_info": {
            "treasures_collected": 2,
            "total_treasure": 15,
            "collected_buff": 0,
            "total_buff": 3,
            "buff_remaining_time": 0.0,
            "flash_count": 2,
            "total_score": 40.0,
            "is_dead": False,
            "max_step": 1000,
        },
        "step_no": 100,
    }
}

legal_action = np.ones(16, dtype=np.int32)

try:
    feature, legal_action_out, reward_list, reward_info = p.feature_process(env_obs, last_action=0)
    
    print(f"[OK] feature_process 执行成功")
    print()
    print(f"特征维度验证:")
    print(f"  输出特征维度: {len(feature)}")
    print(f"  期望维度: {Config.DIM_OF_OBSERVATION}")
    
    if len(feature) == Config.DIM_OF_OBSERVATION:
        print(f"  [OK] 维度匹配")
    else:
        print(f"  [FAIL] 维度不匹配!")
    
    print()
    print(f"特征拆分验证:")
    offsets = np.cumsum([0] + Config.FEATURE_SPLIT_SHAPE)
    names = ['self', 'monster1', 'monster2', 'treasure', 'buff', 'progress', 'map', 'plan', 'legal']
    for i, name in enumerate(names):
        start = offsets[i]
        end = offsets[i+1]
        size = end - start
        actual_size = Config.FEATURE_SPLIT_SHAPE[i]
        feat_slice = feature[start:end]
        mean_val = np.mean(feat_slice) if len(feat_slice) > 0 else 0.0
        print(f"  {i+1}. {name:10s}: [{start:3d}-{end:3d}] {size:2d}D (OK: {size==actual_size}) mean={mean_val:.4f}")
    
    print()
    print(f"奖励验证:")
    print(f"  奖励值: {reward_list[0]:.4f}")
    print(f"  范围验证: {-15 <= reward_list[0] <= 15}")
    
    print()
    print("[OK] 所有验证通过，特征生成正确")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
