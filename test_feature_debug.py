#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
详细的特征生成调试
"""

from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.conf.conf import Config
import numpy as np

print("详细特征生成调试:")
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

# 调试：在 feature_process 前提取 legal_action
legal_action_raw = p._extract_legal_action(env_obs["observation"])
print(f"提取的 legal_action: {legal_action_raw}")
print(f"长度: {len(legal_action_raw)}")
print()

try:
    feature, legal_action_out, reward_list, reward_info = p.feature_process(env_obs, last_action=0)
    
    print(f"[OK] feature_process 执行成功")
    print(f"feature 长度: {len(feature)}")
    print(f"legal_action_out 长度: {len(legal_action_out) if isinstance(legal_action_out, (list, np.ndarray)) else 'N/A'}")
    print()
    
    # 检查特征的各个部分
    print("输出特征的结构 (基于 FEATURE_SPLIT_SHAPE):")
    offsets = np.cumsum([0] + Config.FEATURE_SPLIT_SHAPE)
    names = ['self', 'monster1', 'monster2', 'treasure', 'buff', 'progress', 'map', 'plan', 'legal']
    
    total = 0
    for i, (name, size) in enumerate(zip(names, Config.FEATURE_SPLIT_SHAPE)):
        start = offsets[i]
        end = start + size
        if end <= len(feature):
            feat_slice = feature[start:end]
            actual_size = len(feat_slice)
            mean_val = np.mean(feat_slice)
            print(f"{i+1}. {name:10s}: 期望{size:2d}D, 实际{actual_size:2d}D, mean={mean_val:8.4f}, range=[{np.min(feat_slice):.2f}, {np.max(feat_slice):.2f}]")
            total += actual_size
        else:
            missing = min(size, len(feature) - start)
            print(f"{i+1}. {name:10s}: 期望{size:2d}D, 实际{missing:2d}D [缺失{size-missing}D]")
            total += missing
    
    print()
    print(f"总长度: {total} (期望{sum(Config.FEATURE_SPLIT_SHAPE)})")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
