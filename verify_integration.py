#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
验证脚本：确认新三层奖励系统已正确集成
Verification script: Confirm three-layer reward system integration
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 70)
    print("集成验证：三层奖励系统 + Preprocessor")
    print("=" * 70)
    print()
    
    # Step 1: 验证导入
    print("[1] 验证模块导入...")
    try:
        from agent_ppo.feature.reward_system_v2 import compute_reward_three_layer, RewardCalculator
        print("   [OK] reward_system_v2 导入成功")
    except Exception as e:
        print(f"   [FAIL] reward_system_v2 导入失败: {e}")
        return False
    
    try:
        from agent_ppo.feature.preprocessor import Preprocessor
        print("   [OK] Preprocessor 导入成功")
    except Exception as e:
        print(f"   [FAIL] Preprocessor 导入失败: {e}")
        return False
    
    # Step 2: 验证配置
    print()
    print("[2] 验证配置...")
    try:
        from agent_ppo.conf.conf import Config
        print(f"   [OK] 特征维度: {Config.FEATURES}")
        print(f"   [OK] 总维度: {Config.DIM_OF_OBSERVATION}")
        assert Config.DIM_OF_OBSERVATION == 159, f"预期 159，得到 {Config.DIM_OF_OBSERVATION}"
        print("   [OK] 配置正确")
    except Exception as e:
        print(f"   [FAIL] 配置错误: {e}")
        return False
    
    # Step 3: 测试奖励计算
    print()
    print("[3] 测试奖励计算...")
    try:
        from collections import deque, Counter
        import numpy as np
        
        reward, details = compute_reward_three_layer(
            step_no=50,
            max_step=1000,
            is_dead=False,
            cur_min_monster_dist_norm=0.5,
            last_min_monster_dist_norm=0.45,
            nearest_treasure_dist_norm=0.3,
            last_nearest_treasure_dist_norm=0.35,
            treasures_collected=3,
            total_treasure=15,
            collected_buff=1,
            total_buff=5,
            buff_remain=20.0,
            flash_count=2,
            last_flash_count=1,
            move_dist=1.5,
            hero_cell=(32, 32),
            recent_cells=deque([(32, 32), (32, 31)], maxlen=12),
            visit_counter=Counter({(32, 32): 10}),
            prev_action=0,
            last_action=1,
            last_treasures_collected=2,
            last_collected_buff=0,
        )
        
        print(f"   [OK] 总奖励: {reward:.4f}")
        print(f"   [OK] Layer A: {details['layer_a']:.4f}")
        print(f"   [OK] Layer B: {details['layer_b']:.4f}")
        print(f"   [OK] Layer C: {details['layer_c']:.4f}")
        assert -15 <= reward <= 15, f"奖励范围错误: {reward}"
        print("   [OK] 奖励值在合理范围内")
    except Exception as e:
        print(f"   [FAIL] 奖励计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: 验证 Preprocessor 初始化
    print()
    print("[4] 验证 Preprocessor 初始化...")
    try:
        p = Preprocessor()
        print(f"   [OK] Preprocessor 初始化成功")
    except Exception as e:
        print(f"   [FAIL] Preprocessor 初始化失败: {e}")
        return False
    
    print()
    print("=" * 70)
    print("验证完成：所有组件正常")
    print("=" * 70)
    print()
    print("下一步：")
    print("  1. 运行: python train_test.py")
    print("  2. 更新 algorithm_name 为 'ppo'")
    print("  3. 观察训练日志中的奖励值是否合理")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
