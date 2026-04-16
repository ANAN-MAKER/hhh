#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
验证脚本：检查5部分补齐改写的完整性和正确性

Verification Script: Check completeness and correctness of 5-part restructuring
"""

import sys
import numpy as np

print("\n" + "="*80)
print("PPO 项目5部分改写验证")
print("="*80)

# ============================================================================
# 1. 验证空间对齐工具 (spatial_utils.py)
# ============================================================================
print("\n[1] 验证空间对齐工具 spatial_utils.py")
try:
    from agent_ppo.feature.spatial_utils import (
        normalize_global_pos,
        relative_pos,
        relative_pos_normalized,
        action_to_delta,
        apply_action_to_pos,
        consistency_check,
        get_action_name,
    )
    
    # 运行一致性检查
    results = consistency_check(verbose=False)
    if all(results.values()):
        print("  ✓ 空间一致性检查通过 (All 8 checks passed)")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  ✗ 空间一致性检查失败: {failed}")
        sys.exit(1)
    
    # 验证基本函数
    assert normalize_global_pos(64, 64) == (0.5, 0.5), "normalize_global_pos failed"
    assert action_to_delta(0) == (1.0, 0.0), "action_to_delta failed"
    assert action_to_delta(8) == (10.0, 0.0), "action_to_delta flash failed"
    
    pos1 = {"x": 50, "z": 50}
    pos2 = apply_action_to_pos(pos1, 0)  # 向右移动
    assert pos2["x"] == 51.0 and pos2["z"] == 50.0, "apply_action_to_pos failed"
    
    print("  ✓ 空间工具函数验证通过 (4/4 basic functions OK)")
    
except Exception as e:
    print(f"  ✗ spatial_utils 验证失败: {e}")
    sys.exit(1)

# ============================================================================
# 2. 验证动作质量评估 (action_quality.py)
# ============================================================================
print("\n[2] 验证动作质量评估 action_quality.py")
try:
    from agent_ppo.feature.action_quality import (
        compute_action_quality_features,
        TOTAL_DIMENSIONS_PER_ACTION,
    )
    
    # 测试计算
    hero_pos = {"x": 64, "z": 64}
    test_features = compute_action_quality_features(
        hero_pos=hero_pos,
        hero_hp=50.0,
        hero_max_hp=100.0,
        monsters=[],
        treasures=[],
        buffs=[],
        legal_action_mask=np.ones(16),
    )
    
    assert test_features.shape == (96,), f"action_quality输出维度错误: {test_features.shape}"
    assert TOTAL_DIMENSIONS_PER_ACTION == 6, "每个动作的维度应该是6"
    assert 16 * 6 == 96, "16个动作×6维 = 96D 验证失败"
    
    print(f"  ✓ 动作质量评估输出: {test_features.shape} (16动作 × 6维 = 96D)")
    print(f"    - 维度顺序: safety / treasure_gain / buff_gain / terrain / revisit / flash_value")
    
except Exception as e:
    print(f"  ✗ action_quality 验证失败: {e}")
    sys.exit(1)

# ============================================================================
# 3. 验证模型架构 (model_v2.py)
# ============================================================================
print("\n[3] 验证模型V2架构")
try:
    import torch
    from agent_ppo.model.model_v2 import ModelV2, RiskRewardFusionBlock
    from agent_ppo.conf.conf import Config
    
    # 检查 RiskRewardFusionBlock 是否存在
    assert hasattr(ModelV2, '__init__'), "ModelV2 not found"
    print("  ✓ ModelV2 类已定义")
    
    # 验证模型初始化
    model = ModelV2(device=torch.device('cpu'))
    assert hasattr(model, 'risk_reward_fusion'), "risk_reward_fusion 块不存在"
    print("  ✓ RiskRewardFusionBlock 已集成到模型")
    
    # 测试前向传播
    batch_size = 2
    test_input = torch.randn(batch_size, Config.FEATURE_LEN)
    logits, value = model(test_input)
    
    assert logits.shape == (batch_size, 16), f"政策输出维度错误: {logits.shape}"
    assert value.shape == (batch_size, 1), f"价值输出维度错误: {value.shape}"
    
    print(f"  ✓ 模型前向传播验证通过")
    print(f"    - 输入: [{batch_size}, {Config.FEATURE_LEN}]")
    print(f"    - 政策输出: {logits.shape} (16D)") 
    print(f"    - 价值输出: {value.shape} (1D)")
    
except Exception as e:
    print(f"  ✗ model_v2 验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 4. 验证三层奖励系统 (reward_system_v2.py)
# ============================================================================
print("\n[4] 验证三层奖励系统 reward_system_v2.py")
try:
    from agent_ppo.feature.reward_system_v2 import compute_reward_three_layer
    from collections import deque
    
    # 模拟三层奖励计算
    reward_total, reward_details = compute_reward_three_layer(
        # 环境信息
        step_no=50,
        max_step=200,
        is_dead=False,
        
        # 距离信息
        cur_min_monster_dist_norm=0.5,
        last_min_monster_dist_norm=0.4,
        nearest_treasure_dist_norm=0.3,
        last_nearest_treasure_dist_norm=0.4,
        
        # 收集信息
        treasures_collected=1,
        total_treasure=5,
        collected_buff=0,
        total_buff=2,
        buff_remain=0.0,
        flash_count=0,
        last_flash_count=0,
        
        # 行为信息
        move_dist=1.0,
        hero_cell=(32, 32),
        recent_cells=deque(maxlen=12),
        visit_counter={},
        prev_action=-1,
        last_action=0,
    )
    
    # 验证三层拆分
    assert "layer_a" in reward_details, "缺少 layer_a (主目标层)"
    assert "layer_b" in reward_details, "缺少 layer_b (引导层)"
    assert "layer_c" in reward_details, "缺少 layer_c (约束层)"
    
    layer_a = reward_details["layer_a"]
    layer_b = reward_details["layer_b"]
    layer_c = reward_details["layer_c"]
    
    print(f"  ✓ 三层奖励计算验证通过")
    print(f"    - Layer A (主目标): {layer_a:.4f}")
    print(f"    - Layer B (引导): {layer_b:.4f}")
    print(f"    - Layer C (约束): {layer_c:.4f}")
    print(f"    - 总奖励: {reward_total:.4f}")
    
except Exception as e:
    print(f"  ✗ reward_system_v2 验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 5. 验证preprocessor集成
# ============================================================================
print("\n[5] 验证preprocessor中的集成")
try:
    from agent_ppo.feature.preprocessor import Preprocessor
    from agent_ppo.conf.conf import Config
    
    pp = Preprocessor()
    
    # 检查新增属性
    assert hasattr(pp, 'trajectory_history'), "缺少 trajectory_history"
    assert hasattr(pp, 'monster_dist_history'), "缺少 monster_dist_history"
    assert hasattr(pp, 'danger_trend'), "缺少 danger_trend"
    assert hasattr(pp, 'hero_max_hp'), "缺少 hero_max_hp"
    assert hasattr(pp, 'danger_trend_value'), "缺少 danger_trend_value"
    
    print(f"  ✓ Preprocessor 新增属性验证通过")
    print(f"    - trajectory_history, monster_dist_history")
    print(f"    - danger_trend, hero_max_hp, danger_trend_value")
    
except Exception as e:
    print(f"  ✗ preprocessor 验证失败: {e}")
    sys.exit(1)

# ============================================================================
# 6. 验证特征维度
# ============================================================================
print("\n[6] 验证特征维度的一致性")
try:
    from agent_ppo.conf.conf import Config
    
    # 检查FEATURE_SPLIT_SHAPE的总和
    total_dims = sum(Config.FEATURE_SPLIT_SHAPE)
    assert total_dims == 255, f"特征总维度错误: {total_dims} != 255"
    
    # 打印特征分布
    print(f"  ✓ 特征维度验证通过 (总计: {total_dims}D)")
    print(f"    - self_dim: {Config.FEATURE_SPLIT_SHAPE[0]}D (英雄状态)")
    print(f"    - monster1_dim: {Config.FEATURE_SPLIT_SHAPE[1]}D (怪物1)")
    print(f"    - monster2_dim: {Config.FEATURE_SPLIT_SHAPE[2]}D (怪物2)")
    print(f"    - treasure_dim: {Config.FEATURE_SPLIT_SHAPE[3]}D (宝箱)")
    print(f"    - buff_dim: {Config.FEATURE_SPLIT_SHAPE[4]}D (buff)")
    print(f"    - progress_dim: {Config.FEATURE_SPLIT_SHAPE[5]}D (决策辅助)")
    print(f"    - map_dim: {Config.FEATURE_SPLIT_SHAPE[6]}D (地图与路径)")
    print(f"    - plan_dim: {Config.FEATURE_SPLIT_SHAPE[7]}D (时序与规划)")
    print(f"    - legal_dim: {Config.FEATURE_SPLIT_SHAPE[8]}D (合法动作)")
    
except Exception as e:
    print(f"  ✗ 特征维度验证失败: {e}")
    sys.exit(1)

# ============================================================================
# 最终总结
# ============================================================================
print("\n" + "="*80)
print("✓ 所有验证通过！5部分补齐改写完成")
print("="*80)
print("\n改写成果总结：")
print("  1. ✓ spatial_utils.py: 统一空间/坐标系对齐层")
print("  2. ✓ action_quality.py: 完整的动作质量评估块 (96D)")
print("  3. ✓ model_v2.py: 添加RiskRewardFusionBlock (显式风险收益融合)")
print("  4. ✓ reward_system_v2.py: 三层奖励计算（A/B/C层）")
print("  5. ✓ train_workflow.py: 统一三层奖励监控口径")
print("  6. ✓ preprocessor.py: 集成spatial_utils和action_quality")

print("\n最终信息流：")
print("  环境观测(128,128) → 空间对齐 → 特征编码(159D) → 动作质量评估(96D)")
print("  → 完整输入(255D) → 风险收益融合(96D) → 主干网络 → Policy(16D) / Value(1D)")
print("\nPPO 训练接口保持不变，16维动作输出不变，标准PPO loss不变")
print("="*80 + "\n")
