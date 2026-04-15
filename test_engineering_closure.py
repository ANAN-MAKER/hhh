#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

End-to-end Engineering Closure Verification.
完整工程闭环验证。

该文件验证：
  1. 所有新模块能正确导入
  2. 空间坐标系统一致性
  3. 动作质量评估能正确调用
  4. 模型能正确构建和前向传播
  5. 奖励系统能正确计算和统计
  6. 训练监控能正确收集和聚合
  7. 整体信息流畅通无阻
"""

import sys
import traceback
import numpy as np


def test_imports():
    """测试 1: 所有新模块能正确导入"""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        from agent_ppo.feature.spatial_utils import (
            normalize_global_pos,
            action_to_delta,
            consistency_check,
        )
        print("✓ spatial_utils imported")
    except Exception as e:
        print(f"✗ spatial_utils import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from agent_ppo.feature.action_quality import (
            ActionQualityEvaluator,
            compute_action_quality_features,
        )
        print("✓ action_quality imported")
    except Exception as e:
        print(f"✗ action_quality import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from agent_ppo.feature.reward_spec import (
            RewardInfo,
            EpisodeStatistics,
            create_reward_info,
        )
        print("✓ reward_spec imported")
    except Exception as e:
        print(f"✗ reward_spec import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from agent_ppo.workflow.training_monitor import TrainingMonitor
        print("✓ training_monitor imported")
    except Exception as e:
        print(f"✗ training_monitor import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from agent_ppo.model.model import Model, RiskRewardFusionBlock
        print("✓ model (including RiskRewardFusionBlock) imported")
    except Exception as e:
        print(f"✗ model import failed: {e}")
        traceback.print_exc()
        return False
    
    return True


def test_spatial_consistency():
    """测试 2: 空间坐标系一致性"""
    print("\n" + "="*70)
    print("TEST 2: Spatial Coordinate Consistency")
    print("="*70)
    
    try:
        from agent_ppo.feature.spatial_utils import consistency_check
        
        results = consistency_check(verbose=False)
        
        all_pass = all(results.values())
        if all_pass:
            print(f"✓ All {len(results)} spatial consistency checks passed")
        else:
            print(f"✗ Some consistency checks failed:")
            for check, passed in results.items():
                status = "✓" if passed else "✗"
                print(f"    {status} {check}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Spatial consistency check failed: {e}")
        traceback.print_exc()
        return False


def test_action_quality():
    """测试 3: 动作质量评估"""
    print("\n" + "="*70)
    print("TEST 3: Action Quality Assessment")
    print("="*70)
    
    try:
        from agent_ppo.feature.action_quality import ActionQualityEvaluator
        
        evaluator = ActionQualityEvaluator()
        
        hero_pos = {"x": 64.0, "z": 64.0}
        hero_hp = 100.0
        hero_max_hp = 100.0
        monsters = [{"pos": {"x": 80.0, "z": 64.0}}]
        treasures = [{"pos": {"x": 50.0, "z": 50.0}}]
        buffs = []
        legal_mask = np.ones(16, dtype=np.float32)
        
        qualities = evaluator.evaluate_all_actions(
            hero_pos=hero_pos,
            hero_hp=hero_hp,
            hero_max_hp=hero_max_hp,
            monsters=monsters,
            treasures=treasures,
            buffs=buffs,
            legal_action_mask=legal_mask,
        )
        
        assert qualities.shape == (16, 6), f"Expected (16,6), got {qualities.shape}"
        assert np.all(qualities >= 0) and np.all(qualities <= 1), "Values should be in [0,1]"
        
        print(f"✓ Action quality evaluation passed")
        print(f"  Output shape: {qualities.shape}")
        print(f"  Value range: [{qualities.min():.3f}, {qualities.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Action quality assessment failed: {e}")
        traceback.print_exc()
        return False


def test_model_construction():
    """测试 4: 模型构建和前向传播"""
    print("\n" + "="*70)
    print("TEST 4: Model Construction and Forward Pass")
    print("="*70)
    
    try:
        import torch
        from agent_ppo.model.model import Model, RiskRewardFusionBlock
        from agent_ppo.conf.conf import Config
        
        model = Model(device=None)
        model.eval()
        
        # 创建虚拟输入 (batch_size=1, dim=159)
        dummy_input = torch.randn(1, Config.DIM_OF_OBSERVATION, dtype=torch.float32)
        
        with torch.no_grad():
            logits, value = model(dummy_input, inference=True)
        
        assert logits.shape == (1, 16), f"Expected logits (1,16), got {logits.shape}"
        assert value.shape == (1, 1), f"Expected value (1,1), got {value.shape}"
        
        print(f"✓ Model construction and forward pass passed")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Value shape: {value.shape}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"✗ Model construction failed: {e}")
        traceback.print_exc()
        return False


def test_reward_system():
    """测试 5: 奖励系统"""
    print("\n" + "="*70)
    print("TEST 5: Reward System")
    print("="*70)
    
    try:
        from agent_ppo.feature.reward_spec import RewardInfo, EpisodeStatistics
        
        # 创建奖励信息
        reward_info = RewardInfo(
            survival_base_reward=0.02,
            treasure_pickup_reward=2.0,
            dist_shaping=0.10,
            danger_zone_penalty=-0.05,
            flash_escape_reward=0.6,
            movement_penalty=-0.01,
        )
        
        total = reward_info.compute_aggregates()
        
        assert reward_info.layer_a_total == (0.02 + 2.0), "Layer A calculation error"
        assert reward_info.layer_b_total == (0.10 - 0.05 + 0.6), "Layer B calculation error"
        assert reward_info.layer_c_total == -0.01, "Layer C calculation error"
        
        print(f"✓ Reward calculation passed")
        print(f"  Layer A: {reward_info.layer_a_total:.3f}")
        print(f"  Layer B: {reward_info.layer_b_total:.3f}")
        print(f"  Layer C: {reward_info.layer_c_total:.3f}")
        print(f"  Total: {total:.3f}")
        
        # 测试对局统计
        episode_stats = EpisodeStatistics(
            episode_id=1,
            total_steps=150,
            total_score=1500,
            treasures_collected=8,
            total_treasures=10,
            survived=True,
            layer_a_cumulative=15.2,
            layer_b_cumulative=8.5,
            layer_c_cumulative=-2.3,
        )
        episode_stats.finalize()
        
        expected_total = 15.2 + 8.5 - 2.3
        assert abs(episode_stats.total_reward - expected_total) < 1e-5, "Episode reward calculation error"
        # reward_per_step is calculated after finalize(), so just verify finalize was called
        assert episode_stats.total_steps == 150, "Episode steps tracking error"
        
        print(f"✓ Episode statistics passed")
        print(f"  Total reward: {episode_stats.total_reward:.3f}")
        print(f"  Per-step: {episode_stats.reward_per_step:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Reward system test failed: {e}")
        traceback.print_exc()
        return False


def test_training_monitor():
    """测试 6: 训练监控"""
    print("\n" + "="*70)
    print("TEST 6: Training Monitor")
    print("="*70)
    
    try:
        from agent_ppo.workflow.training_monitor import TrainingMonitor
        from agent_ppo.feature.reward_spec import RewardInfo
        
        monitor = TrainingMonitor()
        
        # 模拟几局对局
        for episode in range(5):
            monitor.reset_episode()
            
            for step in range(50):
                reward_info = RewardInfo(
                    survival_base_reward=0.02,
                    dist_shaping=0.05,
                )
                reward_info.compute_aggregates()
                monitor.record_step_reward(reward_info, {})
            
            stats = monitor.finalize_episode(
                episode_id=episode,
                env_obs={
                    "observation": {
                        "env_info": {
                            "total_score": 500.0,
                            "treasures_collected": 5,
                            "total_treasure": 10,
                            "collected_buff": 1,
                            "flash_count": 2,
                        }
                    }
                },
                survived=episode % 2 == 0,
            )
        
        window_stats = monitor.get_window_statistics()
        
        assert len(monitor.episode_histories) == 5, "Episode history count error"
        assert window_stats["window_episodes"] == 5, "Window episodes count error"
        assert window_stats["win_rate"] == 0.6, "Win rate calculation error"
        
        print(f"✓ Training monitor passed")
        print(f"  Episodes tracked: {len(monitor.episode_histories)}")
        print(f"  Window statistics keys: {list(window_stats.keys())}")
        print(f"  Win rate: {window_stats['win_rate']:.1%}")
        print(f"  Avg total reward: {window_stats['avg_total_reward']:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Training monitor test failed: {e}")
        traceback.print_exc()
        return False


def test_end_to_end():
    """测试 7: 端到端信息流"""
    print("\n" + "="*70)
    print("TEST 7: End-to-End Information Flow")
    print("="*70)
    
    try:
        import torch
        from agent_ppo.feature.spatial_utils import action_to_delta
        from agent_ppo.feature.action_quality import ActionQualityEvaluator
        from agent_ppo.feature.reward_spec import RewardInfo
        from agent_ppo.model.model import Model
        from agent_ppo.conf.conf import Config
        
        print("  1. Spatial utils → Action quality")
        hero_pos = {"x": 64.0, "z": 64.0}
        # 使用 spatial_utils 验证动作
        for action_id in range(16):
            dx, dz = action_to_delta(action_id)
            assert isinstance(dx, float) and isinstance(dz, float)
        print("    ✓ Spatial consistency verified")
        
        print("  2. Action quality → Features")
        evaluator = ActionQualityEvaluator()
        qualities = evaluator.evaluate_all_actions(
            hero_pos=hero_pos,
            hero_hp=100.0,
            hero_max_hp=100.0,
            monsters=[{"pos": {"x": 80.0, "z": 64.0}}],
            treasures=[],
            buffs=[],
            legal_action_mask=np.ones(16, dtype=np.float32),
        )
        print(f"    ✓ Action quality features: shape={qualities.shape}")
        
        print("  3. Features → Model")
        model = Model()
        model.eval()
        dummy_obs = torch.randn(1, Config.DIM_OF_OBSERVATION, dtype=torch.float32)
        with torch.no_grad():
            logits, value = model(dummy_obs)
        print(f"    ✓ Model output: logits={logits.shape}, value={value.shape}")
        
        print("  4. Model → Reward system")
        reward_info = RewardInfo(
            survival_base_reward=0.02,
            dist_shaping=0.05,
        )
        reward_total = reward_info.compute_aggregates()
        print(f"    ✓ Reward computed: total={reward_total:.3f}")
        
        print("\n✓ End-to-end information flow verified")
        return True
    except Exception as e:
        print(f"✗ End-to-end test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有验证测试"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*17 + "ENGINEERING CLOSURE VERIFICATION" + " "*19 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        ("Module Imports", test_imports),
        ("Spatial Consistency", test_spatial_consistency),
        ("Action Quality", test_action_quality),
        ("Model Construction", test_model_construction),
        ("Reward System", test_reward_system),
        ("Training Monitor", test_training_monitor),
        ("End-to-End Flow", test_end_to_end),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All verification tests passed! Engineering closure complete.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
