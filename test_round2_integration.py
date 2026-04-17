#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
第二轮代码修改综合验收测试

测试项目：
1. Task A - 动作仿真器集成到action_quality
2. Tasks B-G - 前序修改的保障
3. 完整的特征提取和training流程
"""

import sys
import os
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Test 1: 动作仿真器能否正常工作
# ============================================================================

def test_action_simulator():
    """测试Task A - 动作仿真器"""
    logger.info("=" * 80)
    logger.info("TEST 1: Action Simulator (Task A)")
    logger.info("=" * 80)
    
    try:
        from agent_ppo.feature.action_simulator import simulate_action, simulate_move, simulate_flash
        logger.info("✓ action_simulator module imported successfully")
        
        # 测试数据
        hero_pos = {"x": 64.0, "z": 64.0}
        local_map = np.ones((21, 21), dtype=np.float32)
        treasures = [{"x": 70.0, "z": 70.0}]
        buffs = [{"x": 72.0, "z": 72.0}]
        
        # 测试simulate_move
        result = simulate_move(hero_pos, 0, local_map, buff_active=False)
        assert "next_pos" in result
        assert "blocked" in result
        logger.info(f"✓ simulate_move works: next_pos={result['next_pos']}, blocked={result['blocked']}")
        
        # 测试simulate_flash
        result = simulate_flash(hero_pos, 8, local_map)
        assert "flash_pos" in result
        assert "reachable" in result
        logger.info(f"✓ simulate_flash works: flash_pos={result['flash_pos']}, reachable={result['reachable']}")
        
        # 测试simulate_action
        result = simulate_action(hero_pos, 0, local_map, buff_active=False, treasures=treasures, buffs=buffs)
        assert "next_pos" in result or "flash_pos" in result
        assert "collected_treasures" in result
        assert "is_flash" in result
        logger.info(f"✓ simulate_action works: collected_treasures={result['collected_treasures']}")
        
        logger.info("✓ Test 1 PASSED: Action Simulator\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 1 FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 2: action_quality能否使用simulate_action
# ============================================================================

def test_action_quality_integration():
    """测试Task A - action_quality集成simulate_action"""
    logger.info("=" * 80)
    logger.info("TEST 2: Action Quality Integration (Task A)")
    logger.info("=" * 80)
    
    try:
        from agent_ppo.feature.action_quality import compute_action_quality_features
        logger.info("✓ action_quality module imported successfully")
        
        # 测试数据
        hero_pos = {"x": 64.0, "z": 64.0}
        monsters = [{"pos": {"x": 50.0, "z": 50.0}, "speed": 1.5}]
        treasures = [{"pos": {"x": 70.0, "z": 70.0}}]
        buffs = [{"pos": {"x": 72.0, "z": 72.0}}]
        legal_action_mask = np.ones(16, dtype=np.float32)
        local_map = np.ones((21, 21), dtype=np.float32)
        
        # 调用compute_action_quality_features，最重要的是测试buff_active参数
        features = compute_action_quality_features(
            hero_pos=hero_pos,
            hero_hp=80.0,
            hero_max_hp=100.0,
            monsters=monsters,
            treasures=treasures,
            buffs=buffs,
            legal_action_mask=legal_action_mask,
            local_map=local_map,
            recent_positions=[],
            flash_cooldown=0,
            danger_trend=0.0,
            buff_active=False,  # Task A的关键：传递buff_active参数
        )
        
        assert features.shape == (96,), f"Expected shape (96,), got {features.shape}"
        assert np.all(np.isfinite(features)), "Features contain NaN or Inf"
        logger.info(f"✓ compute_action_quality_features works: shape={features.shape}, sample values={features[:6]}")
        
        # 测试buff_active=True的情况
        features_with_buff = compute_action_quality_features(
            hero_pos=hero_pos,
            hero_hp=80.0,
            hero_max_hp=100.0,
            monsters=monsters,
            treasures=treasures,
            buffs=buffs,
            legal_action_mask=legal_action_mask,
            local_map=local_map,
            recent_positions=[],
            flash_cooldown=0,
            danger_trend=0.0,
            buff_active=True,  # Task A的关键：测试buff_active=True
        )
        
        assert features_with_buff.shape == (96,), f"Expected shape (96,), got {features_with_buff.shape}"
        logger.info(f"✓ compute_action_quality_features with buff_active=True works")
        
        logger.info("✓ Test 2 PASSED: Action Quality Integration\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 2 FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 3: preprocessor能否完整运行
# ============================================================================

def test_preprocessor_integration():
    """测试preprocessor的完整流程"""
    logger.info("=" * 80)
    logger.info("TEST 3: Preprocessor Integration")
    logger.info("=" * 80)
    
    try:
        from agent_ppo.feature.preprocessor import Preprocessor
        logger.info("✓ Preprocessor module imported successfully")
        
        preprocessor = Preprocessor()
        
        # 构造完整的环境观测
        obs = {
            "observation": {
                "step_no": 0,
                "frame_state": {
                    "heroes": {
                        "x": 64.0,
                        "z": 64.0,
                        "hp": 100.0,
                        "max_hp": 100.0,
                        "flash_cooldown": 0.0,
                        "buff_remaining_time": 0.0,
                        "status": 1,
                    },
                    "monsters": [
                        {
                            "x": 50.0,
                            "z": 50.0,
                            "speed": 1.5,
                            "hero_l2_distance": 20.0,
                        }
                    ],
                    "organs": [
                        {
                            "x": 70.0,
                            "z": 70.0,
                            "sub_type": 1,
                            "status": 1,
                        },
                        {
                            "x": 72.0,
                            "z": 72.0,
                            "sub_type": 2,
                            "status": 1,
                        }
                    ],
                },
                "env_info": {
                    "max_step": 200,
                    "treasures_collected": 0,
                    "total_treasure": 5,
                    "collected_buff": 0,
                    "total_buff": 2,
                    "flash_count": 0,
                    "total_score": 0.0,
                    "is_dead": False,
                },
                "map_info": np.ones((21, 21), dtype=np.float32),
                "legal_action": [1] * 16,
            }
        }
        
        # 调用feature_process
        feature, legal_action, reward, reward_info = preprocessor.feature_process(obs, last_action=0)
        
        assert feature.shape == (255,), f"Expected feature shape (255,), got {feature.shape}"
        assert len(legal_action) == 16, f"Expected legal_action length 16, got {len(legal_action)}"
        assert len(reward) == 1, f"Expected reward length 1, got {len(reward)}"
        assert np.all(np.isfinite(feature)), "Feature contains NaN or Inf"
        logger.info(f"✓ preprocessor.feature_process works: feature_shape={feature.shape}")
        
        # 验证action_quality特征被正确计算
        assert reward_info.get("action_quality_computed", False), "action_quality was not computed"
        logger.info(f"✓ action_quality_computed = True")
        
        logger.info("✓ Test 3 PASSED: Preprocessor Integration\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 3 FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 4: Policy Distribution (Task B)
# ============================================================================

def test_policy_distribution():
    """测试Task B - 统一Policy分布"""
    logger.info("=" * 80)
    logger.info("TEST 4: Unified Policy Distribution (Task B)")
    logger.info("=" * 80)
    
    try:
        from agent_ppo.agent import Agent
        from agent_ppo.conf.conf import Config
        logger.info("✓ Agent and Config modules imported successfully")
        
        config = Config()
        agent = Agent(config)
        
        # 验证agent有_build_masked_dist方法
        assert hasattr(agent, '_build_masked_dist'), "Agent missing _build_masked_dist method"
        logger.info("✓ Agent has _build_masked_dist method")
        
        # 测试predict方法能否生成有效的action和logprob
        obs = np.random.randn(1, config.OBS_DIM).astype(np.float32)
        legal_action = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        
        action, logprob = agent.predict(obs, legal_action)
        
        assert action.shape == (1,), f"Expected action shape (1,), got {action.shape}"
        assert logprob.shape == (1,), f"Expected logprob shape (1,), got {logprob.shape}"
        assert np.all(np.isfinite(logprob)), "logprob contains NaN or Inf"
        logger.info(f"✓ agent.predict works: action={action[0]}, logprob={logprob[0]:.4f}")
        
        logger.info("✓ Test 4 PASSED: Unified Policy Distribution\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 4 FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 5: Data Structures (Task C)
# ============================================================================

def test_data_structures():
    """测试Task C - 扩展的数据结构"""
    logger.info("=" * 80)
    logger.info("TEST 5: Extended Data Structures (Task C)")
    logger.info("=" * 80)
    
    try:
        from agent_ppo.feature.definition import ActData, SampleData
        logger.info("✓ definition module imported successfully")
        
        # 测试ActData有logprob字段
        act_data = ActData()
        assert hasattr(act_data, 'logprob'), "ActData missing logprob field"
        logger.info("✓ ActData has logprob field")
        
        # 测试SampleData有新增字段
        sample_data = SampleData()
        required_fields = ['old_logprob', 'old_value', 'next_obs', 'next_legal_action', 'return_']
        for field in required_fields:
            assert hasattr(sample_data, field), f"SampleData missing {field} field"
        logger.info(f"✓ SampleData has all required fields: {required_fields}")
        
        logger.info("✓ Test 5 PASSED: Extended Data Structures\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 5 FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 6: PPO Training (Task E)
# ============================================================================

def test_ppo_training():
    """测试Task E - 标准PPO训练"""
    logger.info("=" * 80)
    logger.info("TEST 6: PPO Training (Task E)")
    logger.info("=" * 80)
    
    try:
        from agent_ppo.algorithm.algorithm import PPOAlgorithm
        from agent_ppo.conf.conf import Config
        logger.info("✓ PPOAlgorithm and Config modules imported successfully")
        
        config = Config()
        algorithm = PPOAlgorithm(config)
        
        # 验证PPOAlgorithm有标准PPO的关键方法
        assert hasattr(algorithm, 'learn'), "PPOAlgorithm missing learn method"
        assert hasattr(algorithm, '_compute_policy_loss'), "PPOAlgorithm missing _compute_policy_loss method"
        assert hasattr(algorithm, '_compute_value_loss'), "PPOAlgorithm missing _compute_value_loss method"
        logger.info("✓ PPOAlgorithm has all required PPO methods")
        
        # 验证配置中有PPO参数
        assert hasattr(config, 'EPOCHS'), "Config missing EPOCHS"
        assert hasattr(config, 'BATCH_SIZE'), "Config missing BATCH_SIZE"
        assert hasattr(config, 'TARGET_KL'), "Config missing TARGET_KL"
        logger.info("✓ Config has all required PPO parameters")
        
        logger.info("✓ Test 6 PASSED: PPO Training\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 6 FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test 7: Model Architecture (Task F)
# ============================================================================

def test_model_architecture():
    """测试Task F - 模型架构双头Critic"""
    logger.info("=" * 80)
    logger.info("TEST 7: Model Architecture (Task F)")
    logger.info("=" * 80)
    
    try:
        from agent_ppo.model.model import Model
        from agent_ppo.conf.conf import Config
        logger.info("✓ Model module imported successfully")
        
        config = Config()
        model = Model(config)
        
        # 验证模型能否前向传播
        obs = np.random.randn(2, config.OBS_DIM).astype(np.float32)
        obs_tensor = model._to_tensor(obs)
        
        # 调用前向传播
        policy_logits, value, auxiliary_outputs = model(obs_tensor)
        
        assert policy_logits.shape == (2, 16), f"Expected policy_logits shape (2, 16), got {policy_logits.shape}"
        logger.info(f"✓ Model forward pass works: policy_logits shape={policy_logits.shape}, value shape={value.shape if hasattr(value, 'shape') else 'scalar'}")
        
        logger.info("✓ Test 7 PASSED: Model Architecture\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test 7 FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    """运行所有测试"""
    logger.info("\n" + "=" * 80)
    logger.info("第二轮代码修改综合验收测试")
    logger.info("=" * 80 + "\n")
    
    tests = [
        test_action_simulator,
        test_action_quality_integration,
        test_preprocessor_integration,
        test_policy_distribution,
        test_data_structures,
        test_ppo_training,
        test_model_architecture,
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append((test_func.__name__, result))
    
    # 总结
    logger.info("=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info("-" * 80)
    logger.info(f"总计: {passed}/{total} 测试通过")
    logger.info("=" * 80 + "\n")
    
    if passed == total:
        logger.info("🎉 所有测试通过！第二轮代码修改完成。")
        return 0
    else:
        logger.error(f"❌ 有 {total - passed} 个测试失败。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
