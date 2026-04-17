#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
第二轮代码修改 - 快速集成验证

主要目标：
1. 验证Task A的动作仿真器被正确集成到action_quality
2. 验证所有前序改动仍然有效
"""

import sys
import os

# 添加代码目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """测试所有关键模块都能正常导入"""
    logger.info("\n[Test 1] 验证所有关键模块导入...")
    
    try:
        from agent_ppo.feature.action_simulator import simulate_action, simulate_move, simulate_flash
        logger.info("  ✓ action_simulator")
        
        from agent_ppo.feature.action_quality import compute_action_quality_features, ActionQualityEvaluator
        logger.info("  ✓ action_quality")
        
        from agent_ppo.feature.preprocessor import Preprocessor
        logger.info("  ✓ preprocessor")
        
        from agent_ppo.agent import Agent
        logger.info("  ✓ agent")
        
        from agent_ppo.feature.definition import ActData, SampleData
        logger.info("  ✓ definition")
        
        from agent_ppo.algorithm.algorithm import PPOAlgorithm
        logger.info("  ✓ algorithm")
        
        from agent_ppo.model.model import Model
        logger.info("  ✓ model")
        
        from agent_ppo.conf.conf import Config
        logger.info("  ✓ conf")
        
        logger.info("✓ Test 1 通过：所有模块导入成功\n")
        return True
    except ImportError as e:
        logger.error(f"✗ Test 1 失败：{e}\n")
        return False


def test_action_simulator():
    """测试Task A - 动作仿真器基本功能"""
    logger.info("[Test 2] 验证Task A - 动作仿真器...")
    
    try:
        from agent_ppo.feature.action_simulator import simulate_action, simulate_move, simulate_flash
        
        hero_pos = {"x": 64.0, "z": 64.0}
        local_map = np.ones((21, 21), dtype=np.float32)
        
        # 测试move
        r1 = simulate_move(hero_pos, 0, local_map, buff_active=False)
        assert "next_pos" in r1 and "blocked" in r1
        logger.info(f"  ✓ simulate_move: next_pos={r1['next_pos']}, blocked={r1['blocked']}")
        
        # 测试flash
        r2 = simulate_flash(hero_pos, 8, local_map)
        assert "flash_pos" in r2 and "reachable" in r2
        logger.info(f"  ✓ simulate_flash: reachable={r2['reachable']}")
        
        # 测试simulate_action
        r3 = simulate_action(hero_pos, 0, local_map, buff_active=False)
        assert "next_pos" in r3 and "collected_treasures" in r3 and "is_flash" in r3
        logger.info(f"  ✓ simulate_action: collected_treasures={r3['collected_treasures']}")
        
        logger.info("✓ Test 2 通过：动作仿真器正常\n")
        return True
    except Exception as e:
        logger.error(f"✗ Test 2 失败：{e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_action_quality_with_buff():
    """测试Task A - action_quality接收buff_active参数"""
    logger.info("[Test 3] 验证Task A - action_quality buff_active参数...")
    
    try:
        from agent_ppo.feature.action_quality import compute_action_quality_features
        
        hero_pos = {"x": 64.0, "z": 64.0}
        monsters = [{"pos": {"x": 50.0, "z": 50.0}, "speed": 1.5}]
        treasures = [{"pos": {"x": 70.0, "z": 70.0}}]
        buffs = [{"pos": {"x": 72.0, "z": 72.0}}]
        legal_action_mask = np.ones(16, dtype=np.float32)
        
        # 关键：验证buff_active参数被接受
        features = compute_action_quality_features(
            hero_pos=hero_pos,
            hero_hp=80.0,
            hero_max_hp=100.0,
            monsters=monsters,
            treasures=treasures,
            buffs=buffs,
            legal_action_mask=legal_action_mask,
            local_map=np.ones((21, 21), dtype=np.float32),
            recent_positions=None,
            flash_cooldown=0,
            danger_trend=0.0,
            buff_active=False,  # Task A的关键参数
        )
        
        assert features.shape == (96,), f"Expected (96,), got {features.shape}"
        assert np.all(np.isfinite(features)), "Features contain NaN/Inf"
        logger.info(f"  ✓ compute_action_quality_features(buff_active=False): shape={features.shape}")
        
        # 测试buff_active=True
        features2 = compute_action_quality_features(
            hero_pos=hero_pos, hero_hp=80.0, hero_max_hp=100.0,
            monsters=monsters, treasures=treasures, buffs=buffs,
            legal_action_mask=legal_action_mask,
            local_map=np.ones((21, 21), dtype=np.float32),
            recent_positions=None, flash_cooldown=0, danger_trend=0.0,
            buff_active=True,  # Task A - 测试True情况
        )
        
        assert features2.shape == (96,)
        logger.info(f"  ✓ compute_action_quality_features(buff_active=True): shape={features2.shape}")
        
        logger.info("✓ Test 3 通过：buff_active参数正确集成\n")
        return True
    except TypeError as e:
        if "buff_active" in str(e):
            logger.error(f"✗ Test 3 失败：buff_active参数未正确集成 - {e}\n")
        else:
            logger.error(f"✗ Test 3 失败：{e}\n")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        logger.error(f"✗ Test 3 失败：{e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessor_action_quality():
    """测试preprocessor能否正确调用action_quality的新参数"""
    logger.info("[Test 4] 验证preprocessor使用action_quality(buff_active=...)...")
    
    try:
        from agent_ppo.feature.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        
        # 构造最小的观测结构
        obs = {
            "observation": {
                "step_no": 0,
                "frame_state": {
                    "heroes": {
                        "x": 64.0, "z": 64.0,
                        "hp": 100.0, "max_hp": 100.0,
                        "flash_cooldown": 0.0,
                        "buff_remaining_time": 5.0,  # 有buff
                        "status": 1,
                    },
                    "monsters": [{"x": 50.0, "z": 50.0, "speed": 1.5, "hero_l2_distance": 20.0}],
                    "organs": [
                        {"x": 70.0, "z": 70.0, "sub_type": 1, "status": 1},
                        {"x": 72.0, "z": 72.0, "sub_type": 2, "status": 1},
                    ],
                },
                "env_info": {
                    "max_step": 200,
                    "treasures_collected": 0, "total_treasure": 5,
                    "collected_buff": 0, "total_buff": 2,
                    "flash_count": 0, "total_score": 0.0,
                    "is_dead": False,
                },
                "map_info": np.ones((21, 21), dtype=np.float32),
                "legal_action": [1] * 16,
            }
        }
        
        feature, legal_action, reward, reward_info = preprocessor.feature_process(obs, last_action=0)
        
        assert feature.shape == (255,), f"Expected (255,), got {feature.shape}"
        assert reward_info.get("action_quality_computed", False), "action_quality未被计算"
        logger.info(f"  ✓ preprocessor.feature_process() 成功执行")
        logger.info(f"    - feature shape: {feature.shape}")
        logger.info(f"    - action_quality_computed: {reward_info.get('action_quality_computed')}")
        logger.info(f"    - 检测到buff: buff_remaining_time=5.0")
        
        logger.info("✓ Test 4 通过：preprocessor正确使用action_quality\n")
        return True
    except Exception as e:
        logger.error(f"✗ Test 4 失败：{e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_agent_policy():
    """测试Task B - Agent的统一Policy分布"""
    logger.info("[Test 5] 验证Task B - 统一Policy分布...")
    
    try:
        from agent_ppo.agent import Agent
        from agent_ppo.conf.conf import Config
        
        config = Config()
        agent = Agent(config)
        
        # 检查agent有_build_masked_dist方法
        assert hasattr(agent, '_build_masked_dist'), "Missing _build_masked_dist"
        logger.info("  ✓ Agent._build_masked_dist() 存在")
        
        # 检查predict返回logprob
        obs = np.random.randn(1, config.OBS_DIM).astype(np.float32)
        legal = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        
        action, logprob = agent.predict(obs, legal)
        
        assert action.shape == (1,), f"action shape: {action.shape}"
        assert logprob.shape == (1,), f"logprob shape: {logprob.shape}"
        logger.info(f"  ✓ agent.predict() 返回 (action, logprob)")
        logger.info(f"    - action: {action[0]}")
        logger.info(f"    - logprob: {logprob[0]:.4f}")
        
        logger.info("✓ Test 5 通过：Policy分布统一\n")
        return True
    except Exception as e:
        logger.error(f"✗ Test 5 失败：{e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    logger.info("=" * 80)
    logger.info("第二轮代码修改综合集成验证")
    logger.info("=" * 80)
    
    tests = [
        ("模块导入", test_imports),
        ("Task A - 动作仿真器", test_action_simulator),
        ("Task A - action_quality buff参数", test_action_quality_with_buff),
        ("preprocessor集成", test_preprocessor_action_quality),
        ("Task B - Policy分布", test_agent_policy),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # 总结
    logger.info("=" * 80)
    logger.info("验证总结")
    logger.info("=" * 80)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        logger.info(f"  {status} {name}")
    
    logger.info("=" * 80)
    logger.info(f"总计：{passed}/{total} 验证通过")
    
    if passed == total:
        logger.info("\n🎉 所有验证通过！第二轮代码修改完成。\n")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} 项验证失败。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
