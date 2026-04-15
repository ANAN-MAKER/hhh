#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
测试新的5维特征系统 - 验证特征维度和合理性

Test the new 5-category feature engineering system.
"""

import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.preprocessor import Preprocessor


def test_feature_dimensions():
    """验证特征维度是否正确."""
    
    print("=" * 70)
    print("特征系统维度验证 (Feature Dimension Validation)")
    print("=" * 70)
    
    expected_features = [
        ("A. 自身状态特征", 24),
        ("B. 外部实体特征", 34),
        ("C. 地图与路径特征", 35),
        ("D. 时序与记忆特征", 30),
        ("E. 决策辅助特征", 20),
        ("F. 合法动作掩码", 16),
    ]
    
    total_expected = sum(d for _, d in expected_features)
    
    print("\n预期的特征维度:")
    print("-" * 70)
    for name, dim in expected_features:
        print(f"  {name:<30} {dim:3d}D")
    print("-" * 70)
    print(f"  {'总计':<30} {total_expected:3d}D")
    
    actual_total = Config.DIM_OF_OBSERVATION
    print(f"\n配置中的实际维度: {actual_total}D")
    
    if actual_total == total_expected:
        print(f"✅ 维度匹配! ({actual_total}D == {total_expected}D)")
    else:
        print(f"❌ 维度不匹配! ({actual_total}D != {total_expected}D)")
        return False
    
    # 验证Config.FEATURES
    print("\nConfig.FEATURES 详情:")
    print("-" * 70)
    for i, (name, expected) in enumerate(expected_features):
        if i < len(Config.FEATURES):
            actual = Config.FEATURES[i]
            match = "✅" if actual == expected else "❌"
            print(f"  [{match}] {name:<30} 期望:{expected:3d}D 实际:{actual:3d}D")
        else:
            print(f"  [❌] {name:<30} 期望:{expected:3d}D 实际:缺失")
    
    return True


def test_preprocessor_initialization():
    """测试预处理器初始化."""
    
    print("\n" + "=" * 70)
    print("预处理器初始化测试 (Preprocessor Initialization)")
    print("=" * 70)
    
    try:
        preprocessor = Preprocessor()
        print("✅ 预处理器初始化成功")
        
        # 检查各种时序记忆
        checks = [
            ("trajectory_history", preprocessor.trajectory_history),
            ("monster_dist_history", preprocessor.monster_dist_history),
            ("treasure_dist_history", preprocessor.treasure_dist_history),
            ("reward_history", preprocessor.reward_history),
            ("danger_trend", preprocessor.danger_trend),
            ("exploration_history", preprocessor.exploration_history),
        ]
        
        print("\n时序记忆结构检查:")
        for name, attr in checks:
            print(f"  ✅ {name:<30} 初始化成功")
        
        return True
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False


def test_feature_building():
    """测试特征构建方法是否存在."""
    
    print("\n" + "=" * 70)
    print("特征构建方法存在性检查 (Feature Building Methods)")
    print("=" * 70)
    
    preprocessor = Preprocessor()
    
    methods = [
        ("A类", "_build_hero_enhanced_features"),
        ("B类", "_build_external_entities_enhanced"),
        ("C类", "_build_map_and_paths_enhanced"),
        ("D类", "_build_temporal_memory_features"),
        ("E类", "_build_decision_auxiliary_features"),
    ]
    
    all_good = True
    for category, method_name in methods:
        if hasattr(preprocessor, method_name):
            method = getattr(preprocessor, method_name)
            if callable(method):
                print(f"  ✅ {category:<10} {method_name:<40} 存在且可调用")
            else:
                print(f"  ❌ {category:<10} {method_name:<40} 不可调用")
                all_good = False
        else:
            print(f"  ❌ {category:<10} {method_name:<40} 不存在")
            all_good = False
    
    return all_good


def print_feature_structure():
    """打印特征结构详细说明."""
    
    print("\n" + "=" * 70)
    print("新的5维特征系统结构 (5-Category Feature Engineering)")
    print("=" * 70)
    
    structure = {
        "A. 自身状态特征 (24D)": {
            "描述": "英雄自身与进度",
            "子项": [
                "位置特征 (2D): 英雄X、Z坐标",
                "速度特征 (3D): 当前速度、加速状态、加速持续时间",
                "技能状态 (4D): 闪现可用性、冷却进度、即将可用、可用度",
                "进度特征 (7D): 步数进度、剩余时间、宝箱进度、增益进度等",
                "其他状态 (8D): 收集数量、生存状态、周期性标志等",
            ]
        },
        "B. 外部实体特征 (34D)": {
            "描述": "游戏中的动态对象",
            "子项": [
                "怪物特征 (14D): 2个怪物各7维 (存在、距离、速度、方向、威胁评分)",
                "宝箱特征 (12D): 3个最近宝箱各4维 (距离、优先级、方向)",
                "Buff特征 (8D): 最近增益的详细信息",
            ]
        },
        "C. 地图与路径特征 (35D)": {
            "描述": "环境与可达性",
            "子项": [
                "局部地图 (25D): 5×5网格障碍信息",
                "方向质量 (8D): 8个移动方向的安全性与收益评分",
                "逃生通路 (2D): 最安全方向、最佳财富方向",
            ]
        },
        "D. 时序与记忆特征 (30D)": {
            "描述": "历史模式与趋势",
            "子项": [
                "历史轨迹 (3D): 轨迹变化、重复因素、探索比率",
                "危险趋势 (4D): 怪物距离走势、最小值、最大值、波动度",
                "收益趋势 (3D): 奖励走势、平均奖励、最大奖励",
                "探索记忆 (5D): 访问区域数、高频访问、覆盖度、多样性",
                "动作历史 (5D): 最近动作模式、多样性",
                "其他趋势 (5D): 宝箱距离走势、波动性等",
            ]
        },
        "E. 决策辅助特征 (20D)": {
            "描述": "决策支持指标",
            "子项": [
                "风险评分 (4D): 当前危险度、接近度、趋势、逃脱难度",
                "收益评分 (4D): 宝箱可达性、增益可达性、收益风险比、总收益",
                "闪现专项评分 (4D): 必要性、收益、安全性、可用状态",
                "综合建议 (8D): 推荐动作 (追宝、寻增益、紧急逃脱、撤退)、置信度",
            ]
        },
        "F. 合法动作掩码 (16D)": {
            "描述": "可执行动作的指示",
            "子项": [
                "16维向量: 1表示该动作合法，0表示非法",
            ]
        }
    }
    
    for category, info in structure.items():
        print(f"\n{category}")
        print(f"  说明: {info['描述']}")
        print(f"  组成:")
        for sub in info['子项']:
            print(f"    • {sub}")


if __name__ == "__main__":
    success = True
    
    # 1. 维度检查
    success &= test_feature_dimensions()
    
    # 2. 初始化检查
    success &= test_preprocessor_initialization()
    
    # 3. 方法检查
    success &= test_feature_building()
    
    # 4. 结构说明
    print_feature_structure()
    
    # 总结
    print("\n" + "=" * 70)
    if success:
        print("✅ 所有验证通过! 新的特征系统已准备就绪。")
    else:
        print("❌ 部分验证失败，请检查上述错误。")
    print("=" * 70)
