#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Test script for new three-layer reward system.
验证新的三层奖励系统是否能正确执行。
"""

import numpy as np
from collections import Counter, deque
from reward_system_v2 import compute_reward_three_layer

def test_basic_reward_calculation():
    """测试基本的奖励计算"""
    print("=" * 60)
    print("测试 1: 基本奖励计算")
    print("=" * 60)
    
    # 模拟一个简单的游戏状态
    reward_total, reward_details = compute_reward_three_layer(
        # 环境信息
        step_no=100,
        max_step=1000,
        is_dead=False,
        
        # 距离信息
        cur_min_monster_dist_norm=0.5,  # 相对安全
        last_min_monster_dist_norm=0.45,  # 怪物距离增加了
        nearest_treasure_dist_norm=0.3,  # 宝藏很近
        last_nearest_treasure_dist_norm=0.35,  # 接近宝藏
        
        # 收集信息
        treasures_collected=3,
        total_treasure=15,
        collected_buff=1,
        total_buff=5,
        buff_remain=20.0,
        flash_count=2,
        last_flash_count=1,
        
        # 行为信息
        move_dist=1.5,
        hero_cell=(32, 32),
        recent_cells=deque([(32, 32), (32, 31), (31, 31)], maxlen=12),
        visit_counter=Counter({(32, 32): 10, (32, 31): 5, (31, 31): 3}),
        prev_action=0,
        last_action=1,
        
        # 上一步的收集数据
        last_treasures_collected=2,  # 刚收集了 1 个
        last_collected_buff=0,  # 刚收集了 1 个buff
    )
    
    print(f"Total Reward: {reward_total:.4f}")
    print(f"\n三层奖励拆分:")
    print(f"  Layer A (主目标): {reward_details['layer_a']:.4f}")
    print(f"    - Survival: {reward_details['survival_reward']:.4f}")
    print(f"    - Score: {reward_details['score_objective']:.4f}")
    print(f"  Layer B (引导): {reward_details['layer_b']:.4f}")
    print(f"    - Distance: {reward_details['distance_shaping']:.4f}")
    print(f"    - Danger: {reward_details['danger_zone_penalty']:.4f}")
    print(f"    - Buff: {reward_details['buff_bonus']:.4f}")
    print(f"    - Flash: {reward_details['flash_escape_bonus']:.4f}")
    print(f"  Layer C (约束): {reward_details['layer_c']:.4f}")
    print(f"    - Movement: {reward_details['movement_penalty']:.4f}")
    print(f"    - Loop: {reward_details['loop_penalty']:.4f}")
    print(f"    - Wasteful Flash: {reward_details['wasteful_flash_penalty']:.4f}")
    
    assert -15.0 <= reward_total <= 15.0, f"奖励应该在 [-15, 15] 范围内，但得到 {reward_total}"
    print("\n✓ 测试通过: 奖励值在合理范围内")


def test_death_penalty():
    """测试死亡惩罚"""
    print("\n" + "=" * 60)
    print("测试 2: 死亡惩罚")
    print("=" * 60)
    
    reward_dead, details_dead = compute_reward_three_layer(
        step_no=100,
        max_step=1000,
        is_dead=True,
        
        cur_min_monster_dist_norm=0.0,
        last_min_monster_dist_norm=0.1,
        nearest_treasure_dist_norm=0.5,
        last_nearest_treasure_dist_norm=0.5,
        
        treasures_collected=2,
        total_treasure=15,
        collected_buff=0,
        total_buff=5,
        buff_remain=0.0,
        flash_count=0,
        last_flash_count=0,
        
        move_dist=0.0,
        hero_cell=(32, 32),
        recent_cells=deque([(32, 32)], maxlen=12),
        visit_counter=Counter({(32, 32): 1}),
        prev_action=0,
        last_action=0,
        
        last_treasures_collected=2,
        last_collected_buff=0,
    )
    
    print(f"死亡时的奖励: {reward_dead:.4f}")
    print(f"存活奖励: {details_dead['survival_reward']:.4f}")
    assert reward_dead < 0, f"死亡应该导致负奖励，但得到 {reward_dead}"
    assert details_dead['survival_reward'] < 0, "死亡时 survival_reward 应该是负的"
    print("✓ 测试通过: 死亡惩罚正确")


def test_treasure_collection():
    """测试宝藏收集奖励"""
    print("\n" + "=" * 60)
    print("测试 3: 宝藏收集")
    print("=" * 60)
    
    reward_with_treasure, details = compute_reward_three_layer(
        step_no=100,
        max_step=1000,
        is_dead=False,
        
        cur_min_monster_dist_norm=0.5,
        last_min_monster_dist_norm=0.5,
        nearest_treasure_dist_norm=0.5,
        last_nearest_treasure_dist_norm=0.5,
        
        treasures_collected=5,  # 刚收集了多个
        total_treasure=15,
        collected_buff=0,
        total_buff=5,
        buff_remain=0.0,
        flash_count=0,
        last_flash_count=0,
        
        move_dist=1.0,
        hero_cell=(32, 32),
        recent_cells=deque([(32, 32)], maxlen=12),
        visit_counter=Counter({(32, 32): 1}),
        prev_action=0,
        last_action=0,
        
        last_treasures_collected=3,  # 这次收集了 2 个
        last_collected_buff=0,
    )
    
    print(f"收集宝藏的总奖励: {reward_with_treasure:.4f}")
    print(f"分数目标奖励: {details['score_objective']:.4f}")
    assert reward_with_treasure > 0, "收集宝藏应该产生正奖励"
    print("✓ 测试通过: 宝藏收集奖励正确")


def test_flash_escape():
    """测试闪现逃脱奖励"""
    print("\n" + "=" * 60)
    print("测试 4: 闪现逃脱")
    print("=" * 60)
    
    # 危险情况下的成功闪现
    reward_emergency_flash, details = compute_reward_three_layer(
        step_no=100,
        max_step=1000,
        is_dead=False,
        
        cur_min_monster_dist_norm=0.4,  # 闪现后距离增加
        last_min_monster_dist_norm=0.15,  # 之前在危险区域
        nearest_treasure_dist_norm=0.5,
        last_nearest_treasure_dist_norm=0.5,
        
        treasures_collected=2,
        total_treasure=15,
        collected_buff=0,
        total_buff=5,
        buff_remain=0.0,
        flash_count=3,  # 刚闪现
        last_flash_count=2,
        
        move_dist=0.5,
        hero_cell=(40, 40),
        recent_cells=deque([(32, 32), (35, 35), (40, 40)], maxlen=12),
        visit_counter=Counter({}),
        prev_action=0,
        last_action=8,  # 闪现方向
        
        last_treasures_collected=2,
        last_collected_buff=0,
    )
    
    print(f"紧急闪现的奖励: {reward_emergency_flash:.4f}")
    print(f"闪现逃脱奖励: {details['flash_escape_bonus']:.4f}")
    assert details['flash_escape_bonus'] > 0, "紧急情况下的成功闪现应该有正奖励"
    print("✓ 测试通过: 闪现逃脱奖励正确")


def test_waypoint_completion():
    """测试完成目标时的奖励"""
    print("\n" + "=" * 60)
    print("测试 5: 完成目标")
    print("=" * 60)
    
    reward_complete, details = compute_reward_three_layer(
        step_no=800,
        max_step=1000,
        is_dead=False,  # 仍然活着
        
        cur_min_monster_dist_norm=0.3,
        last_min_monster_dist_norm=0.3,
        nearest_treasure_dist_norm=0.5,
        last_nearest_treasure_dist_norm=0.5,
        
        treasures_collected=15,  # 收集了所有宝藏
        total_treasure=15,
        collected_buff=0,
        total_buff=5,
        buff_remain=0.0,
        flash_count=0,
        last_flash_count=0,
        
        move_dist=1.0,
        hero_cell=(32, 32),
        recent_cells=deque([(32, 32)], maxlen=12),
        visit_counter=Counter({(32, 32): 1}),
        prev_action=0,
        last_action=0,
        
        last_treasures_collected=15,
        last_collected_buff=0,
    )
    
    print(f"完成目标的奖励: {reward_complete:.4f}")
    print(f"分数目标: {details['score_objective']:.4f}")
    print(f"存活奖励: {details['survival_reward']:.4f}")
    # 完成目标应该有额外的高奖励
    assert reward_complete > 5.0, "完成所有目标应该有较高奖励"
    print("✓ 测试通过: 完成目标奖励正确")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("三层奖励系统测试套件")
    print("=" * 60)
    
    try:
        test_basic_reward_calculation()
        test_death_penalty()
        test_treasure_collection()
        test_flash_escape()
        test_waypoint_completion()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
