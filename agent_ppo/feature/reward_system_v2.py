#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Restructured Reward System based on 3-Layer Framework.
基于三层框架的重新设计的奖励系统。

框架架构：
═══════════════════════════════════════════════════════════════
  A. 主目标层 (Primary Objectives): 回答"整体对不对"
     - 生存奖励: 多活一步的基础价值
     - 终局成败: 最后关头的生死判决
     - 宝箱得分: 最终得分导向

  B. 引导层 (Guidance Layer): 回答"过程走向好不好"  
     - 距离成形: 离怪物更远和靠近怪物
     - 危险惩罚: 进入高危区域的代价
     - 资源获取: 主动获得buff的价值
     - 技能脱险: 用闪现成功脱险的收益

  C. 约束层 (Constraint Layer): 回答"行为质量高不高"
     - 低效移动: 原地蹭步、不动惩罚
     - 重复行为: 绕圈、来回横跳
     - 陷阱规避: 往死路钻、卡墙
     - 技能浪费: 无意义闪现、低价值技能使用
═══════════════════════════════════════════════════════════════
"""

import numpy as np

# 常量定义（避免从 preprocessor 导入，防止循环导入）
OPPOSITE_ACTION = {
    0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3,
    8: 12, 9: 13, 10: 14, 11: 15, 12: 8, 13: 9, 14: 10, 15: 11,
}

MAP_DIAG = float(np.sqrt(128.0**2 + 128.0**2))  # ~181
STUCK_WINDOW = 6


# 辅助函数（从 preprocessor 中复制，避免循环导入）
def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1]."""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _signed_norm(v, v_abs_max):
    """Normalize signed value to [-1, 1]."""
    return float(np.clip(float(v) / max(float(v_abs_max), 1e-6), -1.0, 1.0))


class RewardCalculator:
    """
    三层结构的奖励计算器
    
    Reward = A层 + B层 + C层
    """
    
    # ============================================================================
    # A. 主目标层 (主权重: 50-60% 的奖励范围)
    # ============================================================================
    
    @staticmethod
    def calculate_survival_objective(
        step_no,
        max_step,
        is_dead,
        treasures_collected,
        total_treasure,
    ):
        """
        A1. 生存主目标奖励
        
        原则:
          - 多活一步 = 逐步进步
          - 成功完成任务 = 明确胜利
          - 被抓住 = 明确失败
        
        返回: (生存奖励, 是否需要终止信号)
        """
        # 基础生存奖励: 每一步都有小的正奖励, 鼓励活下去
        # 值的设定: 想让模型重视生存，基础值可稍高，但不要压过宝箱
        step_normalization = step_no / max(max_step, 1)
        survive_base_reward = 0.02  # 每步基础 +0.02
        
        # 任务完成奖励: 活到最后一步
        completion_bonus = 0.0
        if step_no >= max_step:
            completion_bonus = 1.0  # 活到最后给大奖励
        
        # 死亡惩罚: 直接失败信号
        death_penalty = 0.0
        if is_dead:
            death_penalty = -10.0  # 强烈的失败信号
        
        survival_reward = survive_base_reward + completion_bonus + death_penalty
        
        return survival_reward, death_penalty < 0
    
    @staticmethod
    def calculate_score_objective(treasures_collected, total_treasure):
        """
        A2. 得分导向奖励
        
        原则:
          - 每获取一个宝箱 = 明显正奖励
          - 不是"完全依赖顺路碰到"，而是主动去追求
        
        直接对标赛题规则: 每个宝箱 = 100分
        为了让奖励在可比量级，将其转化为足够明显的反馈
        """
        # 每收集一个宝箱的奖励
        treasure_pickup_reward_per_item = 2.0  # 100分的比例映射
        
        # 总得分导向
        treasure_score_objective = treasure_pickup_reward_per_item * treasures_collected
        
        # 但加上一个"还有潜力"的惩罚，避免太早放弃
        # 如果收集进度很低，应该有压力去追求
        uncollected_ratio = 1.0 - (treasures_collected / max(total_treasure, 1))
        lazy_penalty = -0.05 * uncollected_ratio  # 没有充分追求得分，轻微惩罚
        
        score_objective = treasure_score_objective + lazy_penalty
        
        return score_objective
    
    # ============================================================================
    # B. 引导层 (引导权重: 25-30% 的奖励范围)
    # ============================================================================
    
    @staticmethod
    def calculate_distance_shaping(cur_min_dist_norm, last_min_dist_norm):
        """
        B1. 距离成形奖励
        
        原则:
          - 离怪物更远: 正奖励 (提高安全性)
          - 靠近怪物: 负奖励 (危险增加)
          - 但不要过度压制去拿宝箱的决策
        
        这一项是让模型学会"保持距离"，但不至于过度苟活。
        """
        # 距离变化的价值
        dist_improvement = cur_min_dist_norm - last_min_dist_norm
        
        # 分区间设定系数，避免线性太单调
        if dist_improvement > 0:
            # 远离怪物: 好事，但要递减
            distance_shaping = 0.15 * min(dist_improvement, 0.3)  # 最多鼓励 +0.045
        else:
            # 靠近怪物: 坏事，要有代价
            distance_shaping = 0.25 * max(dist_improvement, -0.2)  # 最多惩罚 -0.05
        
        return distance_shaping
    
    @staticmethod
    def calculate_danger_zone_penalty(cur_min_dist_norm, last_min_dist_norm):
        """
        B2. 危险区惩罚
        
        原则:
          - 进入高危区 (距离很近) = 明显惩罚
          - 持续处于高危区 = 累积惩罚
          - 但如果是为了拿宝箱，应该在得分奖励里补偿
        
        这一项独立于距离成形，专门针对"过于危险"的局面。
        """
        danger_zone_penalty = 0.0
        
        # 临界阈值: 0.15 表示怪物距离只有 15% 的地图对角线
        DANGER_ZONE_THRESHOLD = 0.15
        
        if cur_min_dist_norm < DANGER_ZONE_THRESHOLD:
            # 在危险区: 惩罚正比于"有多危险"
            danger_level = (DANGER_ZONE_THRESHOLD - cur_min_dist_norm) / DANGER_ZONE_THRESHOLD
            danger_zone_penalty = -0.08 * danger_level  # 最多 -0.08
        
        # 进一步恶化: 距离持续缩小
        if cur_min_dist_norm < last_min_dist_norm and cur_min_dist_norm < DANGER_ZONE_THRESHOLD:
            # 在危险区还在靠近，更坏
            danger_zone_penalty -= 0.05
        
        return danger_zone_penalty
    
    @staticmethod
    def calculate_buff_acquisition_bonus(collected_buff, total_buff, buff_remain, last_collected_buff):
        """
        B3. Buff 获取激励
        
        原则:
          - 获取 buff = 明显正反馈
          - Buff 在生效期内 = 持续小正反馈
          - 充分利用 buff 完成高风险任务 = 额外奖励
        
        这一层让模型理解: buff不是可选项，是战术资源。
        """
        buff_acquisition_bonus = 0.0
        
        # 获取 buff 时的奖励
        buff_delta = collected_buff - last_collected_buff
        if buff_delta > 0:
            buff_acquisition_bonus += 0.5  # 拿到 buff 是好事
        
        # Buff 在生效期间的持续激励
        if buff_remain > 0:
            # 有加速 buff 时，活着更有价值（可以跑得更快）
            buff_active_bonus = 0.02  # 每步 +0.02
            buff_acquisition_bonus += buff_active_bonus
        
        # 充分利用 buff 的额外奖励: 如果有 buff 并且在收集宝箱
        buff_utilization_ratio = collected_buff / max(total_buff, 1)
        if buff_remain > 0 and buff_utilization_ratio > 0.3:
            buff_acquisition_bonus += 0.03  # 在用 buff，这很好
        
        return buff_acquisition_bonus
    
    @staticmethod
    def calculate_flash_escape_bonus(
        flash_delta, 
        last_min_dist_norm, 
        cur_min_dist_norm,
        last_nearest_treasure_dist_norm,
        nearest_treasure_dist_norm,
    ):
        """
        B4. 闪现脱险奖励
        
        原则:
          - 在危险时用闪现逃脱 = 明显正奖励
          - 用闪现靠近宝箱（在安全范围内）= 中等奖励
          - 无意义闪现 = 惩罚 (在 C 层处理)
        
        这一项专注"有意义的闪现使用"。
        """
        flash_escape_bonus = 0.0
        
        if flash_delta <= 0:
            # 没有使用闪现，没有奖励
            return 0.0
        
        # 分类闪现的使用情况
        
        # 情况1: 紧急逃脱 (在危险时使用并成功拉开距离)
        emergency_escape = (
            last_min_dist_norm < 0.20 and 
            cur_min_dist_norm > last_min_dist_norm + 0.05
        )
        
        if emergency_escape:
            flash_escape_bonus += 0.6  # 很好的决定
            return flash_escape_bonus
        
        # 情况2: 战术抢宝 (相对安全的闪现靠近宝箱)
        tactical_treasure_flash = (
            last_nearest_treasure_dist_norm - nearest_treasure_dist_norm > 0.12 and
            cur_min_dist_norm > 0.25  # 仍然安全
        )
        
        if tactical_treasure_flash:
            flash_escape_bonus += 0.25  # 好的战术决策
            return flash_escape_bonus
        
        # 情况3: 其他闪现 (无法归类为上述两种，可能是无意义的)
        # 在 C 层处理惩罚
        
        return flash_escape_bonus
    
    # ============================================================================
    # C. 约束层 (约束权重: 10-15% 的奖励范围, 以惩罚为主)
    # ============================================================================
    
    @staticmethod
    def calculate_movement_quality_penalty(
        move_dist, 
        cur_min_dist_norm,
        recent_cells,
        hero_cell,
        visit_counter,
    ):
        """
        C1. 低效移动惩罚
        
        原则:
          - 完全不动或移动极少 = 时间浪费，惩罚
          - 但只在"相对安全"时惩罚（不要在全力逃命时还逼模型动）
        
        这一层防止: 原地蹭步数、卡在角落
        """
        movement_penalty = 0.0
        
        # 只在相对安全时检查移动
        SAFE_THRESHOLD = 0.22
        
        if cur_min_dist_norm > SAFE_THRESHOLD:
            # 相对安全的情况下
            
            if move_dist < 0.3:
                # 移动太少或不动
                movement_penalty -= 0.05  # 轻微惩罚，鼓励探索
            
            # 检查"卡在同一个地点"
            if hero_cell in visit_counter and visit_counter[hero_cell] > 5:
                # 这个地方访问太多次了，可能在蹭步数
                over_visit_count = visit_counter[hero_cell] - 5
                movement_penalty -= 0.03 * min(over_visit_count / 10, 0.5)  # 最多 -0.015
        
        return movement_penalty
    
    @staticmethod
    def calculate_loop_behavior_penalty(recent_cells, hero_cell, prev_action, last_action):
        """
        C2. 重复行为惩罚
        
        原则:
          - 绕圈 (访问相同区域) = 智障行为，惩罚
          - 来回横跳 (相反方向反复切换) = 徘徊，惩罚
          - 但动作不要太严格，给模型探索空间
        
        这一层防止: 来回摇晃、绕圈、无目的游走
        """
        loop_penalty = 0.0
        
        # 策略1: 绕圈检测 (最近6步访问的地点多样性)
        if len(recent_cells) >= 6:
            recent_6 = list(recent_cells)[-6:]
            unique_cells = len(set(recent_6))
            
            if unique_cells <= 2:
                # 只在 2 个地点反复, 明显在绕圈
                loop_penalty -= 0.08
            elif unique_cells <= 3:
                # 在 3 个地点反复，可能在绕圈
                loop_penalty -= 0.04
        
        # 策略2: 相反动作检测 (来回摇晃)
        if (prev_action in OPPOSITE_ACTION and 
            last_action in OPPOSITE_ACTION and
            OPPOSITE_ACTION[last_action] == prev_action):
            # 连续相反动作，来回摇晃
            loop_penalty -= 0.06
        
        return loop_penalty
    
    @staticmethod
    def calculate_wasteful_flash_penalty(
        flash_delta,
        last_min_dist_norm,
        cur_min_dist_norm,
        last_nearest_treasure_dist_norm,
        nearest_treasure_dist_norm,
    ):
        """
        C3. 无意义闪现惩罚
        
        原则:
          - 闪现后没有变安全，也没靠近宝箱 = 浪费技能
          - 闪现进入危险区 = 送死行为
        
        这一层防止: 乱交技能、低价值闪现
        
        注: 有意义的闪现在 B4 已奖励，这里只处理无意义的
        """
        wasteful_flash_penalty = 0.0
        
        if flash_delta <= 0:
            return 0.0
        
        # 判断闪现是否"有意义"
        
        # 情况1: 闪现后更危险 (靠近了怪物)
        flash_gets_closer = cur_min_dist_norm < last_min_dist_norm - 0.03
        if flash_gets_closer:
            wasteful_flash_penalty -= 0.25  # 送死行为，严厉惩罚
            return wasteful_flash_penalty
        
        # 情况2: 闪现没带来任何收益
        no_escape_benefit = cur_min_dist_norm <= last_min_dist_norm + 0.02
        no_treasure_benefit = (last_nearest_treasure_dist_norm - nearest_treasure_dist_norm) < 0.08
        
        if no_escape_benefit and no_treasure_benefit:
            # 不仅没逃脱，也没靠近宝箱，纯浪费
            wasteful_flash_penalty -= 0.12
        
        return wasteful_flash_penalty


def compute_reward_three_layer(
    # 环境信息
    step_no,
    max_step,
    is_dead,
    
    # 距离信息
    cur_min_monster_dist_norm,
    last_min_monster_dist_norm,
    nearest_treasure_dist_norm,
    last_nearest_treasure_dist_norm,
    
    # 收集信息
    treasures_collected,
    total_treasure,
    collected_buff,
    total_buff,
    buff_remain,
    flash_count,
    last_flash_count,
    
    # 行为信息
    move_dist,
    hero_cell,
    recent_cells,
    visit_counter,
    prev_action,
    last_action,
    
    # 可选的上一步的收集数据（用于计算delta）
    last_treasures_collected=None,
    last_collected_buff=None,
):
    """
    三层结构的总奖励计算
    
    Reward = A层 + B层 + C层
    
    各层的权重平衡:
      - A层 (主目标): 50-60%
      - B层 (引导): 25-30%
      - C层 (约束): 10-15%
    """
    
    calc = RewardCalculator()
    
    # 计算收集增量
    treasure_delta = treasures_collected - (last_treasures_collected if last_treasures_collected is not None else treasures_collected)
    buff_delta = collected_buff - (last_collected_buff if last_collected_buff is not None else collected_buff)
    
    # ===== A. 主目标层 =====
    survival_reward, should_terminate = calc.calculate_survival_objective(
        step_no, max_step, is_dead, treasures_collected, total_treasure
    )
    
    score_objective = calc.calculate_score_objective(treasures_collected, total_treasure)
    
    layer_a = survival_reward + score_objective
    
    # ===== B. 引导层 =====
    distance_shaping = calc.calculate_distance_shaping(
        cur_min_monster_dist_norm, last_min_monster_dist_norm
    )
    
    danger_zone_penalty = calc.calculate_danger_zone_penalty(
        cur_min_monster_dist_norm, last_min_monster_dist_norm
    )
    
    buff_bonus = calc.calculate_buff_acquisition_bonus(
        collected_buff, total_buff, buff_remain, last_collected_buff if last_collected_buff is not None else collected_buff
    )
    
    flash_escape_bonus = calc.calculate_flash_escape_bonus(
        flash_count - last_flash_count,
        last_min_monster_dist_norm,
        cur_min_monster_dist_norm,
        last_nearest_treasure_dist_norm,
        nearest_treasure_dist_norm,
    )
    
    layer_b = distance_shaping + danger_zone_penalty + buff_bonus + flash_escape_bonus
    
    # ===== C. 约束层 =====
    movement_penalty = calc.calculate_movement_quality_penalty(
        move_dist, cur_min_monster_dist_norm, recent_cells, hero_cell, visit_counter
    )
    
    loop_penalty = calc.calculate_loop_behavior_penalty(
        recent_cells, hero_cell, prev_action, last_action
    )
    
    wasteful_flash_penalty = calc.calculate_wasteful_flash_penalty(
        flash_count - last_flash_count,
        last_min_monster_dist_norm,
        cur_min_monster_dist_norm,
        last_nearest_treasure_dist_norm,
        nearest_treasure_dist_norm,
    )
    
    layer_c = movement_penalty + loop_penalty + wasteful_flash_penalty
    
    # ===== 总奖励 =====
    reward_total = layer_a + layer_b + layer_c
    
    # 裁剪到合理范围 (防止梯度爆炸)
    reward_total = float(np.clip(reward_total, -15.0, 15.0))
    
    # 详细的奖励拆分信息
    reward_details = {
        # A 层
        "survival_reward": float(survival_reward),
        "score_objective": float(score_objective),
        "layer_a": float(layer_a),
        
        # B 层
        "distance_shaping": float(distance_shaping),
        "danger_zone_penalty": float(danger_zone_penalty),
        "buff_bonus": float(buff_bonus),
        "flash_escape_bonus": float(flash_escape_bonus),
        "layer_b": float(layer_b),
        
        # C 层
        "movement_penalty": float(movement_penalty),
        "loop_penalty": float(loop_penalty),
        "wasteful_flash_penalty": float(wasteful_flash_penalty),
        "layer_c": float(layer_c),
        
        # 总和
        "reward_total": reward_total,
        "min_monster_dist_norm": float(cur_min_monster_dist_norm),
        "nearest_treasure_dist_norm": float(nearest_treasure_dist_norm),
    }
    
    return reward_total, reward_details
