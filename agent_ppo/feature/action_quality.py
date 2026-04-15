#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Action quality assessment and evaluation module.
动作质量评估模块。

该模块评估每个候选动作的质量，提供给模型一个"中间表示"层，帮助模型理解：
  - 哪个动作更安全（离怪物更远）
  - 哪个动作更容易拿宝箱（靠近宝箱）
  - 哪个动作更容易进死路（方向是否开阔）
  - 哪个闪现动作是高价值脱险
  - 哪个动作虽然合法，但物理后果很差

特别说明：
  - 这是"评估特征"，不是最终动作选择器
  - 不改 PPO 输出逻辑，不改环境 legal_action
  - 只是把"动作后果"显式告诉模型，让模型更好地学习
  
输出格式：
  - 16 个动作 × 6 维度 = 96 维动作规划特征
  - 6 个维度: safety / treasure / buff / terrain / revisit / flash_value
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from agent_ppo.feature.spatial_utils import (
    action_to_delta,
    apply_action_to_pos,
    relative_pos,
    relative_pos_normalized,
    get_direction_ahead_in_local_map,
    get_action_name,
    action_is_flash,
)


# ============================================================================
# 常数定义
# ============================================================================

MAP_SIZE = 128.0
MAX_DIST = float(np.sqrt(MAP_SIZE**2 + MAP_SIZE**2))
TREASURE_DIST_THRESHOLD = 15.0  # 宝箱距离阈值
BUFF_DIST_THRESHOLD = 15.0       # buff 距离阈值
DANGER_DIST_THRESHOLD = 10.0     # 危险距离阈值

# 从动作组织到特征的维度顺序
ACTION_QUALITY_FEATURE_DIMS = {
    "safety_score": 1,          # 安全得分 [0, 1]
    "treasure_gain_score": 1,   # 宝箱收益分 [0, 1]
    "buff_gain_score": 1,       # buff 收益分 [0, 1]
    "terrain_score": 1,         # 地形开阔分 [0, 1]
    "revisit_penalty": 1,       # 重复访问惩罚 [0, 1]
    "flash_value_score": 1,     # 闪现价值分 [0, 1]（仅对闪现有意义）
}
TOTAL_DIMENSIONS_PER_ACTION = sum(ACTION_QUALITY_FEATURE_DIMS.values())  # 6


class ActionQualityEvaluator:
    """
    动作质量评估器
    
    职责：
      - 评估 16 个动作中每个动作的"品质"
      - 不做硬 mask，只提供"建议"信息
      - 输出 16*6=96 维的动作规划特征
    """
    
    def __init__(self):
        self.safety_weight = 2.0
        self.treasure_weight = 2.0
        self.buff_weight = 1.0
        self.terrain_weight = 1.0
        self.revisit_weight = 0.5
        self.flash_value_weight = 1.5
    
    def evaluate_all_actions(
        self,
        hero_pos: Dict[str, float],
        hero_hp: float,
        hero_max_hp: float,
        monsters: List[Dict],
        treasures: List[Dict],
        buffs: List[Dict],
        legal_action_mask: np.ndarray,
        local_map: Optional[np.ndarray] = None,
        recent_positions: Optional[List[Dict]] = None,
        flash_cooldown: int = 0,
        danger_trend: float = 0.0,
    ) -> np.ndarray:
        """
        评估所有 16 个动作的质量。
        
        Args:
            hero_pos: 英雄位置 {"x": float, "z": float}
            hero_hp: 英雄当前血量
            hero_max_hp: 英雄最大血量
            monsters: 怪物列表，每个元素是 {"pos": {"x", "z"}, "speed": float, ...}
            treasures: 宝箱列表
            buffs: buff 列表
            legal_action_mask: 16D 合法动作掩码 [0 或 1]
            local_map: 局部地图 (21x21)，可选
            recent_positions: 最近访问过的位置队列，用于检查重复
            flash_cooldown: 闪现冷却剩余步数
            danger_trend: 危险趋势指标 [-1, 1]，负数表示局势恶化
        
        Returns:
            (16, 6) 的 numpy 数组，表示 16 个动作的 6 维质量评估
            - 每行是一个动作，6 列分别表示 6 个维度
            - 返回后会被 flatten 成 96D 特征，送入到模型
        """
        action_qualities = np.zeros((16, TOTAL_DIMENSIONS_PER_ACTION), dtype=np.float32)
        
        for action_id in range(16):
            # 计算这个动作执行后的后果
            action_qualities[action_id] = self._evaluate_single_action(
                action_id=action_id,
                hero_pos=hero_pos,
                hero_hp=hero_hp,
                hero_max_hp=hero_max_hp,
                monsters=monsters,
                treasures=treasures,
                buffs=buffs,
                legal_action=legal_action_mask[action_id] if legal_action_mask is not None else 1.0,
                local_map=local_map,
                recent_positions=recent_positions,
                flash_cooldown=flash_cooldown,
                danger_trend=danger_trend,
            )
        
        return action_qualities
    
    def _evaluate_single_action(
        self,
        action_id: int,
        hero_pos: Dict[str, float],
        hero_hp: float,
        hero_max_hp: float,
        monsters: List[Dict],
        treasures: List[Dict],
        buffs: List[Dict],
        legal_action: float,
        local_map: Optional[np.ndarray],
        recent_positions: Optional[List[Dict]],
        flash_cooldown: int,
        danger_trend: float,
    ) -> np.ndarray:
        """
        评估单个动作的 6 个维度。
        
        返回 [safety_score, treasure_gain, buff_gain, terrain, revisit_penalty, flash_value]
        """
        # 如果不合法，该动作所有维度得分都偏低
        legal_factor = float(legal_action)
        
        # 预测执行此动作后的新位置
        next_pos = apply_action_to_pos(hero_pos, action_id, MAP_SIZE)
        
        # 四个维度的评分
        safety_score = self._compute_safety_score(
            next_pos, monsters, hero_hp, hero_max_hp, danger_trend, legal_factor
        )
        
        treasure_gain_score = self._compute_treasure_gain_score(
            hero_pos, next_pos, treasures, legal_factor
        )
        
        buff_gain_score = self._compute_buff_gain_score(
            hero_pos, next_pos, buffs, legal_factor
        )
        
        terrain_score = self._compute_terrain_score(
            action_id, next_pos, local_map, legal_factor
        )
        
        revisit_penalty = self._compute_revisit_penalty(
            next_pos, recent_positions, legal_factor
        )
        
        flash_value_score = self._compute_flash_value_score(
            action_id, hero_pos, next_pos, monsters, hero_hp, hero_max_hp,
            flash_cooldown, danger_trend, legal_factor
        )
        
        return np.array([
            safety_score,
            treasure_gain_score,
            buff_gain_score,
            terrain_score,
            revisit_penalty,
            flash_value_score,
        ], dtype=np.float32)
    
    # ========================================================================
    # 各个维度的评分函数
    # ========================================================================
    
    def _compute_safety_score(
        self,
        next_pos: Dict[str, float],
        monsters: List[Dict],
        hero_hp: float,
        hero_max_hp: float,
        danger_trend: float,
        legal_factor: float,
    ) -> float:
        """
        安全得分：执行动作后，离怪物是否更远，是否进入危险区。
        
        返回 [0, 1]，高分表示安全。
        """
        if not monsters:
            # 没有怪物，该动作很安全
            return float(legal_factor)
        
        # 计算到最近怪物的距离
        min_dist_to_monster = float('inf')
        for monster in monsters:
            monster_pos = monster.get("pos", {})
            dx = float(monster_pos.get("x", 0)) - float(next_pos.get("x", 0))
            dz = float(monster_pos.get("z", 0)) - float(next_pos.get("z", 0))
            dist = float(np.sqrt(dx**2 + dz**2))
            min_dist_to_monster = min(min_dist_to_monster, dist)
        
        # 距离越远越安全
        if min_dist_to_monster > DANGER_DIST_THRESHOLD:
            dist_safety = 1.0
        elif min_dist_to_monster < 5.0:
            dist_safety = 0.0  # 非常危险
        else:
            # 线性插值
            dist_safety = (min_dist_to_monster - 5.0) / (DANGER_DIST_THRESHOLD - 5.0)
        
        # 健康状态调整：血量越少，安全需求越高
        hp_ratio = max(0.0, hero_hp / max(hero_max_hp, 1.0))
        health_factor = 0.7 + 0.3 * hp_ratio  # [0.7, 1.0]
        
        # 危险趋势调整：局势恶化时，提升对安全的需求
        trend_factor = 1.0 - 0.3 * max(0.0, danger_trend)  # danger_trend > 0 时降低得分
        
        safety_score = dist_safety * health_factor * trend_factor * legal_factor
        return float(np.clip(safety_score, 0.0, 1.0))
    
    def _compute_treasure_gain_score(
        self,
        hero_pos: Dict[str, float],
        next_pos: Dict[str, float],
        treasures: List[Dict],
        legal_factor: float,
    ) -> float:
        """
        宝箱收益分：执行动作后，是否更接近宝箱。
        
        返回 [0, 1]，高分表示靠近宝箱。
        """
        if not treasures:
            return 0.0  # 没有宝箱，不算收益
        
        # 计算到最近宝箱的距离变化
        min_dist_before = float('inf')
        min_dist_after = float('inf')
        
        for treasure in treasures:
            treasure_pos = treasure.get("pos", {})
            
            # 执行前距离
            dx_before = float(treasure_pos.get("x", 0)) - float(hero_pos.get("x", 0))
            dz_before = float(treasure_pos.get("z", 0)) - float(hero_pos.get("z", 0))
            dist_before = float(np.sqrt(dx_before**2 + dz_before**2))
            min_dist_before = min(min_dist_before, dist_before)
            
            # 执行后距离
            dx_after = float(treasure_pos.get("x", 0)) - float(next_pos.get("x", 0))
            dz_after = float(treasure_pos.get("z", 0)) - float(next_pos.get("z", 0))
            dist_after = float(np.sqrt(dx_after**2 + dz_after**2))
            min_dist_after = min(min_dist_after, dist_after)
        
        # 距离减少（靠近宝箱）得正分
        distance_improvement = min_dist_before - min_dist_after
        
        if distance_improvement > 0:
            # 成功靠近
            gain_score = min(distance_improvement / 10.0, 1.0)
        else:
            # 没有靠近或远离
            gain_score = 0.0
        
        return float(np.clip(gain_score * legal_factor, 0.0, 1.0))
    
    def _compute_buff_gain_score(
        self,
        hero_pos: Dict[str, float],
        next_pos: Dict[str, float],
        buffs: List[Dict],
        legal_factor: float,
    ) -> float:
        """
        buff 收益分：执行动作后，是否更接近 buff。
        
        返回 [0, 1]，高分表示靠近 buff。
        """
        if not buffs:
            return 0.0
        
        # 类似宝箱的逻辑
        min_dist_before = float('inf')
        min_dist_after = float('inf')
        
        for buff in buffs:
            buff_pos = buff.get("pos", {})
            
            dx_before = float(buff_pos.get("x", 0)) - float(hero_pos.get("x", 0))
            dz_before = float(buff_pos.get("z", 0)) - float(hero_pos.get("z", 0))
            dist_before = float(np.sqrt(dx_before**2 + dz_before**2))
            min_dist_before = min(min_dist_before, dist_before)
            
            dx_after = float(buff_pos.get("x", 0)) - float(next_pos.get("x", 0))
            dz_after = float(buff_pos.get("z", 0)) - float(next_pos.get("z", 0))
            dist_after = float(np.sqrt(dx_after**2 + dz_after**2))
            min_dist_after = min(min_dist_after, dist_after)
        
        distance_improvement = min_dist_before - min_dist_after
        
        if distance_improvement > 0:
            gain_score = min(distance_improvement / 10.0, 1.0)
        else:
            gain_score = 0.0
        
        return float(np.clip(gain_score * legal_factor, 0.0, 1.0))
    
    def _compute_terrain_score(
        self,
        action_id: int,
        next_pos: Dict[str, float],
        local_map: Optional[np.ndarray],
        legal_factor: float,
    ) -> float:
        """
        地形开阔分：该方向是开阔还是容易卡住，判断方向的前进潜力。
        
        返回 [0, 1]，高分表示地形开阔。
        """
        # 基础地形评分（从 spatial_utils 获取）
        direction_info = get_direction_ahead_in_local_map(action_id, local_map)
        openness = float(direction_info.get("openness", 0.5))
        
        # 位置越接近边界，通过性可能越低
        x = float(next_pos.get("x", MAP_SIZE / 2))
        z = float(next_pos.get("z", MAP_SIZE / 2))
        
        # 距离最近边界的距离
        min_dist_to_boundary = min(x, z, MAP_SIZE - x, MAP_SIZE - z)
        boundary_penalty = 1.0 - max(0.0, (10.0 - min_dist_to_boundary) / 10.0)
        
        terrain_score = openness * boundary_penalty * legal_factor
        return float(np.clip(terrain_score, 0.0, 1.0))
    
    def _compute_revisit_penalty(
        self,
        next_pos: Dict[str, float],
        recent_positions: Optional[List[Dict]],
        legal_factor: float,
    ) -> float:
        """
        重复访问惩罚：执行动作是否会导致进入最近访问过的区域（绕圈）。
        
        返回 [0, 1]，高分表示不绕圈。
        """
        if not recent_positions or len(recent_positions) == 0:
            return float(legal_factor)  # 没有历史记录，视为健康
        
        # 检查新位置是否与最近位置太接近
        x_next = float(next_pos.get("x", 0))
        z_next = float(next_pos.get("z", 0))
        
        revisit_count = 0
        for pos in recent_positions[-6:]:  # 检查最近 6 步
            if pos is None:
                continue
            x_prev = float(pos.get("x", 0))
            z_prev = float(pos.get("z", 0))
            dist = float(np.sqrt((x_next - x_prev)**2 + (z_next - z_prev)**2))
            if dist < 5.0:  # 太接近
                revisit_count += 1
        
        # 基于重复次数计算惩罚
        if revisit_count == 0:
            return float(legal_factor)
        else:
            penalty = max(0.0, 1.0 - 0.3 * revisit_count)
            return float(penalty * legal_factor)
    
    def _compute_flash_value_score(
        self,
        action_id: int,
        hero_pos: Dict[str, float],
        next_pos: Dict[str, float],
        monsters: List[Dict],
        hero_hp: float,
        hero_max_hp: float,
        flash_cooldown: int,
        danger_trend: float,
        legal_factor: float,
    ) -> float:
        """
        闪现价值分：仅对闪现动作有意义。
        
        评估：
          - 这是"脱险型"闪现还是"浪费型"闪现
          - 是否在高危时刻执行
          - 是否能有效改善局势
        
        返回 [0, 1]，高分表示闪现有价值。
        """
        if not action_is_flash(action_id):
            # 不是闪现，返回中立分
            return 0.5 * legal_factor
        
        # 闪现冷却检查
        if flash_cooldown > 0:
            # 闪现还在冷却，这个动作实际上不能用
            return 0.0
        
        if not monsters or hero_hp <= 0:
            # 没有怪物或血量为 0，闪现意义不大
            return 0.1 * legal_factor
        
        # 计算闪现前后到怪物的距离变化
        min_dist_before = float('inf')
        min_dist_after = float('inf')
        
        for monster in monsters:
            monster_pos = monster.get("pos", {})
            
            dx_before = float(monster_pos.get("x", 0)) - float(hero_pos.get("x", 0))
            dz_before = float(monster_pos.get("z", 0)) - float(hero_pos.get("z", 0))
            dist_before = float(np.sqrt(dx_before**2 + dz_before**2))
            min_dist_before = min(min_dist_before, dist_before)
            
            dx_after = float(monster_pos.get("x", 0)) - float(next_pos.get("x", 0))
            dz_after = float(monster_pos.get("z", 0)) - float(next_pos.get("z", 0))
            dist_after = float(np.sqrt(dx_after**2 + dz_after**2))
            min_dist_after = min(min_dist_after, dist_after)
        
        distance_improvement = min_dist_after - min_dist_before  # 正值表示远离怪物
        
        # 危机程度：血量比和怪物距离
        hp_ratio = hero_hp / max(hero_max_hp, 1.0)
        in_crisis = hp_ratio < 0.5 or min_dist_before < 15.0
        
        if distance_improvement > 5.0:
            # 闪现成功脱险
            flash_value = 0.8
        elif distance_improvement > 0:
            # 闪现有所改善
            flash_value = 0.5
        else:
            # 闪现没有改善，甚至恶化
            flash_value = 0.1
        
        # 在危机时闪现更有价值
        if in_crisis:
            flash_value = min(1.0, flash_value * 1.5)
        
        return float(np.clip(flash_value * legal_factor, 0.0, 1.0))


# ============================================================================
# 导出函数（用于预处理流程）
# ============================================================================

def compute_action_quality_features(
    hero_pos: Dict[str, float],
    hero_hp: float,
    hero_max_hp: float,
    monsters: List[Dict],
    treasures: List[Dict],
    buffs: List[Dict],
    legal_action_mask: np.ndarray,
    local_map: Optional[np.ndarray] = None,
    recent_positions: Optional[List[Dict]] = None,
    flash_cooldown: int = 0,
    danger_trend: float = 0.0,
) -> np.ndarray:
    """
    主导出函数：计算动作质量评估特征。
    
    返回 96D 的特征向量（16 动作 × 6 维）。
    """
    evaluator = ActionQualityEvaluator()
    
    # 返回 (16, 6) 的矩阵
    action_qualities = evaluator.evaluate_all_actions(
        hero_pos=hero_pos,
        hero_hp=hero_hp,
        hero_max_hp=hero_max_hp,
        monsters=monsters,
        treasures=treasures,
        buffs=buffs,
        legal_action_mask=legal_action_mask,
        local_map=local_map,
        recent_positions=recent_positions,
        flash_cooldown=flash_cooldown,
        danger_trend=danger_trend,
    )
    
    # Flatten 成 96D
    return action_qualities.flatten()
