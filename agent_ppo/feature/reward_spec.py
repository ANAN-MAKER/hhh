#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Unified Reward System Specification and Monitoring Framework.
统一的奖励体系规范与监控框架。

该文件定义了奖励计算的标准输出格式和日志监控口径。
确保奖励设计（三层结构）和训练监控保持一致的语义理解。

═══════════════════════════════════════════════════════════════════════════════
三层奖励结构 (A/B/C Layer) 说明
═══════════════════════════════════════════════════════════════════════════════

A 层 - 主目标层 (Primary Objectives, 50-60% 权重)
  职责: 回答"整体对不对" - 这个情节是否成功、得分是否提高
  组成:
    - survival_reward      : 生存基础奖励 (每步+0.02) + 完成奖励 (活到最后+1.0)
    - death_penalty        : 死亡惩罚 (-10.0)
    - score_reward         : 宝箱得分奖励 (每个+2.0) + 懒惰惩罚 (-0.05*uncollected_ratio)
  
  关键指标:
    - layer_a_total        : A 层总奖励 = survival_reward + score_reward

B 层 - 引导层 (Guidance Layer, 25-30% 权重)
  职责: 回答"过程走向好不好" - 是否在学习正确的行为模式
  组成:
    - dist_shaping         : 距离成形 (靠近+0.15, 远离-0.05)
    - danger_penalty       : 危险区惩罚 (在危险区-0.08)
    - buff_reward          : buff 获取与使用 (+0.5 获取, +0.02 使用)
    - flash_reward         : 闪现脱险 (+0.6 紧急脱险, +0.25 战术抢宝)
  
  关键指标:
    - layer_b_total        : B 层总奖励 = dist_shaping + danger_penalty + buff_reward + flash_reward

C 层 - 约束层 (Constraint Layer, 10-15% 权重，主要惩罚)
  职责: 回答"行为质量高不高" - 是否避免了低效、危险的行为
  组成:
    - movement_penalty     : 低效移动 (不动-0.05, 原地蹭-0.02)
    - loiter_penalty       : 绕圈惩罚 (重复区域-0.08 per step)
    - invalid_style_penalty: 陷阱规避 (往死路钻-0.10)
    - wasteful_flash_penalty: 技能浪费 (闪现无意义-0.15)
  
  关键指标:
    - layer_c_total        : C 层总奖励 = movement_penalty + loiter_penalty + invalid_style_penalty + wasteful_flash_penalty

═══════════════════════════════════════════════════════════════════════════════
标准奖励返回结构
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any
import numpy as np


@dataclass
class RewardInfo:
    """
    标准奖励信息对象
    
    所有奖励计算完成后，统一返回这个结构，确保上层（workflow、logger）
    能够以一致的方式理解和记录奖励。
    """
    
    # ========================================================================
    # A 层 - 主目标层
    # ========================================================================
    
    # A1: 生存奖励
    survival_base_reward: float = 0.0      # 每步基础生存奖励
    completion_bonus: float = 0.0          # 活到最后的完成奖励
    death_penalty: float = 0.0             # 死亡惩罚
    
    # A2: 得分奖励
    treasure_pickup_reward: float = 0.0    # 拿宝箱的奖励
    treasure_lazy_penalty: float = 0.0     # 懒惰不追求宝箱的惩罚
    
    # ========================================================================
    # B 层 - 引导层
    # ========================================================================
    
    # B1: 距离成形
    dist_shaping: float = 0.0              # 与怪物距离变化的成形
    
    # B2: 危险区惩罚
    danger_zone_penalty: float = 0.0       # 进入危险区的惩罚
    
    # B3: Buff 奖励
    buff_acquisition_reward: float = 0.0   # 获取 buff 的奖励
    buff_active_bonus: float = 0.0         # buff 生效期间的奖励
    buff_utilization_reward: float = 0.0   # 充分利用 buff 的奖励
    
    # B4: 闪现脱险
    flash_escape_reward: float = 0.0       # 闪现脱险的奖励（紧急 + 战术）
    
    # ========================================================================
    # C 层 - 约束层
    # ========================================================================
    
    # C1: 低效移动
    movement_penalty: float = 0.0          # 不动或极少移动的惩罚
    
    # C2: 绕圈检测
    loiter_penalty: float = 0.0            # 绕圈惩罚
    
    # C3: 陷阱规避
    invalid_style_penalty: float = 0.0     # 往死路钻的惩罚
    
    # C4: 技能浪费
    wasteful_flash_penalty: float = 0.0    # 无意义闪现的惩罚
    
    # ========================================================================
    # 中间计算值和调试信息（可选）
    # ========================================================================
    
    # 位置和状态信息（用于日志和调试）
    min_monster_dist_norm: float = 0.0     # 当前到最近怪物的距离（归一化）
    nearest_treasure_dist_norm: float = 0.0  # 当前到最近宝箱的距离（归一化）
    move_dist: float = 0.0                 # 本步移动距离
    stuck_score: float = 0.0               # 卡住的程度（0-1）
    
    # ========================================================================
    # 聚合字段
    # ========================================================================
    
    # 三层总和（由 workflow 计算）
    layer_a_total: float = 0.0             # A 层总奖励
    layer_b_total: float = 0.0             # B 层总奖励
    layer_c_total: float = 0.0             # C 层总奖励
    reward_total: float = 0.0              # 总奖励 = A + B + C
    
    def compute_aggregates(self):
        """自动计算三层总和。"""
        self.layer_a_total = (
            self.survival_base_reward +
            self.completion_bonus +
            self.death_penalty +
            self.treasure_pickup_reward +
            self.treasure_lazy_penalty
        )
        
        self.layer_b_total = (
            self.dist_shaping +
            self.danger_zone_penalty +
            self.buff_acquisition_reward +
            self.buff_active_bonus +
            self.buff_utilization_reward +
            self.flash_escape_reward
        )
        
        self.layer_c_total = (
            self.movement_penalty +
            self.loiter_penalty +
            self.invalid_style_penalty +
            self.wasteful_flash_penalty
        )
        
        self.reward_total = self.layer_a_total + self.layer_b_total + self.layer_c_total
        
        return self.reward_total
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典（用于 JSON 序列化和日志）。"""
        return asdict(self)
    
    def get_layer_breakdown(self) -> Dict[str, float]:
        """获取三层分解结果。"""
        return {
            "layer_a_total": self.layer_a_total,
            "layer_b_total": self.layer_b_total,
            "layer_c_total": self.layer_c_total,
            "reward_total": self.reward_total,
        }


@dataclass
class EpisodeStatistics:
    """
    单局对局统计（由 workflow 累计）
    
    用于给出清晰的"这一局学到了什么"的总结。
    """
    
    # 基本信息
    episode_id: int = 0
    total_steps: int = 0
    total_score: float = 0.0
    treasures_collected: int = 0
    total_treasures: int = 0
    buffs_collected: int = 0
    flashes_used: int = 0
    survived: bool = False
    
    # ========================================================================
    # A 层累计
    # ========================================================================
    
    layer_a_cumulative: float = 0.0        # A 层总累计奖励
    avg_survival_per_step: float = 0.0     # 平均每步生存奖励
    total_treasure_rewards: float = 0.0    # 本局所有宝箱奖励
    
    # ========================================================================
    # B 层累计
    # ========================================================================
    
    layer_b_cumulative: float = 0.0        # B 层总累计奖励
    total_dist_shaping: float = 0.0        # 距离成形的总贡献
    total_danger_penalty: float = 0.0      # 危险区惩罚的总贡献
    total_buff_reward: float = 0.0         # buff 奖励的总贡献
    total_flash_reward: float = 0.0        # 闪现脱险的总贡献
    
    # ========================================================================
    # C 层累计
    # ========================================================================
    
    layer_c_cumulative: float = 0.0        # C 层总累计惩罚
    total_movement_penalty: float = 0.0    # 低效移动的总惩罚
    total_loiter_penalty: float = 0.0      # 绕圈的总惩罚
    total_invalid_penalty: float = 0.0     # 陷阱规避的总惩罚
    total_wasteful_flash_penalty: float = 0.0  # 技能浪费的总惩罚
    
    # ========================================================================
    # 平均指标
    # ========================================================================
    
    avg_min_monster_dist: float = 0.0      # 平均最小怪物距离（归一化）
    avg_nearest_treasure_dist: float = 0.0 # 平均最近宝箱距离（归一化）
    
    # ========================================================================
    # 最终指标
    # ========================================================================
    
    total_reward: float = 0.0              # 本局总奖励
    reward_per_step: float = 0.0           # 平均每步奖励
    
    def finalize(self):
        """计算最终统计。"""
        if self.total_steps > 0:
            self.reward_per_step = self.total_reward / self.total_steps
            self.avg_survival_per_step = self.layer_a_cumulative / self.total_steps if self.layer_a_cumulative != 0 else 0.0
        
        self.total_reward = self.layer_a_cumulative + self.layer_b_cumulative + self.layer_c_cumulative
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)
    
    def get_summary_string(self) -> str:
        """获取可读的总结字符串。"""
        status = "WIN" if self.survived else "FAIL"
        return (
            f"[Episode {self.episode_id}] {status} | "
            f"Steps={self.total_steps} | "
            f"Score={self.total_score:.0f} | "
            f"Treasure={self.treasures_collected}/{self.total_treasures} | "
            f"Flashes={self.flashes_used} | "
            f"Reward={self.total_reward:.2f} | "
            f"A={self.layer_a_cumulative:.2f} B={self.layer_b_cumulative:.2f} C={self.layer_c_cumulative:.2f}"
        )


def create_reward_info() -> RewardInfo:
    """工厂方法：创建空的 RewardInfo。"""
    return RewardInfo()


def create_episode_statistics(episode_id: int = 0) -> EpisodeStatistics:
    """工厂方法：创建空的 EpisodeStatistics。"""
    return EpisodeStatistics(episode_id=episode_id)


# ============================================================================
# 日志记录规范
# ============================================================================

def format_reward_step_log(
    step_no: int,
    reward_info: RewardInfo,
    action_id: int,
) -> str:
    """
    格式化单步奖励日志。
    
    用法: logger.debug(format_reward_step_log(...))
    """
    return (
        f"STEP {step_no:4d} | "
        f"ACT {action_id:2d} | "
        f"Reward={reward_info.reward_total:+.3f} | "
        f"A={reward_info.layer_a_total:+.3f} "
        f"B={reward_info.layer_b_total:+.3f} "
        f"C={reward_info.layer_c_total:+.3f}"
    )


def format_reward_episode_log(episode_stats: EpisodeStatistics) -> str:
    """
    格式化对局结束奖励日志。
    
    用法: logger.info(format_reward_episode_log(...))
    """
    return (
        f"[EPISODE {episode_stats.episode_id:5d}] "
        f"Result={'WIN' if episode_stats.survived else 'FAIL':4s} | "
        f"Steps={episode_stats.total_steps:3d} | "
        f"Score={episode_stats.total_score:7.0f} | "
        f"Treasure={episode_stats.treasures_collected}/{episode_stats.total_treasures} | "
        f"Flash={episode_stats.flashes_used} | "
        f"RewardTotal={episode_stats.total_reward:+.3f} | "
        f"LayerA={episode_stats.layer_a_cumulative:+.3f} "
        f"LayerB={episode_stats.layer_b_cumulative:+.3f} "
        f"LayerC={episode_stats.layer_c_cumulative:+.3f}"
    )


if __name__ == "__main__":
    # 测试示例
    print("=" * 120)
    print("Reward System Specification Test")
    print("=" * 120)
    
    # 创建示例奖励信息
    reward_info = RewardInfo(
        survival_base_reward=0.02,
        treasure_pickup_reward=2.0,
        dist_shaping=0.10,
        flash_escape_reward=0.6,
        movement_penalty=-0.05,
    )
    reward_total = reward_info.compute_aggregates()
    print("\n单步奖励示例:")
    print(f"  总奖励: {reward_total:.3f}")
    print(f"  A 层: {reward_info.layer_a_total:.3f} (生存+得分)")
    print(f"  B 层: {reward_info.layer_b_total:.3f} (引导)")
    print(f"  C 层: {reward_info.layer_c_total:.3f} (约束)")
    
    # 创建示例对局统计
    episode_stats = EpisodeStatistics(
        episode_id=123,
        total_steps=150,
        total_score=1500,
        treasures_collected=8,
        total_treasures=10,
        buffs_collected=2,
        flashes_used=3,
        survived=True,
        layer_a_cumulative=15.2,
        layer_b_cumulative=8.5,
        layer_c_cumulative=-2.3,
    )
    episode_stats.finalize()
    print("\n对局统计示例:")
    print(f"  {episode_stats.get_summary_string()}")
    
    print("\n" + "=" * 120)
