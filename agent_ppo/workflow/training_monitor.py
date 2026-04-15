#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training Monitoring and Statistics Aggregation.
训练监控与统计聚合模块。

该模块职责：
  1. 在训练流程中统一收集和记录奖励信息
  2. 按 A/B/C 三层设计统计每局对局
  3. 提供清晰的日志输出，便于训练分析
  4. 验证奖励设计和监控口径的一致性
"""

from collections import defaultdict
from typing import Dict, Any, Optional
import numpy as np

from agent_ppo.feature.reward_spec import (
    RewardInfo,
    EpisodeStatistics,
    format_reward_episode_log,
)


class TrainingMonitor:
    """
    训练监控器
    
    职责：
      - 收集每一步的奖励信息
      - 按 A/B/C 三层统计每局对局
      - 维护训练统计的一致性
      - 输出标准化的监控日志
    """
    
    def __init__(self, logger=None):
        self.logger = logger
        
        # 当前对局的奖励累计
        self.current_episode_rewards = []
        self.current_episode_steps = 0
        self.current_episode_env_info = {}
        
        # 历史统计
        self.episode_histories = []
        self.window_size = 100  # 最近 100 局的窗口
    
    def reset_episode(self):
        """重置对局统计（对局开始时调用）。"""
        self.current_episode_rewards = []
        self.current_episode_steps = 0
        self.current_episode_env_info = {}
    
    def record_step_reward(self, reward_info: RewardInfo, env_step_info: Dict[str, Any]):
        """
        记录单步奖励。
        
        Args:
            reward_info: 本步奖励信息（应该已调用 compute_aggregates()）
            env_step_info: 环境步长信息（位置、状态等）
        """
        self.current_episode_rewards.append(reward_info)
        self.current_episode_steps += 1
    
    def finalize_episode(
        self,
        episode_id: int,
        env_obs: Dict[str, Any],
        survived: bool,
    ) -> EpisodeStatistics:
        """
        结束对局，计算最终统计。
        
        Args:
            episode_id: 对局 ID
            env_obs: 最终环境观测
            survived: 是否生存（WIN vs FAIL）
        
        Returns:
            EpisodeStatistics: 该对局的完整统计
        """
        # 提取环境信息
        env_info = env_obs.get("observation", {}).get("env_info", {})
        
        # 计算三层累计
        layer_a_cum = 0.0
        layer_b_cum = 0.0
        layer_c_cum = 0.0
        
        for reward_info in self.current_episode_rewards:
            layer_a_cum += reward_info.layer_a_total
            layer_b_cum += reward_info.layer_b_total
            layer_c_cum += reward_info.layer_c_total
        
        # 提取数据
        total_score = float(env_info.get("total_score", 0.0))
        treasures_collected = int(env_info.get("treasures_collected", 0))
        total_treasures = int(env_info.get("total_treasure", 0))
        collected_buff = int(env_info.get("collected_buff", 0))
        flash_count = int(env_info.get("flash_count", 0))
        
        # 计算平均距离等指标
        avg_min_dist = 0.0
        avg_treasure_dist = 0.0
        if self.current_episode_rewards:
            avg_min_dist = np.mean([r.min_monster_dist_norm for r in self.current_episode_rewards])
            avg_treasure_dist = np.mean([r.nearest_treasure_dist_norm for r in self.current_episode_rewards])
        
        # 创建统计对象
        stats = EpisodeStatistics(
            episode_id=episode_id,
            total_steps=self.current_episode_steps,
            total_score=total_score,
            treasures_collected=treasures_collected,
            total_treasures=total_treasures,
            buffs_collected=collected_buff,
            flashes_used=flash_count,
            survived=survived,
            layer_a_cumulative=layer_a_cum,
            layer_b_cumulative=layer_b_cum,
            layer_c_cumulative=layer_c_cum,
            avg_min_monster_dist=avg_min_dist,
            avg_nearest_treasure_dist=avg_treasure_dist,
        )
        
        # 详细统计各项贡献
        layer_a_details = self._aggregate_rewards(
            "survival_base_reward", "completion_bonus", "death_penalty",
            "treasure_pickup_reward", "treasure_lazy_penalty"
        )
        layer_b_details = self._aggregate_rewards(
            "dist_shaping", "danger_zone_penalty",
            "buff_acquisition_reward", "buff_active_bonus", "buff_utilization_reward",
            "flash_escape_reward"
        )
        layer_c_details = self._aggregate_rewards(
            "movement_penalty", "loiter_penalty",
            "invalid_style_penalty", "wasteful_flash_penalty"
        )
        
        stats.total_treasure_rewards = layer_a_details.get("treasure_pickup_reward", 0.0) + layer_a_details.get("treasure_lazy_penalty", 0.0)
        stats.total_dist_shaping = layer_b_details.get("dist_shaping", 0.0)
        stats.total_danger_penalty = layer_b_details.get("danger_zone_penalty", 0.0)
        stats.total_buff_reward = (layer_b_details.get("buff_acquisition_reward", 0.0)
                                   + layer_b_details.get("buff_active_bonus", 0.0)
                                   + layer_b_details.get("buff_utilization_reward", 0.0))
        stats.total_flash_reward = layer_b_details.get("flash_escape_reward", 0.0)
        
        stats.total_movement_penalty = layer_c_details.get("movement_penalty", 0.0)
        stats.total_loiter_penalty = layer_c_details.get("loiter_penalty", 0.0)
        stats.total_invalid_penalty = layer_c_details.get("invalid_style_penalty", 0.0)
        stats.total_wasteful_flash_penalty = layer_c_details.get("wasteful_flash_penalty", 0.0)
        
        stats.finalize()
        
        # 存储历史
        self.episode_histories.append(stats)
        if len(self.episode_histories) > self.window_size:
            self.episode_histories.pop(0)
        
        return stats
    
    def _aggregate_rewards(self, *field_names) -> Dict[str, float]:
        """聚合某些奖励字段的总和。"""
        result = {}
        for field_name in field_names:
            total = sum(
                getattr(reward, field_name, 0.0)
                for reward in self.current_episode_rewards
            )
            result[field_name] = total
        return result
    
    def get_window_statistics(self) -> Dict[str, Any]:
        """
        获取窗口内（最近 N 局）的统计数据。
        
        用于定期输出到日志和监控系统。
        """
        if not self.episode_histories:
            return {}
        
        histories = self.episode_histories[-self.window_size:]
        
        total_episodes = len(histories)
        win_episodes = sum(1 for s in histories if s.survived)
        win_rate = win_episodes / total_episodes if total_episodes > 0 else 0.0
        
        avg_steps = np.mean([s.total_steps for s in histories])
        avg_score = np.mean([s.total_score for s in histories])
        avg_treasures = np.mean([s.treasures_collected for s in histories])
        avg_flashes = np.mean([s.flashes_used for s in histories])
        
        avg_layer_a = np.mean([s.layer_a_cumulative for s in histories])
        avg_layer_b = np.mean([s.layer_b_cumulative for s in histories])
        avg_layer_c = np.mean([s.layer_c_cumulative for s in histories])
        avg_total = np.mean([s.total_reward for s in histories])
        
        avg_min_dist = np.mean([s.avg_min_monster_dist for s in histories])
        avg_treasure_dist = np.mean([s.avg_nearest_treasure_dist for s in histories])
        
        return {
            "window_episodes": total_episodes,
            "win_rate": win_rate,
            "avg_steps": avg_steps,
            "avg_score": avg_score,
            "avg_treasures": avg_treasures,
            "avg_flashes": avg_flashes,
            "avg_layer_a_reward": avg_layer_a,
            "avg_layer_b_reward": avg_layer_b,
            "avg_layer_c_reward": avg_layer_c,
            "avg_total_reward": avg_total,
            "avg_min_monster_dist": avg_min_dist,
            "avg_treasure_dist": avg_treasure_dist,
        }
    
    def log_episode_summary(self, stats: EpisodeStatistics, include_details: bool = True):
        """
        记录对局总结。
        
        Args:
            stats: EpisodeStatistics 对象
            include_details: 是否包含详细的奖励分解
        """
        if not self.logger:
            return
        
        # 主要总结日志
        summary_msg = format_reward_episode_log(stats)
        self.logger.info(summary_msg)
        
        if include_details:
            # 详细分解日志
            details_msg = (
                f"  Layer Details: "
                f"A[survival={stats.layer_a_cumulative:.2f}] "
                f"B[shaping={stats.total_dist_shaping:.2f} "
                f"danger={stats.total_danger_penalty:.2f} "
                f"buff={stats.total_buff_reward:.2f} "
                f"flash={stats.total_flash_reward:.2f}] "
                f"C[move={stats.total_movement_penalty:.2f} "
                f"loiter={stats.total_loiter_penalty:.2f} "
                f"invalid={stats.total_invalid_penalty:.2f} "
                f"flash_waste={stats.total_wasteful_flash_penalty:.2f}]"
            )
            self.logger.debug(details_msg)
    
    def log_window_summary(self, force: bool = False):
        """
        记录窗口统计总结。
        
        Args:
            force: 是否强制输出（否则只在达到窗口大小时输出）
        """
        if not self.logger or (not force and len(self.episode_histories) < 10):
            return
        
        stats = self.get_window_statistics()
        msg = (
            f"[WINDOW STATS] "
            f"Episodes={stats.get('window_episodes', 0)} "
            f"WinRate={stats.get('win_rate', 0.0):.1%} | "
            f"AvgScore={stats.get('avg_score', 0.0):.0f} "
            f"AvgTreasure={stats.get('avg_treasures', 0.0):.1f} "
            f"AvgFlash={stats.get('avg_flashes', 0.0):.1f} | "
            f"AvgReward={stats.get('avg_total_reward', 0.0):.2f} "
            f"(A={stats.get('avg_layer_a_reward', 0.0):.2f} "
            f"B={stats.get('avg_layer_b_reward', 0.0):.2f} "
            f"C={stats.get('avg_layer_c_reward', 0.0):.2f})"
        )
        self.logger.info(msg)


if __name__ == "__main__":
    # 测试示例
    print("=" * 100)
    print("Training Monitor Test")
    print("=" * 100)
    
    monitor = TrainingMonitor()
    
    # 模拟对局
    for episode in range(5):
        monitor.reset_episode()
        for step in range(100):
            reward_info = RewardInfo(
                survival_base_reward=0.02,
                dist_shaping=0.05,
                movement_penalty=-0.01,
            )
            reward_info.compute_aggregates()
            monitor.record_step_reward(reward_info, {})
        
        survived = episode % 2 == 0
        stats = monitor.finalize_episode(
            episode_id=episode,
            env_obs={
                "observation": {
                    "env_info": {
                        "total_score": 500.0 + episode * 100,
                        "treasures_collected": 5 + episode,
                        "total_treasure": 10,
                        "collected_buff": 1,
                        "flash_count": 2 + episode,
                    }
                }
            },
            survived=survived,
        )
        print(f"\nEpisode {episode}: {stats.get_summary_string()}")
    
    print("\n" + "=" * 100)
    print("Window Statistics:")
    window_stats = monitor.get_window_statistics()
    for key, value in window_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
