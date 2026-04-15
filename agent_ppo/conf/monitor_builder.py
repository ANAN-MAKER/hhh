#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase.
峡谷追猎监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(
            group_name="训练指标",
            group_name_en="training",
        )
        .add_panel(
            name="奖励表现",
            name_en="reward_metrics",
            type="line",
        )
        .add_metric(
            metrics_name="sample_reward",
            expr="avg(reward{})",
        )
        .add_metric(
            metrics_name="episode_reward",
            expr="avg(episode_reward{})",
        )
        .add_metric(
            metrics_name="avg_step_reward",
            expr="avg(avg_step_reward{})",
        )
        .end_panel()
        .add_panel(
            name="损失指标",
            name_en="loss_metrics",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="PPO稳定性",
            name_en="ppo_stability",
            type="line",
        )
        .add_metric(
            metrics_name="approx_kl",
            expr="avg(approx_kl{})",
        )
        .add_metric(
            metrics_name="clip_frac",
            expr="avg(clip_frac{})",
        )
        .end_panel()
        .add_panel(
            name="对局结果",
            name_en="episode_outcome",
            type="line",
        )
        .add_metric(
            metrics_name="episode_score",
            expr="avg(episode_score{})",
        )
        .add_metric(
            metrics_name="episode_steps",
            expr="avg(episode_steps{})",
        )
        .add_metric(
            metrics_name="episode_step_ratio",
            expr="avg(episode_step_ratio{})",
        )
        .add_metric(
            metrics_name="win_flag",
            expr="avg(win_flag{})",
        )
        .end_panel()
        .add_panel(
            name="资源获取",
            name_en="resource_metrics",
            type="line",
        )
        .add_metric(
            metrics_name="episode_treasure_count",
            expr="avg(episode_treasure_count{})",
        )
        .add_metric(
            metrics_name="episode_treasure_ratio",
            expr="avg(episode_treasure_ratio{})",
        )
        .add_metric(
            metrics_name="episode_buff_count",
            expr="avg(episode_buff_count{})",
        )
        .add_metric(
            metrics_name="episode_flash_count",
            expr="avg(episode_flash_count{})",
        )
        .end_panel()
        .add_panel(
            name="走位安全",
            name_en="movement_safety",
            type="line",
        )
        .add_metric(
            metrics_name="avg_min_monster_dist",
            expr="avg(avg_min_monster_dist{})",
        )
        .add_metric(
            metrics_name="avg_nearest_treasure_dist",
            expr="avg(avg_nearest_treasure_dist{})",
        )
        .add_metric(
            metrics_name="avg_move_dist",
            expr="avg(avg_move_dist{})",
        )
        .add_metric(
            metrics_name="avg_stuck_score",
            expr="avg(avg_stuck_score{})",
        )
        .end_panel()
        .add_panel(
            name="塑形拆分",
            name_en="shaping_breakdown",
            type="line",
        )
        .add_metric(
            metrics_name="treasure_bonus",
            expr="avg(treasure_bonus{})",
        )
        .add_metric(
            metrics_name="exploration_bonus",
            expr="avg(exploration_bonus{})",
        )
        .add_metric(
            metrics_name="anti_loiter_penalty",
            expr="avg(anti_loiter_penalty{})",
        )
        .add_metric(
            metrics_name="escape_bonus",
            expr="avg(escape_bonus{})",
        )
        .add_metric(
            metrics_name="flash_bonus",
            expr="avg(flash_bonus{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
