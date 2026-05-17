#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("HoK 1v1 PPO")
        .add_group(group_name="Algorithm", group_name_en="algorithm")
        .add_panel(name="Loss", name_en="loss", type="line")
        .add_metric(metrics_name="total_loss", expr="round(avg(total_loss{}), 0.01)")
        .add_metric(metrics_name="value_loss", expr="round(avg(value_loss{}), 0.01)")
        .add_metric(metrics_name="policy_loss", expr="round(avg(policy_loss{}), 0.01)")
        .add_metric(metrics_name="entropy_loss", expr="round(avg(entropy_loss{}), 0.01)")
        .end_panel()
        .end_group()
        .add_group(group_name="Reward Shaping", group_name_en="reward_shaping")
        .add_panel(name="Reward Sum", name_en="reward_sum_panel", type="line")
        .add_metric(metrics_name="reward_sum", expr="round(avg(reward_sum{}), 0.01)")
        .end_panel()
        .add_panel(name="Reward Parts", name_en="reward_parts", type="line")
        .add_metric(metrics_name="terminal_reward", expr="round(avg(terminal_reward{}), 0.01)")
        .add_metric(metrics_name="pushing_reward", expr="round(avg(pushing_reward{}), 0.01)")
        .add_metric(metrics_name="combat_reward", expr="round(avg(combat_reward{}), 0.01)")
        .add_metric(metrics_name="farming_reward", expr="round(avg(farming_reward{}), 0.01)")
        .add_metric(metrics_name="tempo_reward", expr="round(avg(tempo_reward{}), 0.01)")
        .end_panel()
        .end_group()
        .add_group(group_name="Episode Metrics", group_name_en="episode_metrics")
        .add_panel(name="Win Rate", name_en="win_rate_panel", type="line")
        .add_metric(metrics_name="win_rate", expr="round(avg(win_rate{}), 0.01)")
        .end_panel()
        .add_panel(name="Tower Pressure", name_en="tower_pressure", type="line")
        .add_metric(metrics_name="self_tower_hp", expr="round(avg(self_tower_hp{}), 0.01)")
        .add_metric(metrics_name="enemy_tower_hp", expr="round(avg(enemy_tower_hp{}), 0.01)")
        .end_panel()
        .add_panel(name="Economy", name_en="economy", type="line")
        .add_metric(metrics_name="money_per_frame", expr="round(avg(money_per_frame{}), 0.01)")
        .end_panel()
        .add_panel(name="Combat", name_en="combat", type="line")
        .add_metric(metrics_name="kill", expr="round(avg(kill{}), 0.01)")
        .add_metric(metrics_name="death", expr="round(avg(death{}), 0.01)")
        .add_metric(metrics_name="hurt_to_hero", expr="round(avg(hurt_to_hero{}), 0.01)")
        .add_metric(metrics_name="hurt_by_hero", expr="round(avg(hurt_by_hero{}), 0.01)")
        .end_panel()
        .add_panel(name="Episode Length", name_en="episode_length", type="line")
        .add_metric(metrics_name="frame", expr="round(avg(frame{}), 0.01)")
        .end_panel()
        .end_group()
        .add_group(group_name="Decision Trace", group_name_en="decision_trace")
        .add_panel(name="Vision Summary", name_en="vision_summary", type="line")
        .add_metric(metrics_name="visible_enemy_hero_rate", expr="round(avg(visible_enemy_hero_rate{}), 0.01)")
        .add_metric(
            metrics_name="avg_visible_enemy_soldier_count",
            expr="round(avg(avg_visible_enemy_soldier_count{}), 0.01)",
        )
        .end_panel()
        .add_panel(name="Decision Rates", name_en="decision_rates", type="line")
        .add_metric(metrics_name="lane_clear_rate", expr="round(avg(lane_clear_rate{}), 0.01)")
        .add_metric(metrics_name="hero_pressure_rate", expr="round(avg(hero_pressure_rate{}), 0.01)")
        .add_metric(metrics_name="tower_pressure_rate", expr="round(avg(tower_pressure_rate{}), 0.01)")
        .add_metric(metrics_name="retreat_rate", expr="round(avg(retreat_rate{}), 0.01)")
        .end_panel()
        .add_panel(name="Decision Scores", name_en="decision_scores", type="line")
        .add_metric(metrics_name="avg_lane_clear_score", expr="round(avg(avg_lane_clear_score{}), 0.01)")
        .add_metric(metrics_name="avg_hero_pressure_score", expr="round(avg(avg_hero_pressure_score{}), 0.01)")
        .add_metric(metrics_name="avg_tower_pressure_score", expr="round(avg(avg_tower_pressure_score{}), 0.01)")
        .add_metric(metrics_name="avg_retreat_score", expr="round(avg(avg_retreat_score{}), 0.01)")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
