#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _panel(builder, name, metrics, panel_type="line"):
    builder = builder.add_panel(name=name, name_en=name, type=panel_type)
    for metrics_name, expr in metrics:
        builder = builder.add_metric(metrics_name=metrics_name, expr=expr)
    return builder.end_panel()


def build_monitor():
    builder = MonitorConfigBuilder().title("hok1v1_luban_stage_a_plus")

    builder = builder.add_group(group_name="result", group_name_en="result")
    builder = _panel(
        builder,
        "win_rate",
        [
            ("eval_vs_luban_v2_5229", "round(avg(win_rate_model_266689{}), 0.01)"),
            ("train_vs_common_ai", "round(avg(win_rate_train_common_ai{}), 0.01)"),
        ],
    )
    builder = _panel(builder, "episode_reward", [("reward_total", "round(avg(reward{}), 0.01)")])
    builder = builder.end_group()

    builder = builder.add_group(group_name="combat_objective", group_name_en="combat_objective")
    builder = _panel(
        builder,
        "tower_state",
        [
            ("self_tower_hp_ratio", "round(avg(self_tower_hp_ratio{}), 0.01)"),
            ("enemy_tower_hp_ratio", "round(avg(enemy_tower_hp_ratio{}), 0.01)"),
            ("tower_hp_advantage", "round(avg(tower_hp_advantage{}), 0.01)"),
            ("enemy_tower_hp_drop", "round(avg(enemy_tower_hp_drop{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "hero_state",
        [
            ("self_hp_ratio", "round(avg(self_hp_ratio{}), 0.01)"),
            ("enemy_hp_ratio", "round(avg(enemy_hp_ratio{}), 0.01)"),
            ("hp_advantage", "round(avg(hp_advantage{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="economy_lane", group_name_en="economy_lane")
    builder = _panel(
        builder,
        "economy",
        [
            ("self_money", "round(avg(self_money{}), 0.01)"),
            ("enemy_money", "round(avg(enemy_money{}), 0.01)"),
            ("gold_advantage", "round(avg(gold_advantage{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "lane",
        [
            ("lane_progress", "round(avg(lane_progress{}), 0.01)"),
            ("my_soldier_count", "round(avg(my_soldier_count{}), 0.01)"),
            ("enemy_soldier_count", "round(avg(enemy_soldier_count{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="reward_items", group_name_en="reward_items")
    builder = _panel(
        builder,
        "stage_a_plus_reward",
        [
            ("reward_hp", "round(avg(reward_hp{}), 0.01)"),
            ("reward_tower", "round(avg(reward_tower{}), 0.01)"),
            ("reward_gold", "round(avg(reward_gold{}), 0.01)"),
            ("reward_exp", "round(avg(reward_exp{}), 0.01)"),
            ("reward_last_hit", "round(avg(reward_last_hit{}), 0.01)"),
            ("reward_death", "round(avg(reward_death{}), 0.01)"),
            ("reward_kill", "round(avg(reward_kill{}), 0.01)"),
            ("reward_forward", "round(avg(reward_forward{}), 0.01)"),
            ("reward_win", "round(avg(reward_win{}), 0.01)"),
            ("reward_push_assist", "round(avg(reward_push_assist{}), 0.01)"),
            ("reward_lane_follow", "round(avg(reward_lane_follow{}), 0.01)"),
            ("reward_finish", "round(avg(reward_finish{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="damage_income_source", group_name_en="damage_income_source")
    builder = _panel(
        builder,
        "damage_source",
        [
            ("hurt_to_hero_delta", "round(avg(hurt_to_hero_delta{}), 0.01)"),
            ("enemy_tower_hp_drop", "round(avg(enemy_tower_hp_drop{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "gold_source",
        [
            ("gold_from_soldier", "round(avg(gold_from_soldier{}), 0.01)"),
            ("gold_from_hero", "round(avg(gold_from_hero{}), 0.01)"),
            ("gold_from_other", "round(avg(gold_from_other{}), 0.01)"),
            ("gold_unattributed", "round(avg(gold_unattributed{}), 0.01)"),
            ("gold_total_delta", "round(avg(gold_total_delta{}), 0.01)"),
            ("soldier_kill_count", "round(avg(soldier_kill_count{}), 0.01)"),
            ("hero_kill_count", "round(avg(hero_kill_count{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="behavior", group_name_en="behavior")
    builder = _panel(
        builder,
        "action_rate",
        [
            ("move_action_rate", "round(avg(move_action_rate{}), 0.01)"),
            ("attack_action_rate", "round(avg(attack_action_rate{}), 0.01)"),
            ("skill_action_rate", "round(avg(skill_action_rate{}), 0.01)"),
            ("summoner_action_rate", "round(avg(summoner_action_rate{}), 0.01)"),
            ("recall_action_rate", "round(avg(recall_action_rate{}), 0.01)"),
            ("low_movement_rate", "round(avg(low_movement_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "target_rate",
        [
            ("hero_target_rate", "round(avg(hero_target_rate{}), 0.01)"),
            ("creep_target_rate", "round(avg(creep_target_rate{}), 0.01)"),
            ("tower_target_rate", "round(avg(tower_target_rate{}), 0.01)"),
        ],
    )
    builder = _panel(
        builder,
        "tower_risk",
        [
            ("enemy_tower_range_rate", "round(avg(enemy_tower_range_rate{}), 0.01)"),
            ("no_minion_tower_dive_rate", "round(avg(no_minion_tower_dive_rate{}), 0.01)"),
            ("safe_push_window_rate", "round(avg(safe_push_window_rate{}), 0.01)"),
            ("tower_attack_with_minion_rate", "round(avg(tower_attack_with_minion_rate{}), 0.01)"),
        ],
    )
    builder = builder.end_group()

    builder = builder.add_group(group_name="training", group_name_en="training")
    builder = _panel(
        builder,
        "loss",
        [
            ("total_loss", "round(avg(total_loss{}), 0.01)"),
            ("value_loss", "round(avg(value_loss{}), 0.01)"),
            ("policy_loss", "round(avg(policy_loss{}), 0.01)"),
            ("entropy_loss", "round(avg(entropy_loss{}), 0.01)"),
        ],
    )
    return builder.end_group().build()
