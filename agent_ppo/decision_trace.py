#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import json
import os
import time
from collections import Counter

from agent_ppo.conf.conf import DebugConfig


BUTTON_NAMES = [
    "illegal",
    "none",
    "move",
    "normal_attack",
    "skill_1",
    "skill_2",
    "skill_3",
    "heal_skill",
    "chosen_skill",
    "recall",
    "skill_4",
    "equipment_skill",
]
TARGET_NAMES = [
    "none",
    "enemy_hero",
    "self_hero",
    "soldier_1",
    "soldier_2",
    "soldier_3",
    "soldier_4",
    "tower",
    "monster",
]
BRANCH_NAMES = ["button", "move_x", "move_z", "skill_x", "skill_z", "target"]
PRIMARY_REASONS = [
    "lane_clear",
    "hero_pressure",
    "tower_pressure",
    "retreat",
]
PRIMARY_REASON_LABELS_ZH = {
    "lane_clear": "清线",
    "hero_pressure": "压人",
    "tower_pressure": "推塔",
    "retreat": "后撤",
}


class DecisionTracer:
    def __init__(self, logger=None, monitor=None):
        self.logger = logger
        self.monitor = monitor
        self.run_id = "run_{time}_{pid}".format(time=time.strftime("%Y%m%d_%H%M%S"), pid=os.getpid())
        self.base_dir = os.path.join(DebugConfig.TRACE_OUTPUT_DIR, self.run_id)
        self.file_enabled = bool(DebugConfig.ENABLE_TRACE)
        self.monitor_enabled = bool(DebugConfig.ENABLE_TRACE_MONITOR)
        self.trace_enabled = self.file_enabled or self.monitor_enabled
        self.trace_topk = max(1, int(DebugConfig.TRACE_TOPK))
        self.trace_interval = max(1, int(DebugConfig.TRACE_FRAME_INTERVAL))
        self.log_interval_frames = max(1, int(getattr(DebugConfig, "TRACE_LOG_INTERVAL_FRAMES", 20)))
        self.last_monitor_report_time = 0.0

        self.current_mode = "predict"
        self.current_opponent_type = "unknown"
        self.current_player_id = None
        self.current_hero_camp = None
        self.current_hero_id = None
        self.current_episode = 0
        self.current_episode_path = None
        self.current_summary_path = None
        self.current_file = None
        self.pending_record = None
        self.summary = self._new_summary()
        self.window_stats = self._new_window_stats()
        self.last_trace_snapshot = None

        if self.trace_enabled:
            os.makedirs(self.base_dir, exist_ok=True)

    def set_context(self, mode, opponent_type):
        self.current_mode = mode
        self.current_opponent_type = opponent_type

    def start_episode(self, observation, hero_camp, player_id):
        if not self.trace_enabled:
            return

        self.finalize_episode()
        self.current_episode += 1
        self.current_player_id = player_id
        self.current_hero_camp = hero_camp
        self.current_hero_id = self._extract_hero_id(observation)
        self.last_trace_snapshot = None

        player_dir = os.path.join(self.base_dir, "player_{0}".format(player_id))
        os.makedirs(player_dir, exist_ok=True)

        self.current_episode_path = os.path.join(player_dir, "episode_{0}.jsonl".format(self.current_episode))
        self.current_summary_path = os.path.join(player_dir, "summary.json")
        self.summary = self._load_summary(self.current_summary_path)
        self.window_stats = self._new_window_stats()
        self.pending_record = None

        if self.file_enabled:
            self.current_file = open(self.current_episode_path, "w", encoding="utf-8")

    def finalize_episode(self, observation=None):
        if not self.trace_enabled:
            return

        self._flush_pending(observation)
        self._report_monitor(force=True)

        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None

        if self.current_summary_path is not None:
            payload = dict(self.summary)
            count = max(1, int(payload["trace_frame_cnt"]))
            payload["avg_state_value"] = payload["avg_state_value_sum"] / count
            payload["avg_selected_button_prob"] = payload["avg_selected_button_prob_sum"] / count
            payload["visible_enemy_hero_rate"] = payload["visible_enemy_hero_count"] / count
            payload["avg_visible_enemy_soldier_count"] = payload["visible_enemy_soldier_sum"] / count
            payload["visible_enemy_soldier_avg"] = payload["avg_visible_enemy_soldier_count"]
            payload["attack_soldier_rate"] = payload["attack_soldier_cnt"] / count
            payload["attack_hero_rate"] = payload["attack_hero_cnt"] / count
            payload["attack_tower_rate"] = payload["attack_tower_cnt"] / count
            for reason_name in PRIMARY_REASONS:
                payload["{0}_rate".format(reason_name)] = payload["reason_count"].get(reason_name, 0) / count
                payload["avg_{0}_score".format(reason_name)] = payload["decision_score_sum"][reason_name] / count
            payload["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.current_summary_path, "w", encoding="utf-8") as summary_file:
                json.dump(payload, summary_file, ensure_ascii=False, indent=2)

        self.current_episode_path = None
        self.current_summary_path = None
        self.last_trace_snapshot = None

    def record(self, observation, obs_data, act_data, action, is_stochastic):
        if not self.trace_enabled:
            return

        self.current_mode = "predict" if is_stochastic else "exploit"
        self._flush_pending(observation)

        if not self._should_trace_mode(self.current_mode):
            return

        frame_no = observation["frame_state"]["frame_no"]
        if frame_no % self.trace_interval != 0:
            return

        selected_key = "sampled" if is_stochastic else "deterministic"
        policy_debug = act_data.policy_debug or {}
        feature_groups = obs_data.feature_groups or {}
        visible_context = obs_data.visible_context or {}
        feature_metadata = obs_data.feature_metadata or {}

        semantic_action = self._build_semantic_action(action, visible_context)
        action_constraints = self._build_action_constraints(observation, policy_debug, action, selected_key)
        policy_output = self._build_policy_output(policy_debug, action, act_data.value, selected_key)
        vision_summary = self._build_vision_summary(visible_context, feature_groups)
        decision_scores = self._build_decision_scores(vision_summary)
        reason_payload, reason_tags = self._build_reason_payload(
            decision_scores=decision_scores,
            vision_summary=vision_summary,
            semantic_action=semantic_action,
        )

        self.pending_record = {
            "episode_id": self.current_episode,
            "frame_no": frame_no,
            "env_id": observation.get("env_id"),
            "hero_camp": feature_metadata.get("main_camp", self.current_hero_camp),
            "mode": self.current_mode,
            "player_id": feature_metadata.get("player_id", self.current_player_id),
            "hero_id": feature_metadata.get("hero_id", self.current_hero_id),
            "visible_context": visible_context,
            "feature_groups": feature_groups,
            "vision_summary": vision_summary,
            "decision_scores": decision_scores,
            "action_constraints": action_constraints,
            "policy_output": policy_output,
            "semantic_action": semantic_action,
            "reason_tags": reason_tags,
            "reason_payload": reason_payload,
            "primary_reason": reason_payload["primary_reason"],
            "reason_text": reason_payload["reason_text_zh"],
            "executed_real_cmd": [],
        }
        self.last_trace_snapshot = {
            "frame_no": frame_no,
            "self_hp_rate": vision_summary["self_hp_rate"],
        }

    def _flush_pending(self, observation=None):
        if self.pending_record is None:
            return

        if observation is not None and DebugConfig.ATTACH_REAL_CMD:
            self.pending_record["executed_real_cmd"] = self._extract_real_cmd(observation)

        self._write_record(self.pending_record)
        self.pending_record = None

    def _write_record(self, record):
        if self.file_enabled and self.current_file is not None:
            self.current_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.current_file.flush()

        self._update_summary(record)
        self._update_window_stats(record)
        self._maybe_log_reason(record)
        self._report_monitor(force=False)

    def _update_summary(self, record):
        button_name = record["semantic_action"]["button"]
        target_name = record["semantic_action"]["target"]
        primary_reason = record["reason_payload"]["primary_reason"]

        self.summary["episodes"] = max(self.summary["episodes"], self.current_episode)
        self.summary["trace_frame_cnt"] += 1
        self.summary["button_count"][button_name] = self.summary["button_count"].get(button_name, 0) + 1
        self.summary["target_count"][target_name] = self.summary["target_count"].get(target_name, 0) + 1
        self.summary["reason_count"][primary_reason] = self.summary["reason_count"].get(primary_reason, 0) + 1
        self.summary["avg_state_value_sum"] += record["policy_output"]["state_value"]
        self.summary["avg_selected_button_prob_sum"] += record["policy_output"]["selected_button_prob"]
        self.summary["visible_enemy_hero_count"] += 1 if record["vision_summary"]["enemy_hero_visible"] else 0
        self.summary["visible_enemy_soldier_sum"] += record["vision_summary"]["visible_enemy_soldier_count"]
        for reason_name in PRIMARY_REASONS:
            self.summary["decision_score_sum"][reason_name] += record["decision_scores"]["{0}_score".format(reason_name)]
        if target_name.startswith("soldier"):
            self.summary["attack_soldier_cnt"] += 1
        if target_name == "enemy_hero":
            self.summary["attack_hero_cnt"] += 1
        if target_name == "tower":
            self.summary["attack_tower_cnt"] += 1

    def _update_window_stats(self, record):
        button_name = record["semantic_action"]["button"]
        target_name = record["semantic_action"]["target"]
        primary_reason = record["reason_payload"]["primary_reason"]

        self.window_stats["trace_frame_cnt"] += 1
        self.window_stats["state_value_sum"] += record["policy_output"]["state_value"]
        self.window_stats["selected_button_prob_sum"] += record["policy_output"]["selected_button_prob"]
        self.window_stats["visible_enemy_hero_count"] += 1 if record["vision_summary"]["enemy_hero_visible"] else 0
        self.window_stats["visible_enemy_soldier_sum"] += record["vision_summary"]["visible_enemy_soldier_count"]
        self.window_stats["button_count"][button_name] += 1
        self.window_stats["target_count"][target_name] += 1
        self.window_stats["reason_count"][primary_reason] += 1
        for reason_name in PRIMARY_REASONS:
            self.window_stats["decision_score_sum"][reason_name] += record["decision_scores"]["{0}_score".format(reason_name)]
        if target_name.startswith("soldier"):
            self.window_stats["attack_soldier_cnt"] += 1
        if target_name == "enemy_hero":
            self.window_stats["attack_hero_cnt"] += 1
        if target_name == "tower":
            self.window_stats["attack_tower_cnt"] += 1

    def _report_monitor(self, force):
        if not self.monitor_enabled or self.monitor is None:
            return
        if self.window_stats["trace_frame_cnt"] <= 0:
            return

        now = time.time()
        if not force and now - self.last_monitor_report_time < DebugConfig.TRACE_MONITOR_INTERVAL_SECONDS:
            return

        count = float(self.window_stats["trace_frame_cnt"])
        metrics = {
            "trace_frame_cnt": self.window_stats["trace_frame_cnt"],
            "avg_state_value": self.window_stats["state_value_sum"] / count,
            "avg_selected_button_prob": self.window_stats["selected_button_prob_sum"] / count,
            "visible_enemy_hero_rate": self.window_stats["visible_enemy_hero_count"] / count,
            "avg_visible_enemy_soldier_count": self.window_stats["visible_enemy_soldier_sum"] / count,
            "visible_enemy_soldier_avg": self.window_stats["visible_enemy_soldier_sum"] / count,
            "attack_soldier_rate": self.window_stats["attack_soldier_cnt"] / count,
            "attack_hero_rate": self.window_stats["attack_hero_cnt"] / count,
            "attack_tower_rate": self.window_stats["attack_tower_cnt"] / count,
        }
        for reason_name in PRIMARY_REASONS:
            metrics["{0}_rate".format(reason_name)] = self.window_stats["reason_count"].get(reason_name, 0) / count
            metrics["avg_{0}_score".format(reason_name)] = (
                self.window_stats["decision_score_sum"][reason_name] / count
            )

        for button_name in BUTTON_NAMES:
            metrics["button_count_{0}".format(button_name)] = self.window_stats["button_count"].get(button_name, 0)
        for target_name in TARGET_NAMES:
            metrics["target_count_{0}".format(target_name)] = self.window_stats["target_count"].get(target_name, 0)
        for reason_name in PRIMARY_REASONS:
            metrics["reason_count_{0}".format(reason_name)] = self.window_stats["reason_count"].get(reason_name, 0)

        self.monitor.put_data({os.getpid(): metrics})
        self.window_stats = self._new_window_stats()
        self.last_monitor_report_time = now

    def _build_action_constraints(self, observation, policy_debug, action, selected_key):
        legal_action = observation.get("legal_action", [])
        button_mask = list(legal_action[: len(BUTTON_NAMES)])
        legal_buttons = [BUTTON_NAMES[index] for index, flag in enumerate(button_mask) if flag]
        target_legal_mask = policy_debug.get("target_legal_mask", {}).get(selected_key, [])
        legal_targets = [TARGET_NAMES[index] for index, flag in enumerate(target_legal_mask) if flag]

        return {
            "legal_buttons": legal_buttons,
            "selected_button": BUTTON_NAMES[action[0]] if action else BUTTON_NAMES[0],
            "selected_button_target_mask": {
                "raw": target_legal_mask,
                "legal_targets": legal_targets,
            },
        }

    def _build_policy_output(self, policy_debug, action, value, selected_key):
        branch_probabilities = policy_debug.get("branch_probabilities", {})
        selected_probabilities = policy_debug.get("selected_probabilities", {}).get(selected_key, {})
        branch_topk = {}
        for branch_name in BRANCH_NAMES:
            label_names = self._labels_for_branch(branch_name)
            branch_probs = branch_probabilities.get(branch_name, [])
            if branch_name == "target" and isinstance(branch_probs, dict):
                branch_probs = branch_probs.get(selected_key, [])
            branch_topk[branch_name] = self._topk(branch_probs, label_names)

        return {
            "branch_topk": branch_topk,
            "selected_action_ids": list(action),
            "selected_button_prob": float(selected_probabilities.get("button", 0.0)),
            "selected_target_prob": float(selected_probabilities.get("target", 0.0)),
            "state_value": float(self._scalar_value(value)),
        }

    def _build_semantic_action(self, action, visible_context):
        button_name = BUTTON_NAMES[action[0]]
        target_name = TARGET_NAMES[action[5]]
        resolved_target = self._resolve_target_name(target_name, visible_context)
        return {
            "button": button_name,
            "button_index": int(action[0]),
            "target": target_name,
            "target_index": int(action[5]),
            "move_x": int(action[1]),
            "move_z": int(action[2]),
            "skill_x": int(action[3]),
            "skill_z": int(action[4]),
            "resolved_target": resolved_target,
            "description": "{0} -> {1}".format(button_name, resolved_target),
        }

    def _build_vision_summary(self, visible_context, feature_groups):
        main_hero = visible_context.get("main_hero", {})
        enemy_heroes = visible_context.get("enemy_heroes", [])
        enemy_towers = visible_context.get("enemy_towers", [])
        enemy_soldiers = visible_context.get("enemy_soldiers", [])
        friendly_soldiers = visible_context.get("friendly_soldiers", [])
        self_towers = visible_context.get("self_towers", [])

        enemy_hero = enemy_heroes[0] if enemy_heroes else None
        enemy_tower = enemy_towers[0] if enemy_towers else None
        self_tower = self_towers[0] if self_towers else None
        lowest_hp_enemy_soldier = self._lowest_hp_unit(enemy_soldiers)

        self_hp_rate = self._unit_hp_rate(main_hero)
        enemy_hero_visible = enemy_hero is not None
        enemy_tower_visible = enemy_tower is not None

        summary = {
            "self_hp_rate": self_hp_rate,
            "self_tower_dist": self._unit_dist_norm(self_tower, default=1.0),
            "enemy_hero_visible": enemy_hero_visible,
            "enemy_hero_hp_rate": self._unit_hp_rate(enemy_hero) if enemy_hero_visible else 0.0,
            "enemy_hero_dist": self._unit_dist_norm(enemy_hero, default=1.0),
            "enemy_hero_in_attack_range": float(
                self._safe_float(feature_groups.get("enemy_hero", {}).get("in_attack_range", 0.0))
            ),
            "enemy_tower_visible": enemy_tower_visible,
            "enemy_tower_hp_rate": self._unit_hp_rate(enemy_tower) if enemy_tower_visible else 0.0,
            "visible_enemy_soldier_count": len(enemy_soldiers),
            "visible_friendly_soldier_count": len(friendly_soldiers),
            "low_hp_enemy_soldier_count": sum(1 for unit in enemy_soldiers if self._unit_hp_rate(unit) <= 0.35),
            "nearest_enemy_soldier_dist": self._nearest_unit_dist(enemy_soldiers),
            "nearest_friendly_soldier_dist": self._nearest_unit_dist(friendly_soldiers),
            "hp_gap_vs_enemy": self._clip(
                self_hp_rate - (self._unit_hp_rate(enemy_hero) if enemy_hero_visible else 0.0),
                -1.0,
                1.0,
            ),
            "hp_loss_recent": 0.0,
            "lowest_hp_enemy_soldier": self._serialize_lowest_hp_unit(lowest_hp_enemy_soldier),
        }

        if self.last_trace_snapshot is not None:
            summary["hp_loss_recent"] = self._clip(
                self.last_trace_snapshot.get("self_hp_rate", self_hp_rate) - self_hp_rate,
                0.0,
                1.0,
            )

        return summary

    def _build_decision_scores(self, vision_summary):
        enemy_soldier_count = vision_summary["visible_enemy_soldier_count"]
        low_hp_enemy_soldier_count = vision_summary["low_hp_enemy_soldier_count"]
        enemy_hero_visible = vision_summary["enemy_hero_visible"]
        enemy_tower_visible = vision_summary["enemy_tower_visible"]
        self_hp_rate = vision_summary["self_hp_rate"]
        enemy_hero_hp_rate = vision_summary["enemy_hero_hp_rate"]
        enemy_hero_dist = vision_summary["enemy_hero_dist"]
        enemy_hero_in_attack_range = vision_summary["enemy_hero_in_attack_range"]
        enemy_tower_hp_rate = vision_summary["enemy_tower_hp_rate"]
        nearest_enemy_soldier_dist = vision_summary["nearest_enemy_soldier_dist"]
        friendly_soldier_count = vision_summary["visible_friendly_soldier_count"]
        self_tower_dist = vision_summary["self_tower_dist"]
        hp_loss_recent = vision_summary["hp_loss_recent"]

        soldier_density_score = self._clip01(enemy_soldier_count / 3.0)
        low_hp_soldier_ratio = low_hp_enemy_soldier_count / max(enemy_soldier_count, 1)
        nearest_soldier_score = 0.0 if enemy_soldier_count <= 0 else self._clip01(1.0 - nearest_enemy_soldier_dist)
        hero_hidden_bonus = 1.0 if not enemy_hero_visible else 0.0

        lane_clear_score = (
            0.35 * soldier_density_score
            + 0.25 * nearest_soldier_score
            + 0.25 * low_hp_soldier_ratio
            + 0.15 * hero_hidden_bonus
        )

        hero_pressure_score = 0.0
        if enemy_hero_visible:
            hp_advantage = self._clip01((self_hp_rate - enemy_hero_hp_rate + 1.0) / 2.0)
            hero_pressure_score = (
                0.35 * self._clip01(1.0 - enemy_hero_hp_rate)
                + 0.20 * hp_advantage
                + 0.25 * self._clip01(enemy_hero_in_attack_range)
                + 0.20 * self_hp_rate
            )

        tower_pressure_score = 0.0
        if enemy_tower_visible:
            frontline_support = self._clip01(friendly_soldier_count / 2.0)
            enemy_threat_relaxed = 1.0 if not enemy_hero_visible else self._clip01(enemy_hero_dist)
            tower_pressure_score = (
                0.35 * frontline_support
                + 0.25 * self._clip01(1.0 - enemy_tower_hp_rate)
                + 0.20 * enemy_threat_relaxed
                + 0.20 * self_hp_rate
            )

        retreat_score = (
            0.40 * self._clip01((0.65 - self_hp_rate) / 0.65)
            + 0.25 * self._clip01(hp_loss_recent * 4.0)
            + 0.20 * (self._clip01(1.0 - enemy_hero_dist) if enemy_hero_visible else 0.0)
            + 0.15 * self._clip01(1.0 - self_tower_dist)
        )

        return {
            "lane_clear_score": round(self._clip01(lane_clear_score), 4),
            "hero_pressure_score": round(self._clip01(hero_pressure_score), 4),
            "tower_pressure_score": round(self._clip01(tower_pressure_score), 4),
            "retreat_score": round(self._clip01(retreat_score), 4),
        }

    def _build_reason_payload(self, decision_scores, vision_summary, semantic_action):
        ordered_reasons = sorted(
            PRIMARY_REASONS,
            key=lambda name: (-decision_scores["{0}_score".format(name)], PRIMARY_REASONS.index(name)),
        )
        primary_reason = ordered_reasons[0]
        secondary_reason = ordered_reasons[1]
        reason_text_zh = self._build_reason_text_zh(
            primary_reason=primary_reason,
            secondary_reason=secondary_reason,
            vision_summary=vision_summary,
            semantic_action=semantic_action,
        )

        reason_tags = [
            "enemy_hero_visible" if vision_summary["enemy_hero_visible"] else "enemy_hero_hidden",
            "enemy_tower_visible" if vision_summary["enemy_tower_visible"] else "enemy_tower_hidden",
            "enemy_soldier_count_{0}".format(vision_summary["visible_enemy_soldier_count"]),
            "primary_{0}".format(primary_reason),
            "secondary_{0}".format(secondary_reason),
            "action_{0}".format(semantic_action["button"]),
        ]

        return (
            {
                "primary_reason": primary_reason,
                "secondary_reason": secondary_reason,
                "reason_text_zh": reason_text_zh,
            },
            reason_tags,
        )

    def _build_reason_text_zh(self, primary_reason, secondary_reason, vision_summary, semantic_action):
        action_desc = semantic_action["description"]
        secondary_label = PRIMARY_REASON_LABELS_ZH[secondary_reason]

        if primary_reason == "lane_clear":
            parts = []
            if vision_summary["enemy_hero_visible"]:
                parts.append("敌方英雄已进入视野，当前不适合盲目深追")
            else:
                parts.append("敌方英雄当前不在视野内")
            parts.append("前方可见{0}个敌方小兵".format(vision_summary["visible_enemy_soldier_count"]))
            if vision_summary["low_hp_enemy_soldier_count"] > 0:
                parts.append("其中{0}个残血可快速处理".format(vision_summary["low_hp_enemy_soldier_count"]))
            if vision_summary["nearest_enemy_soldier_dist"] < 1.0:
                parts.append(
                    "最近兵线距离较近(dist={0:.2f})".format(vision_summary["nearest_enemy_soldier_dist"])
                )
            return "，".join(parts) + "，因此优先清线；次要考虑为{0}，最终执行 {1}。".format(
                secondary_label,
                action_desc,
            )

        if primary_reason == "hero_pressure":
            parts = [
                "敌方英雄在视野内",
                "敌方血量比例{0:.2f}".format(vision_summary["enemy_hero_hp_rate"]),
                "我方血量比例{0:.2f}".format(vision_summary["self_hp_rate"]),
            ]
            if vision_summary["enemy_hero_in_attack_range"] >= 0.5:
                parts.append("目标已进入攻击范围")
            else:
                parts.append("目标距离为{0:.2f}".format(vision_summary["enemy_hero_dist"]))
            return "，".join(parts) + "，因此优先压人；次要考虑为{0}，最终执行 {1}。".format(
                secondary_label,
                action_desc,
            )

        if primary_reason == "tower_pressure":
            parts = [
                "敌方防御塔在视野内",
                "塔下附近有{0}个己方兵线可提供掩护".format(vision_summary["visible_friendly_soldier_count"]),
                "敌方塔血量比例{0:.2f}".format(vision_summary["enemy_tower_hp_rate"]),
            ]
            if vision_summary["enemy_hero_visible"] and vision_summary["enemy_hero_dist"] < 0.35:
                parts.append("但敌方英雄仍在近身威胁范围")
            else:
                parts.append("敌方英雄暂未形成近身威胁")
            return "，".join(parts) + "，因此优先推塔；次要考虑为{0}，最终执行 {1}。".format(
                secondary_label,
                action_desc,
            )

        parts = [
            "我方血量比例{0:.2f}".format(vision_summary["self_hp_rate"]),
            "最近一次 trace 生命下降{0:.2f}".format(vision_summary["hp_loss_recent"]),
            "己方塔距离{0:.2f}".format(vision_summary["self_tower_dist"]),
        ]
        if vision_summary["enemy_hero_visible"]:
            parts.append("敌方英雄逼近(dist={0:.2f})".format(vision_summary["enemy_hero_dist"]))
        else:
            parts.append("敌方英雄当前不在视野内")
        return "，".join(parts) + "，因此优先后撤；次要考虑为{0}，最终执行 {1}。".format(
            secondary_label,
            action_desc,
        )

    def _maybe_log_reason(self, record):
        if self.logger is None:
            return
        if record["frame_no"] % self.log_interval_frames != 0:
            return

        vision_summary = record["vision_summary"]
        reason_payload = record["reason_payload"]
        decision_scores = record["decision_scores"]
        semantic_action = record["semantic_action"]

        if vision_summary["enemy_hero_visible"]:
            hero_brief = "visible(hp={0:.2f}, dist={1:.2f})".format(
                vision_summary["enemy_hero_hp_rate"],
                vision_summary["enemy_hero_dist"],
            )
        else:
            hero_brief = "hidden"

        if vision_summary["enemy_tower_visible"]:
            tower_brief = "visible(hp={0:.2f})".format(vision_summary["enemy_tower_hp_rate"])
        else:
            tower_brief = "hidden"

        message = (
            "[DecisionTrace] frame={frame} mode={mode} primary={primary} secondary={secondary} "
            "scores(lane={lane:.2f}, hero={hero:.2f}, tower={tower:.2f}, retreat={retreat:.2f}) "
            "vision(hero={hero_brief}, enemy_soldiers={enemy_soldiers}, low_hp_soldiers={low_hp_soldiers}, "
            "tower={tower_brief}) action={action} reason={reason}"
        ).format(
            frame=record["frame_no"],
            mode=record["mode"],
            primary=PRIMARY_REASON_LABELS_ZH[reason_payload["primary_reason"]],
            secondary=PRIMARY_REASON_LABELS_ZH[reason_payload["secondary_reason"]],
            lane=decision_scores["lane_clear_score"],
            hero=decision_scores["hero_pressure_score"],
            tower=decision_scores["tower_pressure_score"],
            retreat=decision_scores["retreat_score"],
            hero_brief=hero_brief,
            enemy_soldiers=vision_summary["visible_enemy_soldier_count"],
            low_hp_soldiers=vision_summary["low_hp_enemy_soldier_count"],
            tower_brief=tower_brief,
            action=semantic_action["description"],
            reason=reason_payload["reason_text_zh"],
        )
        self.logger.info(message)

    def _resolve_target_name(self, target_name, visible_context):
        if target_name.startswith("soldier"):
            soldier_index = max(int(target_name.split("_")[-1]) - 1, 0)
            visible_soldiers = visible_context.get("enemy_soldiers", [])
            if soldier_index < len(visible_soldiers):
                runtime_id = visible_soldiers[soldier_index].get("runtime_id")
                return "{0}(runtime_id={1})".format(target_name, runtime_id)
        if target_name == "enemy_hero" and visible_context.get("enemy_heroes"):
            runtime_id = visible_context["enemy_heroes"][0].get("runtime_id")
            return "{0}(runtime_id={1})".format(target_name, runtime_id)
        if target_name == "tower" and visible_context.get("enemy_towers"):
            runtime_id = visible_context["enemy_towers"][0].get("runtime_id")
            return "{0}(runtime_id={1})".format(target_name, runtime_id)
        return target_name

    def _extract_real_cmd(self, observation):
        frame_state = observation.get("frame_state", {})
        player_id = observation.get("player_id")
        for hero in frame_state.get("hero_states", []):
            if hero.get("runtime_id") == player_id or hero.get("player_id") == player_id:
                return hero.get("real_cmd", [])
        return []

    def _extract_hero_id(self, observation):
        frame_state = observation.get("frame_state", {})
        player_id = observation.get("player_id")
        for hero in frame_state.get("hero_states", []):
            if hero.get("runtime_id") == player_id or hero.get("player_id") == player_id:
                return hero.get("config_id")
        return "unknown"

    def _should_trace_mode(self, mode):
        trace_mode = str(DebugConfig.TRACE_MODE).lower()
        if trace_mode == "exploit_only":
            return mode == "exploit"
        if trace_mode == "predict_only":
            return mode == "predict"
        return True

    def _labels_for_branch(self, branch_name):
        if branch_name == "button":
            return BUTTON_NAMES
        if branch_name == "target":
            return TARGET_NAMES
        return ["{0}_{1}".format(branch_name, index) for index in range(16)]

    def _topk(self, probabilities, labels):
        indexed = list(enumerate(probabilities))
        indexed.sort(key=lambda item: item[1], reverse=True)
        topk = []
        for index, prob in indexed[: self.trace_topk]:
            topk.append(
                {
                    "index": int(index),
                    "label": labels[index] if index < len(labels) else str(index),
                    "prob": float(prob),
                }
            )
        return topk

    def _scalar_value(self, value):
        if isinstance(value, (float, int)):
            return value
        if hasattr(value, "reshape"):
            flattened = value.reshape(-1)
            if len(flattened) > 0:
                return flattened[0]
        if isinstance(value, (list, tuple)) and value:
            return value[0]
        return 0.0

    def _load_summary(self, summary_path):
        if not os.path.exists(summary_path):
            return self._new_summary()
        with open(summary_path, "r", encoding="utf-8") as summary_file:
            payload = json.load(summary_file)
        return self._merge_with_default(self._new_summary(), payload)

    def _merge_with_default(self, default_payload, loaded_payload):
        merged = {}
        for key, default_value in default_payload.items():
            if key not in loaded_payload:
                merged[key] = default_value
                continue
            loaded_value = loaded_payload[key]
            if isinstance(default_value, dict) and isinstance(loaded_value, dict):
                merged[key] = self._merge_with_default(default_value, loaded_value)
            else:
                merged[key] = loaded_value
        for key, loaded_value in loaded_payload.items():
            if key not in merged:
                merged[key] = loaded_value
        return merged

    def _new_summary(self):
        return {
            "episodes": 0,
            "trace_frame_cnt": 0,
            "button_count": {},
            "target_count": {},
            "reason_count": {},
            "decision_score_sum": {reason_name: 0.0 for reason_name in PRIMARY_REASONS},
            "avg_state_value_sum": 0.0,
            "avg_selected_button_prob_sum": 0.0,
            "visible_enemy_hero_count": 0,
            "visible_enemy_soldier_sum": 0,
            "attack_soldier_cnt": 0,
            "attack_hero_cnt": 0,
            "attack_tower_cnt": 0,
            "last_updated": "",
        }

    def _new_window_stats(self):
        return {
            "trace_frame_cnt": 0,
            "state_value_sum": 0.0,
            "selected_button_prob_sum": 0.0,
            "visible_enemy_hero_count": 0,
            "visible_enemy_soldier_sum": 0,
            "button_count": Counter(),
            "target_count": Counter(),
            "reason_count": Counter(),
            "decision_score_sum": {reason_name: 0.0 for reason_name in PRIMARY_REASONS},
            "attack_soldier_cnt": 0,
            "attack_hero_cnt": 0,
            "attack_tower_cnt": 0,
        }

    def _serialize_lowest_hp_unit(self, unit):
        if unit is None:
            return None
        return {
            "runtime_id": unit.get("runtime_id"),
            "hp_rate": round(self._unit_hp_rate(unit), 4),
            "dist_norm": round(self._unit_dist_norm(unit, default=1.0), 4),
        }

    def _lowest_hp_unit(self, units):
        if not units:
            return None
        return min(units, key=lambda unit: (self._unit_hp_rate(unit), self._unit_dist_norm(unit, default=1.0)))

    def _nearest_unit_dist(self, units):
        if not units:
            return 1.0
        return min(self._unit_dist_norm(unit, default=1.0) for unit in units)

    def _unit_hp_rate(self, unit):
        if unit is None:
            return 0.0
        max_hp = max(float(unit.get("max_hp", 0)), 1.0)
        return self._clip01(float(unit.get("hp", 0)) / max_hp)

    def _unit_dist_norm(self, unit, default=1.0):
        if unit is None:
            return default
        return self._clip01(self._safe_float(unit.get("dist_norm", default)))

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _clip(self, value, lower, upper):
        return max(lower, min(upper, float(value)))

    def _clip01(self, value):
        return self._clip(value, 0.0, 1.0)
