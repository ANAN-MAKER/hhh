#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from agent_ppo.conf.conf import GameConfig


REWARD_KEYS = [
    "terminal_reward",
    "pushing_reward",
    "combat_reward",
    "farming_reward",
    "tempo_reward",
    "reward_sum",
]


class GameRewardManager:
    def __init__(self, main_hero_runtime_id: int) -> None:
        self.main_hero_runtime_id = main_hero_runtime_id
        self.main_hero_camp: Optional[int] = None
        self.prev_snapshot: Optional[Dict[str, Any]] = None

    def result(self, frame_data: Dict[str, Any], terminated: bool = False, truncated: bool = False, win: Optional[bool] = None) -> Dict[str, float]:
        snapshot = self._extract_snapshot(frame_data)
        reward_dict = self._empty_reward_dict()

        if self.prev_snapshot is not None:
            reward_dict["pushing_reward"] = self._calc_pushing_reward(snapshot, self.prev_snapshot)
            reward_dict["combat_reward"] = self._calc_combat_reward(snapshot, self.prev_snapshot)
            reward_dict["farming_reward"] = self._calc_farming_reward(snapshot, self.prev_snapshot)
            reward_dict["tempo_reward"] = self._calc_tempo_reward(snapshot, self.prev_snapshot)

        terminal = self._calc_terminal_reward(
            snapshot=snapshot,
            terminated=terminated,
            truncated=truncated,
            win=win,
        )
        reward_dict["terminal_reward"] = terminal

        dense = (
            reward_dict["pushing_reward"]
            + reward_dict["combat_reward"]
            + reward_dict["farming_reward"]
            + reward_dict["tempo_reward"]
        )
        dense = self._clip(dense, -1.0, 1.0)
        reward_dict["reward_sum"] = self._clip(dense + terminal, -5.0, 5.0)

        self.prev_snapshot = snapshot
        return reward_dict

    def _extract_snapshot(self, frame_data):
        main_hero = None
        enemy_hero = None
        for hero in frame_data.get("hero_states", []):
            if hero.get("runtime_id") == self.main_hero_runtime_id:
                main_hero = hero
                self.main_hero_camp = hero.get("camp")
            elif self.main_hero_camp is None or hero.get("camp") != self.main_hero_camp:
                enemy_hero = hero

        if main_hero is None:
            raise RuntimeError("Unable to locate the controlled hero in reward_process")

        if enemy_hero is None:
            for hero in frame_data.get("hero_states", []):
                if hero.get("runtime_id") != self.main_hero_runtime_id:
                    enemy_hero = hero
                    break

        self_tower = None
        enemy_tower = None
        for npc in frame_data.get("npc_states", []):
            if not self._is_normal_tower(npc):
                continue
            if npc.get("camp") == main_hero.get("camp"):
                self_tower = npc
            else:
                enemy_tower = npc

        self_tower_hp_rate = self._hp_rate(self_tower)
        enemy_tower_hp_rate = self._hp_rate(enemy_tower)
        forward_progress = self._calc_forward_progress(main_hero, self_tower, enemy_tower)

        return {
            "frame_no": frame_data.get("frame_no", 0),
            "main_camp": main_hero.get("camp"),
            "hp_rate": self._hp_rate(main_hero),
            "money": float(main_hero.get("money", main_hero.get("money_cnt", 0))),
            "exp": float(main_hero.get("exp", 0)),
            "level": float(main_hero.get("level", 0)),
            "kill_cnt": float(main_hero.get("kill_cnt", 0)),
            "dead_cnt": float(main_hero.get("dead_cnt", 0)),
            "total_hurt_to_hero": float(main_hero.get("total_hurt_to_hero", 0)),
            "total_be_hurt_by_hero": float(main_hero.get("total_be_hurt_by_hero", 0)),
            "self_tower_hp_rate": self_tower_hp_rate,
            "enemy_tower_hp_rate": enemy_tower_hp_rate,
            "self_tower_hp": float(self_tower.get("hp", 0)) if self_tower else 0.0,
            "enemy_tower_hp": float(enemy_tower.get("hp", 0)) if enemy_tower else 0.0,
            "forward_progress": forward_progress,
        }

    def _calc_terminal_reward(self, snapshot, terminated, truncated, win):
        if truncated:
            return 0.0
        if not terminated:
            return 0.0
        if win is not None:
            return 5.0 if bool(win) else -5.0
        if snapshot["enemy_tower_hp"] <= 0 < snapshot["self_tower_hp"]:
            return 5.0
        if snapshot["self_tower_hp"] <= 0 < snapshot["enemy_tower_hp"]:
            return -5.0
        return 0.0

    def _calc_pushing_reward(self, snapshot, prev_snapshot):
        enemy_tower_delta = prev_snapshot["enemy_tower_hp_rate"] - snapshot["enemy_tower_hp_rate"]
        self_tower_delta = prev_snapshot["self_tower_hp_rate"] - snapshot["self_tower_hp_rate"]
        return 3.0 * enemy_tower_delta - 3.0 * self_tower_delta

    def _calc_combat_reward(self, snapshot, prev_snapshot):
        kill_delta = snapshot["kill_cnt"] - prev_snapshot["kill_cnt"]
        dead_delta = snapshot["dead_cnt"] - prev_snapshot["dead_cnt"]
        hurt_to_hero_delta = snapshot["total_hurt_to_hero"] - prev_snapshot["total_hurt_to_hero"]
        be_hurt_by_hero_delta = snapshot["total_be_hurt_by_hero"] - prev_snapshot["total_be_hurt_by_hero"]
        return (
            1.0 * kill_delta
            - 1.0 * dead_delta
            + 0.001 * hurt_to_hero_delta
            - 0.0015 * be_hurt_by_hero_delta
        )

    def _calc_farming_reward(self, snapshot, prev_snapshot):
        money_delta = snapshot["money"] - prev_snapshot["money"]
        exp_delta = snapshot["exp"] - prev_snapshot["exp"]
        return 0.002 * money_delta + 0.002 * exp_delta

    def _calc_tempo_reward(self, snapshot, prev_snapshot):
        if snapshot["hp_rate"] <= 0.6:
            return 0.0
        forward_progress_delta = snapshot["forward_progress"] - prev_snapshot["forward_progress"]
        if abs(forward_progress_delta) > 0.5:
            return 0.0
        return 0.02 * forward_progress_delta

    def _calc_forward_progress(self, main_hero, self_tower, enemy_tower):
        if main_hero is None or self_tower is None or enemy_tower is None:
            return 0.0

        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        self_tower_pos = (self_tower["location"]["x"], self_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])

        lane_len = math.dist(self_tower_pos, enemy_tower_pos)
        if lane_len <= 0:
            return 0.0
        hero_to_enemy = math.dist(hero_pos, enemy_tower_pos)
        progress = (lane_len - hero_to_enemy) / lane_len
        return self._clip(progress, -1.0, 1.0)

    def _hp_rate(self, unit):
        if unit is None:
            return 0.0
        max_hp = max(float(unit.get("max_hp", 0)), 1.0)
        return self._clip(float(unit.get("hp", 0)) / max_hp, 0.0, 1.0)

    def _is_normal_tower(self, npc):
        sub_type = npc.get("sub_type")
        config_id = self._safe_int(npc.get("config_id"))
        if config_id in GameConfig.TOWER_CONFIG_IDS:
            return True
        if isinstance(sub_type, int):
            return sub_type == GameConfig.SUB_TYPE_TOWER
        if isinstance(sub_type, str):
            return "TOWER" in sub_type and "SPRING" not in sub_type
        return False

    def _safe_int(self, value, default=-1):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _empty_reward_dict(self):
        return {key: 0.0 for key in REWARD_KEYS}

    def _clip(self, value, lower, upper):
        return max(lower, min(upper, float(value)))
