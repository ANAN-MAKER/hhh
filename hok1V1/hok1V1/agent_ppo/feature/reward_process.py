#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math
from agent_ppo.conf.conf import GameConfig


class GameRewardManager:
    def __init__(self, main_hero_player_id):
        self.main_hero_player_id = main_hero_player_id
        self.prev_stats = None
        self.reward_weights = GameConfig.REWARD_WEIGHT_DICT
        self.last_enemy_creep_ids = None

    def result(self, frame_data, action=None):
        cur_stats = self._extract_stats(frame_data, action)
        if self.prev_stats is None:
            self.prev_stats = cur_stats
            return self._zero_reward()

        reward_items = self._compute_reward_items(self.prev_stats, cur_stats)
        reward_sum = 0.0
        for name, value in reward_items.items():
            reward_sum += value * self.reward_weights.get(name, 0.0)

        self.prev_stats = cur_stats
        reward_items["reward_sum"] = self._clip(reward_sum, -100.0, 100.0)
        reward_items.update(self._legacy_aliases(reward_items))
        reward_items.update({f"{name}_weight": weight for name, weight in self.reward_weights.items()})
        reward_items["reward_stage"] = GameConfig.REWARD_STAGE
        reward_items["enable_tower_danger_reward"] = float(GameConfig.ENABLE_TOWER_DANGER_REWARD)
        reward_items["enable_push_assist_reward"] = float(GameConfig.ENABLE_PUSH_ASSIST_REWARD)
        reward_items["enable_lane_follow_reward"] = float(GameConfig.ENABLE_LANE_FOLLOW_REWARD)
        reward_items["enable_finish_reward"] = float(GameConfig.ENABLE_FINISH_REWARD)
        reward_items["enable_skill_reward"] = float(GameConfig.ENABLE_SKILL_REWARD)
        return reward_items

    def _zero_reward(self):
        reward_items = {name: 0.0 for name in self.reward_weights}
        reward_items["reward_sum"] = 0.0
        reward_items.update(self._legacy_aliases(reward_items))
        reward_items.update({f"{name}_weight": weight for name, weight in self.reward_weights.items()})
        reward_items["reward_stage"] = GameConfig.REWARD_STAGE
        reward_items["enable_tower_danger_reward"] = float(GameConfig.ENABLE_TOWER_DANGER_REWARD)
        reward_items["enable_push_assist_reward"] = float(GameConfig.ENABLE_PUSH_ASSIST_REWARD)
        reward_items["enable_lane_follow_reward"] = float(GameConfig.ENABLE_LANE_FOLLOW_REWARD)
        reward_items["enable_finish_reward"] = float(GameConfig.ENABLE_FINISH_REWARD)
        reward_items["enable_skill_reward"] = float(GameConfig.ENABLE_SKILL_REWARD)
        return reward_items

    def _compute_reward_items(self, prev, cur):
        enemy_hp_drop = prev["enemy_hp_ratio"] - cur["enemy_hp_ratio"]
        self_hp_drop = prev["self_hp_ratio"] - cur["self_hp_ratio"]
        enemy_tower_drop = prev["enemy_tower_hp_ratio"] - cur["enemy_tower_hp_ratio"]
        self_tower_drop = prev["self_tower_hp_ratio"] - cur["self_tower_hp_ratio"]

        reward_items = {
            "reward_hp": self._clip(enemy_hp_drop - self_hp_drop, -1.0, 1.0),
            "reward_tower": self._clip(enemy_tower_drop - self_tower_drop, -1.0, 1.0),
            "reward_gold": self._clip((cur["self_money"] - prev["self_money"]) - (cur["enemy_money"] - prev["enemy_money"]), -200.0, 200.0),
            "reward_exp": self._clip((cur["self_exp"] - prev["self_exp"]) - (cur["enemy_exp"] - prev["enemy_exp"]), -200.0, 200.0),
            "reward_last_hit": self._clip(cur["last_hit_delta"], 0.0, 4.0),
            "reward_death": self._clip(cur["self_dead_cnt"] - prev["self_dead_cnt"], 0.0, 1.0),
            "reward_kill": self._clip(cur["self_kill_cnt"] - prev["self_kill_cnt"], 0.0, 1.0),
            "reward_forward": self._clip(cur["lane_progress"] - prev["lane_progress"], -1.0, 1.0),
            "reward_win": self._win_reward(cur),
            "reward_tower_danger": 0.0,
            "reward_push_assist": 0.0,
            "reward_lane_follow": 0.0,
            "reward_finish": 0.0,
        }

        if GameConfig.ENABLE_TOWER_DANGER_REWARD:
            reward_items["reward_tower_danger"] = self._tower_danger_reward(cur)
        if GameConfig.ENABLE_PUSH_ASSIST_REWARD:
            reward_items["reward_push_assist"] = self._push_assist_reward(cur)
        if GameConfig.ENABLE_LANE_FOLLOW_REWARD:
            reward_items["reward_lane_follow"] = self._lane_follow_reward(prev, cur)
        if GameConfig.ENABLE_FINISH_REWARD:
            reward_items["reward_finish"] = self._finish_reward(cur)
        return reward_items

    def _extract_stats(self, frame_data, action=None):
        main_hero, enemy_hero = self._split_heroes(frame_data)
        main_camp = main_hero.get("camp", -1) if main_hero else -1
        main_tower = self._get_tower(frame_data, main_camp)
        enemy_tower = self._get_enemy_tower(frame_data, main_camp)
        my_soldiers = self._soldiers(frame_data, main_camp)
        enemy_soldiers = self._enemy_soldiers(frame_data, main_camp)
        enemy_creep_ids = {self._unit_id(creep) for creep in enemy_soldiers}
        enemy_creep_ids.discard(None)
        if self.last_enemy_creep_ids is None:
            last_hit_delta = 0.0
        else:
            last_hit_delta = float(len(self.last_enemy_creep_ids - enemy_creep_ids))
        self.last_enemy_creep_ids = enemy_creep_ids

        return {
            "self_hp_ratio": self._hp_ratio(main_hero),
            "enemy_hp_ratio": self._hp_ratio(enemy_hero),
            "self_tower_hp_ratio": self._hp_ratio(main_tower),
            "enemy_tower_hp_ratio": self._hp_ratio(enemy_tower),
            "self_money": self._resource(main_hero, "money"),
            "enemy_money": self._resource(enemy_hero, "money"),
            "self_exp": self._resource(main_hero, "exp"),
            "enemy_exp": self._resource(enemy_hero, "exp"),
            "self_dead_cnt": self._value(main_hero, "dead_cnt"),
            "self_kill_cnt": self._value(main_hero, "kill_cnt"),
            "lane_progress": self._lane_progress(main_hero, main_tower, enemy_tower),
            "last_hit_delta": last_hit_delta,
            "self_tower_dead": main_tower is not None and self._value(main_tower, "hp") <= 0,
            "enemy_tower_dead": enemy_tower is not None and self._value(enemy_tower, "hp") <= 0,
            "self_in_enemy_tower_range": self._in_tower_range(main_hero, enemy_tower),
            "has_ally_minion_under_enemy_tower": self._has_ally_minion_under_enemy_tower(my_soldiers, enemy_tower),
            "enemy_low_hp": self._hp_ratio(enemy_hero) < 0.30,
            "safe_to_push": self._safe_to_push(main_hero, enemy_hero, my_soldiers, enemy_tower),
            "ally_minion_ahead": self._ally_minion_ahead(main_hero, my_soldiers, main_tower, enemy_tower),
            "enemy_close_dangerous": self._enemy_close_dangerous(main_hero, enemy_hero),
            "enemy_visible": enemy_hero is not None and self._hp_ratio(enemy_hero) > 0.0,
            "action_button": self._action_button(action),
            "action_target": self._action_target(action),
        }

    def _legacy_aliases(self, reward_items):
        return {
            "hp_point": reward_items.get("reward_hp", 0.0),
            "tower_hp_point": reward_items.get("reward_tower", 0.0),
            "gold_point": reward_items.get("reward_gold", 0.0),
            "exp_point": reward_items.get("reward_exp", 0.0),
            "last_hit_point": reward_items.get("reward_last_hit", 0.0),
            "death_point": reward_items.get("reward_death", 0.0),
            "kill_point": reward_items.get("reward_kill", 0.0),
            "forward": reward_items.get("reward_forward", 0.0),
            "push_assist": reward_items.get("reward_push_assist", 0.0),
            "lane_follow": reward_items.get("reward_lane_follow", 0.0),
            "finish": reward_items.get("reward_finish", 0.0),
        }

    def _win_reward(self, cur):
        if cur["enemy_tower_dead"] and not cur["self_tower_dead"]:
            return 1.0
        if cur["self_tower_dead"] and not cur["enemy_tower_dead"]:
            return -1.0
        return 0.0

    def _tower_danger_reward(self, cur):
        if cur["self_in_enemy_tower_range"] and not cur["has_ally_minion_under_enemy_tower"] and not cur["enemy_low_hp"]:
            return -0.03
        return 0.0

    def _push_assist_reward(self, cur):
        if cur["safe_to_push"] and cur["action_target"] == GameConfig.TARGET_TOWER:
            return 0.02
        return 0.0

    def _lane_follow_reward(self, prev, cur):
        moved_forward = cur["lane_progress"] > prev["lane_progress"] + 0.001
        if (
            cur["action_button"] == GameConfig.BUTTON_MOVE
            and moved_forward
            and cur["ally_minion_ahead"]
            and not cur["enemy_close_dangerous"]
        ):
            return 0.01
        return 0.0

    def _finish_reward(self, cur):
        if (
            cur["enemy_visible"]
            and cur["enemy_low_hp"]
            and cur["action_target"] == GameConfig.TARGET_ENEMY
            and cur["action_button"] in (GameConfig.BUTTON_ATTACK, GameConfig.BUTTON_SUMMONER, *GameConfig.SKILL_BUTTONS)
        ):
            return 0.02
        return 0.0

    def _split_heroes(self, frame_data):
        heroes = frame_data.get("hero_states", [])
        main_hero = None
        for hero in heroes:
            if hero.get("player_id") == self.main_hero_player_id or hero.get("runtime_id") == self.main_hero_player_id:
                main_hero = hero
                break
        if main_hero is None and heroes:
            main_hero = heroes[0]
        enemy_hero = None
        if main_hero is not None:
            for hero in heroes:
                if hero is not main_hero and hero.get("camp") != main_hero.get("camp"):
                    enemy_hero = hero
                    break
        return main_hero, enemy_hero

    def _lane_progress(self, hero, main_tower, enemy_tower):
        if not hero or not main_tower or not enemy_tower:
            return 0.0
        start = self._position(main_tower)
        end = self._position(enemy_tower)
        point = self._position(hero)
        vx, vz = end[0] - start[0], end[1] - start[1]
        lane_len_sq = max(vx * vx + vz * vz, 1.0)
        progress = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vz) / lane_len_sq
        if self._hp_ratio(hero) < 0.25:
            progress *= 0.3
        return self._clip(progress, 0.0, 1.0)

    def _in_tower_range(self, hero, tower):
        if not hero or not tower:
            return False
        tower_range = self._value(tower, "attack_range") or 9000.0
        return math.dist(self._position(hero), self._position(tower)) <= tower_range

    def _has_ally_minion_under_enemy_tower(self, soldiers, enemy_tower):
        if not enemy_tower:
            return False
        tower_range = self._value(enemy_tower, "attack_range") or 9000.0
        tower_pos = self._position(enemy_tower)
        return any(math.dist(self._position(soldier), tower_pos) <= tower_range for soldier in soldiers)

    def _safe_to_push(self, main_hero, enemy_hero, soldiers, enemy_tower):
        return (
            self._has_ally_minion_under_enemy_tower(soldiers, enemy_tower)
            and self._hp_ratio(main_hero) >= 0.30
            and not self._enemy_close_dangerous(main_hero, enemy_hero)
        )

    def _ally_minion_ahead(self, main_hero, soldiers, main_tower, enemy_tower):
        if not main_hero or not main_tower or not enemy_tower:
            return False
        hero_progress = self._lane_progress(main_hero, main_tower, enemy_tower)
        for soldier in soldiers:
            if self._lane_progress(soldier, main_tower, enemy_tower) > hero_progress + 0.03:
                return True
        return False

    def _enemy_close_dangerous(self, main_hero, enemy_hero):
        if not main_hero or not enemy_hero or self._hp_ratio(enemy_hero) <= 0.0:
            return False
        return math.dist(self._position(main_hero), self._position(enemy_hero)) <= 12000.0 and self._hp_ratio(enemy_hero) > 0.25

    def _action_button(self, action):
        if action is None or len(action) <= 0:
            return -1
        return int(action[0])

    def _action_target(self, action):
        if action is None or len(action) <= 5:
            return -1
        return int(action[5])

    def _get_tower(self, frame_data, camp):
        for organ in frame_data.get("npc_states", []):
            if organ.get("camp") == camp and int(self._value(organ, "sub_type")) in GameConfig.TOWER_SUB_TYPES:
                return organ
        return None

    def _get_enemy_tower(self, frame_data, camp):
        for organ in frame_data.get("npc_states", []):
            if organ.get("camp") != camp and int(self._value(organ, "sub_type")) in GameConfig.TOWER_SUB_TYPES:
                return organ
        return None

    def _soldiers(self, frame_data, camp):
        return [
            npc
            for npc in frame_data.get("npc_states", [])
            if isinstance(npc, dict)
            and npc.get("camp") == camp
            and not self._is_organ(npc)
            and self._value(npc, "hp") > 0
        ]

    def _enemy_soldiers(self, frame_data, camp):
        return [
            npc
            for npc in frame_data.get("npc_states", [])
            if isinstance(npc, dict)
            and npc.get("camp") != camp
            and not self._is_organ(npc)
            and self._value(npc, "hp") > 0
        ]

    def _is_organ(self, unit):
        sub_type = int(self._value(unit, "sub_type"))
        return sub_type in GameConfig.TOWER_SUB_TYPES or sub_type in GameConfig.CRYSTAL_SUB_TYPES

    def _hp_ratio(self, unit):
        max_hp = self._value(unit, "max_hp")
        if max_hp <= 0:
            return 0.0
        return self._clip(self._value(unit, "hp") / max_hp, 0.0, 1.0)

    def _position(self, unit):
        if not unit:
            return 0.0, 0.0
        location = unit.get("location", {})
        return float(location.get("x", 0.0) or 0.0), float(location.get("z", 0.0) or 0.0)

    def _unit_id(self, unit):
        if not unit:
            return None
        for key in ("runtime_id", "actor_runtime_id", "obj_id"):
            value = unit.get(key)
            if value is not None:
                return str(value)
        location = unit.get("location", {})
        return (
            f"{unit.get('camp')}_{unit.get('sub_type')}_{unit.get('config_id')}_"
            f"{int(location.get('x', 0) or 0)}_{int(location.get('z', 0) or 0)}"
        )

    def _resource(self, unit, key):
        if key == "money":
            return self._value_any(unit, ("money", "gold", "money_cnt"))
        return self._value(unit, key)

    def _value_any(self, unit, keys):
        for key in keys:
            value = self._value(unit, key)
            if value:
                return value
        return 0.0

    def _value(self, unit, key):
        if not unit:
            return 0.0
        return float(unit.get(key, 0.0) or 0.0)

    def _clip(self, value, min_value, max_value):
        return max(min_value, min(max_value, float(value)))
