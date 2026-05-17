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
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from agent_ppo.conf.conf import DimConfig, GameConfig, NormConfig


class BattlefieldFeatureProcessor:
    def __init__(self, camp: str) -> None:
        self.main_camp = camp
        self._mirror = self._camp_token(camp) == "2"

    def build(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        frame_state = observation["frame_state"]
        legal_action = observation.get("legal_action", [])

        main_hero = self._find_main_hero(frame_state, observation)
        if main_hero is None:
            raise RuntimeError("Unable to locate the controlled hero in frame_state")

        enemy_hero = self._find_enemy_hero(frame_state, main_hero)
        npcs = frame_state.get("npc_states", [])

        self_towers = self._collect_towers(npcs, main_hero, same_camp=True)
        enemy_towers = self._collect_towers(npcs, main_hero, same_camp=False)
        enemy_soldiers = self._collect_soldiers(npcs, main_hero, same_camp=False)
        friendly_soldiers = self._collect_soldiers(npcs, main_hero, same_camp=True)

        self_tower = self_towers[0] if self_towers else None
        enemy_tower = enemy_towers[0] if enemy_towers else None

        feature_groups = OrderedDict()
        feature_groups["self_hero"] = self._build_self_hero_group(main_hero)
        feature_groups["enemy_hero"] = self._build_enemy_hero_group(main_hero, enemy_hero)
        feature_groups["self_tower"] = self._build_self_tower_group(main_hero, self_tower)
        feature_groups["enemy_tower"] = self._build_enemy_tower_group(main_hero, enemy_tower)
        feature_groups["enemy_soldiers"] = self._build_soldier_slots(enemy_soldiers)
        feature_groups["friendly_soldiers"] = self._build_soldier_slots(friendly_soldiers)
        feature_groups["global_context"] = self._build_global_context(
            frame_state=frame_state,
            main_hero=main_hero,
            enemy_hero=enemy_hero,
            self_tower=self_tower,
            enemy_tower=enemy_tower,
        )
        feature_groups["matchup"] = self._build_matchup_group(main_hero, enemy_hero)

        feature_vector = self._flatten_feature_groups(feature_groups)

        visible_context = {
            "main_hero": self._serialize_unit(main_hero, main_hero, force_visible=True),
            "enemy_heroes": (
                [self._serialize_unit(enemy_hero, main_hero, force_visible=False)]
                if enemy_hero is not None and self._is_visible_to_main(enemy_hero, main_hero)
                else []
            ),
            "self_towers": [self._serialize_unit(unit, main_hero, force_visible=True) for unit in self_towers[:1]],
            "enemy_towers": [self._serialize_unit(unit, main_hero, force_visible=False) for unit in enemy_towers[:1]],
            "enemy_soldiers": [self._serialize_unit(unit, main_hero, force_visible=False) for unit in enemy_soldiers],
            "friendly_soldiers": [self._serialize_unit(unit, main_hero, force_visible=True) for unit in friendly_soldiers],
        }

        hero_id = int(main_hero.get("config_id", -1))
        feature_metadata = {
            "hero_id": hero_id,
            "hero_id_index": GameConfig.HERO_ID_TO_INDEX.get(hero_id, -1),
            "player_id": observation.get("player_id"),
            "main_camp": main_hero.get("camp", self.main_camp),
            "mirrored_to_blue_side": self._mirror,
            "visible_enemy_hero": 1 if visible_context["enemy_heroes"] else 0,
            "visible_enemy_soldier_count": len(visible_context["enemy_soldiers"]),
            "visible_enemy_tower_count": len(visible_context["enemy_towers"]),
            "enemy_soldier_target_slots": [unit.get("runtime_id") for unit in enemy_soldiers],
        }

        return {
            "feature_vector": feature_vector,
            "feature_groups": self._jsonable_feature_groups(feature_groups),
            "visible_context": visible_context,
            "feature_metadata": feature_metadata,
        }

    def _flatten_feature_groups(self, feature_groups):
        vector = []
        for group_name in (
            "self_hero",
            "enemy_hero",
            "self_tower",
            "enemy_tower",
            "enemy_soldiers",
            "friendly_soldiers",
            "global_context",
            "matchup",
        ):
            group_value = feature_groups[group_name]
            if isinstance(group_value, list):
                for slot in group_value:
                    vector.extend(slot.values())
            else:
                vector.extend(group_value.values())
        if len(vector) != DimConfig.DIM_OF_FEATURE[0]:
            raise RuntimeError(
                "Feature dim mismatch: expect {0}, got {1}".format(
                    DimConfig.DIM_OF_FEATURE[0],
                    len(vector),
                )
            )
        return vector

    def _jsonable_feature_groups(self, feature_groups):
        payload = OrderedDict()
        for key, value in feature_groups.items():
            if isinstance(value, list):
                payload[key] = [dict(slot) for slot in value]
            else:
                payload[key] = dict(value)
        return payload

    def _build_self_hero_group(self, hero):
        self_skill_state = self._build_skill_state_features(hero)
        return OrderedDict(
            [
                ("alive", self._alive_flag(hero)),
                ("hp_rate", self._hp_rate(hero)),
                ("ep_rate", self._ep_rate(hero)),
                ("level_norm", self._normalize_level(hero.get("level", 0))),
                ("money_norm", self._clip01(float(hero.get("money", hero.get("money_cnt", 0))) / NormConfig.MAX_MONEY)),
                ("pos_x_norm", self._normalize_abs_pos(hero["location"]["x"])),
                ("pos_z_norm", self._normalize_abs_pos(hero["location"]["z"])),
                ("attack_range_norm", self._clip01(float(hero.get("attack_range", 0)) / NormConfig.MAX_ATTACK_RANGE)),
                ("mov_spd_norm", self._clip01(float(hero.get("mov_spd", 0)) / NormConfig.MAX_MOVE_SPEED)),
                ("atk_spd_norm", self._clip01(float(hero.get("atk_spd", 0)) / NormConfig.MAX_ATK_SPEED)),
                ("is_in_grass", 1.0 if hero.get("is_in_grass", False) else 0.0),
                ("kill_cnt_norm", self._clip01(float(hero.get("kill_cnt", 0)) / NormConfig.MAX_KDA_COUNT)),
                ("dead_cnt_norm", self._clip01(float(hero.get("dead_cnt", 0)) / NormConfig.MAX_KDA_COUNT)),
                ("phy_atk_norm", self._clip01(float(hero.get("phy_atk", 0)) / NormConfig.MAX_PHY_ATK)),
                ("mgc_atk_norm", self._clip01(float(hero.get("mgc_atk", 0)) / NormConfig.MAX_MGC_ATK)),
                ("phy_def_norm", self._clip01(float(hero.get("phy_def", 0)) / NormConfig.MAX_PHY_DEF)),
                ("mgc_def_norm", self._clip01(float(hero.get("mgc_def", 0)) / NormConfig.MAX_MGC_DEF)),
                ("skill1_usable", self_skill_state["skill1_usable"]),
                ("skill1_cooldown_ratio", self_skill_state["skill1_cooldown_ratio"]),
                ("skill2_usable", self_skill_state["skill2_usable"]),
                ("skill2_cooldown_ratio", self_skill_state["skill2_cooldown_ratio"]),
                ("skill3_usable", self_skill_state["skill3_usable"]),
                ("skill3_cooldown_ratio", self_skill_state["skill3_cooldown_ratio"]),
                ("summoner_usable", self_skill_state["summoner_usable"]),
                ("summoner_cooldown_ratio", self_skill_state["summoner_cooldown_ratio"]),
            ]
        )

    def _build_enemy_hero_group(self, main_hero, enemy_hero):
        invisible_skill_state = self._zero_skill_state_features()
        invisible = OrderedDict(
            [
                ("visible", 0.0),
                ("alive", 0.0),
                ("hp_rate", 0.0),
                ("ep_rate", 0.0),
                ("level_norm", 0.0),
                ("rel_x_norm", 0.0),
                ("rel_z_norm", 0.0),
                ("dist_norm", 0.0),
                ("in_attack_range", 0.0),
                ("enemy_is_in_grass", 0.0),
                ("phy_atk_norm", 0.0),
                ("mgc_atk_norm", 0.0),
                ("phy_def_norm", 0.0),
                ("mgc_def_norm", 0.0),
                ("atk_spd_norm", 0.0),
                ("money_norm", 0.0),
                ("skill1_usable", invisible_skill_state["skill1_usable"]),
                ("skill1_cooldown_ratio", invisible_skill_state["skill1_cooldown_ratio"]),
                ("skill2_usable", invisible_skill_state["skill2_usable"]),
                ("skill2_cooldown_ratio", invisible_skill_state["skill2_cooldown_ratio"]),
                ("skill3_usable", invisible_skill_state["skill3_usable"]),
                ("skill3_cooldown_ratio", invisible_skill_state["skill3_cooldown_ratio"]),
                ("summoner_usable", invisible_skill_state["summoner_usable"]),
                ("summoner_cooldown_ratio", invisible_skill_state["summoner_cooldown_ratio"]),
            ]
        )
        if enemy_hero is None or not self._is_visible_to_main(enemy_hero, main_hero):
            return invisible

        distance = self._distance(main_hero, enemy_hero)
        enemy_skill_state = self._build_skill_state_features(enemy_hero)
        return OrderedDict(
            [
                ("visible", 1.0),
                ("alive", self._alive_flag(enemy_hero)),
                ("hp_rate", self._hp_rate(enemy_hero)),
                ("ep_rate", self._ep_rate(enemy_hero)),
                ("level_norm", self._normalize_level(enemy_hero.get("level", 0))),
                ("rel_x_norm", self._normalize_rel(enemy_hero["location"]["x"] - main_hero["location"]["x"])),
                ("rel_z_norm", self._normalize_rel(enemy_hero["location"]["z"] - main_hero["location"]["z"])),
                ("dist_norm", self._normalize_dist(distance)),
                ("in_attack_range", 1.0 if distance <= max(float(main_hero.get("attack_range", 0)), 0.0) else 0.0),
                ("enemy_is_in_grass", 1.0 if enemy_hero.get("is_in_grass", False) else 0.0),
                ("phy_atk_norm", self._clip01(float(enemy_hero.get("phy_atk", 0)) / NormConfig.MAX_PHY_ATK)),
                ("mgc_atk_norm", self._clip01(float(enemy_hero.get("mgc_atk", 0)) / NormConfig.MAX_MGC_ATK)),
                ("phy_def_norm", self._clip01(float(enemy_hero.get("phy_def", 0)) / NormConfig.MAX_PHY_DEF)),
                ("mgc_def_norm", self._clip01(float(enemy_hero.get("mgc_def", 0)) / NormConfig.MAX_MGC_DEF)),
                ("atk_spd_norm", self._clip01(float(enemy_hero.get("atk_spd", 0)) / NormConfig.MAX_ATK_SPEED)),
                ("money_norm", self._clip01(float(enemy_hero.get("money", enemy_hero.get("money_cnt", 0))) / NormConfig.MAX_MONEY)),
                ("skill1_usable", enemy_skill_state["skill1_usable"]),
                ("skill1_cooldown_ratio", enemy_skill_state["skill1_cooldown_ratio"]),
                ("skill2_usable", enemy_skill_state["skill2_usable"]),
                ("skill2_cooldown_ratio", enemy_skill_state["skill2_cooldown_ratio"]),
                ("skill3_usable", enemy_skill_state["skill3_usable"]),
                ("skill3_cooldown_ratio", enemy_skill_state["skill3_cooldown_ratio"]),
                ("summoner_usable", enemy_skill_state["summoner_usable"]),
                ("summoner_cooldown_ratio", enemy_skill_state["summoner_cooldown_ratio"]),
            ]
        )

    def _build_self_tower_group(self, main_hero, self_tower):
        if self_tower is None:
            return OrderedDict(
                [
                    ("alive", 0.0),
                    ("hp_rate", 0.0),
                    ("rel_x_norm", 0.0),
                    ("rel_z_norm", 0.0),
                ]
            )
        return OrderedDict(
            [
                ("alive", self._alive_flag(self_tower)),
                ("hp_rate", self._hp_rate(self_tower)),
                ("rel_x_norm", self._normalize_rel(self_tower["location"]["x"] - main_hero["location"]["x"])),
                ("rel_z_norm", self._normalize_rel(self_tower["location"]["z"] - main_hero["location"]["z"])),
            ]
        )

    def _build_enemy_tower_group(self, main_hero, enemy_tower):
        if enemy_tower is None:
            return OrderedDict(
                [
                    ("visible", 0.0),
                    ("alive", 0.0),
                    ("hp_rate", 0.0),
                    ("rel_x_norm", 0.0),
                    ("rel_z_norm", 0.0),
                ]
            )
        return OrderedDict(
            [
                ("visible", 1.0),
                ("alive", self._alive_flag(enemy_tower)),
                ("hp_rate", self._hp_rate(enemy_tower)),
                ("rel_x_norm", self._normalize_rel(enemy_tower["location"]["x"] - main_hero["location"]["x"])),
                ("rel_z_norm", self._normalize_rel(enemy_tower["location"]["z"] - main_hero["location"]["z"])),
            ]
        )

    def _build_soldier_slots(self, soldiers):
        slots = []
        for slot_index in range(DimConfig.SOLDIER_SLOT_COUNT):
            if slot_index < len(soldiers):
                soldier = soldiers[slot_index]
                slots.append(
                    OrderedDict(
                        [
                            ("visible", 1.0),
                            ("alive", self._alive_flag(soldier)),
                            ("hp_rate", soldier["hp_rate"]),
                            ("rel_x_norm", soldier["rel_x_norm"]),
                            ("rel_z_norm", soldier["rel_z_norm"]),
                            ("dist_norm", soldier["dist_norm"]),
                        ]
                    )
                )
            else:
                slots.append(
                    OrderedDict(
                        [
                            ("visible", 0.0),
                            ("alive", 0.0),
                            ("hp_rate", 0.0),
                            ("rel_x_norm", 0.0),
                            ("rel_z_norm", 0.0),
                            ("dist_norm", 0.0),
                        ]
                    )
                )
        return slots

    def _build_global_context(self, frame_state, main_hero, enemy_hero, self_tower, enemy_tower):
        enemy_level = float(enemy_hero.get("level", 0)) if enemy_hero is not None else 0.0
        enemy_money = float(enemy_hero.get("money", enemy_hero.get("money_cnt", 0))) if enemy_hero is not None else 0.0
        self_level = float(main_hero.get("level", 0))
        self_money = float(main_hero.get("money", main_hero.get("money_cnt", 0)))
        self_hero_type = GameConfig.HERO_ID_TO_INDEX.get(main_hero.get("config_id", -1), -1)
        enemy_hero_type = GameConfig.HERO_ID_TO_INDEX.get(enemy_hero.get("config_id", -1), -1) if enemy_hero is not None else -1

        return OrderedDict(
            [
                ("frame_progress_norm", self._clip01(float(frame_state.get("frame_no", 0)) / GameConfig.MAX_FRAME_NO)),
                ("money_diff_norm", self._normalize_diff(self_money - enemy_money, NormConfig.MAX_MONEY)),
                ("level_diff_norm", self._normalize_diff(self_level - enemy_level, NormConfig.MAX_LEVEL_DIFF)),
                ("self_hero_type", float(self_hero_type) / max(len(GameConfig.HERO_ID_TO_INDEX) - 1, 1)),
                ("enemy_hero_type", float(max(enemy_hero_type, 0)) / max(len(GameConfig.HERO_ID_TO_INDEX) - 1, 1)),
            ]
        )

    def _build_matchup_group(self, main_hero, enemy_hero):
        self_hero_type = GameConfig.HERO_ID_TO_INDEX.get(main_hero.get("config_id", -1), -1)
        enemy_hero_type = GameConfig.HERO_ID_TO_INDEX.get(enemy_hero.get("config_id", -1), -1) if enemy_hero is not None else -1
        if max(self_hero_type, enemy_hero_type, 0) < 0:
            return OrderedDict([("m0", 1.0), ("m1", 0.0), ("m2", 0.0), ("m3", 0.0)])
        matchup_index = self_hero_type * 2 + enemy_hero_type
        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[matchup_index] = 1.0
        return OrderedDict(
            [
                ("m0", one_hot[0]),
                ("m1", one_hot[1]),
                ("m2", one_hot[2]),
                ("m3", one_hot[3]),
            ]
        )

    def _find_main_hero(self, frame_state, observation):
        player_id = observation.get("player_id")
        for hero in frame_state.get("hero_states", []):
            if hero.get("runtime_id") == player_id or hero.get("player_id") == player_id:
                return hero
        for hero in frame_state.get("hero_states", []):
            if self._same_camp(hero.get("camp"), self.main_camp):
                return hero
        return None

    def _find_enemy_hero(self, frame_state, main_hero):
        for hero in frame_state.get("hero_states", []):
            if not self._same_camp(hero.get("camp"), main_hero.get("camp")):
                return hero
        return None

    def _collect_towers(self, npcs, main_hero, same_camp):
        towers = []
        for npc in npcs:
            if not self._is_normal_tower(npc):
                continue
            is_same_camp = self._same_camp(npc.get("camp"), main_hero.get("camp"))
            if is_same_camp != same_camp:
                continue
            if not same_camp and not self._is_visible_to_main(npc, main_hero):
                continue
            towers.append(self._augment_relative_fields(npc, main_hero))
        towers.sort(key=lambda item: item["dist_raw"])
        return towers

    def _collect_soldiers(self, npcs, main_hero, same_camp):
        soldiers = []
        for npc in npcs:
            if not self._is_lane_soldier(npc):
                continue
            is_same_camp = self._same_camp(npc.get("camp"), main_hero.get("camp"))
            if is_same_camp != same_camp:
                continue
            if not same_camp and not self._is_visible_to_main(npc, main_hero):
                continue
            soldiers.append(self._augment_relative_fields(npc, main_hero))
        soldiers.sort(key=lambda item: item["dist_raw"])
        return soldiers[: DimConfig.SOLDIER_SLOT_COUNT]

    def _augment_relative_fields(self, unit, main_hero):
        cloned = dict(unit)
        distance = self._distance(main_hero, unit)
        rel_x = unit["location"]["x"] - main_hero["location"]["x"]
        rel_z = unit["location"]["z"] - main_hero["location"]["z"]
        cloned["hp_rate"] = self._hp_rate(unit)
        cloned["dist_raw"] = distance
        cloned["dist_norm"] = self._normalize_dist(distance)
        cloned["rel_x_norm"] = self._normalize_rel(rel_x)
        cloned["rel_z_norm"] = self._normalize_rel(rel_z)
        return cloned

    def _serialize_unit(self, unit, main_hero, force_visible):
        camp_visible = unit.get("camp_visible", [])
        return {
            "runtime_id": unit.get("runtime_id"),
            "config_id": unit.get("config_id"),
            "camp": unit.get("camp"),
            "sub_type": unit.get("sub_type"),
            "hp": unit.get("hp"),
            "max_hp": unit.get("max_hp"),
            "location": unit.get("location"),
            "camp_visible": camp_visible,
            "visible_to_main": True if force_visible else self._is_visible_to_main(unit, main_hero),
            "dist_norm": self._normalize_dist(self._distance(main_hero, unit)),
        }

    def _is_visible_to_main(self, unit, main_hero):
        if self._same_camp(unit.get("camp"), main_hero.get("camp")):
            return True
        camp_visible = unit.get("camp_visible", [])
        index = self._camp_index(main_hero.get("camp"))
        if isinstance(camp_visible, (list, tuple)) and len(camp_visible) > index:
            return bool(camp_visible[index])
        return False

    def _is_normal_tower(self, npc):
        sub_type = npc.get("sub_type")
        if isinstance(sub_type, int):
            return sub_type == GameConfig.NORMAL_TOWER_SUBTYPE
        if isinstance(sub_type, str):
            return "TOWER" in sub_type and "SPRING" not in sub_type
        return False

    def _is_lane_soldier(self, npc):
        if self._camp_token(npc.get("camp")) not in {"1", "2"}:
            return False
        if self._is_organ_actor(npc):
            return False
        if str(npc.get("sub_type", "")).upper().find("MONSTER") >= 0:
            return False
        return True

    def _is_organ_actor(self, npc):
        actor_type = npc.get("actor_type")
        if isinstance(actor_type, str):
            return "ORGAN" in actor_type
        if isinstance(actor_type, int):
            return actor_type == 3
        return False

    def _alive_flag(self, unit):
        if unit is None:
            return 0.0
        return 1.0 if float(unit.get("hp", 0)) > 0 else 0.0

    def _hp_rate(self, unit):
        if unit is None:
            return 0.0
        max_hp = max(float(unit.get("max_hp", 0)), 1.0)
        return self._clip01(float(unit.get("hp", 0)) / max_hp)

    def _ep_rate(self, unit):
        max_ep = float(unit.get("max_ep", 0))
        if max_ep <= 0:
            return 0.0
        return self._clip01(float(unit.get("ep", 0)) / max_ep)

    def _distance(self, src, dst):
        src_loc = src["location"]
        dst_loc = dst["location"]
        return math.sqrt(
            float(src_loc["x"] - dst_loc["x"]) ** 2 + float(src_loc["z"] - dst_loc["z"]) ** 2
        )

    def _normalize_abs_pos(self, value):
        mirrored = self._mirror_value(value)
        return self._clip01((float(mirrored) + NormConfig.MAX_ABS_COORD) / (NormConfig.MAX_ABS_COORD * 2.0))

    def _normalize_rel(self, value):
        mirrored = self._mirror_value(value)
        return self._clip01((float(mirrored) + NormConfig.MAX_REL_COORD) / (NormConfig.MAX_REL_COORD * 2.0))

    def _normalize_dist(self, distance):
        return self._clip01(float(distance) / NormConfig.MAX_DIST)

    def _normalize_level(self, level):
        level_value = max(float(level), 1.0)
        return self._clip01((level_value - 1.0) / (NormConfig.MAX_LEVEL - 1.0))

    def _normalize_diff(self, value, max_abs_value):
        denominator = max(float(max_abs_value), 1.0)
        return self._clip01((float(value) + denominator) / (denominator * 2.0))

    def _mirror_value(self, value):
        return -value if self._mirror else value

    def _mask_value(self, mask, index):
        if len(mask) <= index:
            return 0.0
        return float(mask[index])

    def _build_skill_state_features(self, hero):
        skill_slots = self._skill_slot_map(hero)
        features = OrderedDict()
        feature_prefix = {
            1: "skill1",
            2: "skill2",
            3: "skill3",
            5: "summoner",
        }
        for slot_type in GameConfig.SKILL_SLOT_TYPES:
            slot = skill_slots.get(slot_type)
            prefix = feature_prefix[slot_type]
            features["{0}_usable".format(prefix)] = self._skill_slot_usable(slot)
            features["{0}_cooldown_ratio".format(prefix)] = self._skill_slot_cooldown_ratio(slot)
        return features

    def _zero_skill_state_features(self):
        return self._build_skill_state_features({})

    def _skill_slot_map(self, hero):
        skill_state = hero.get("skill_state", {})
        slot_map = {}
        if not skill_state:
            return slot_map
        for slot in skill_state.get("slot_states", []):
            slot_type = slot.get("slot_type")
            if slot_type in GameConfig.SKILL_SLOT_TYPES:
                slot_map[slot_type] = slot
        return slot_map

    def _skill_slot_usable(self, slot):
        if not slot:
            return 0.0
        return 1.0 if slot.get("usable", False) else 0.0

    def _skill_slot_cooldown_ratio(self, slot):
        if not slot:
            return 0.0
        cooldown_max = max(float(slot.get("cooldown_max", 0)), 1.0)
        cooldown = float(slot.get("cooldown", 0))
        return self._clip01(cooldown / cooldown_max)

    def _camp_token(self, camp):
        if isinstance(camp, int):
            if camp == 1:
                return "1"
            if camp == 2:
                return "2"
            return str(camp)
        if isinstance(camp, str):
            if camp.endswith("1"):
                return "1"
            if camp.endswith("2"):
                return "2"
            return camp
        return "unknown"

    def _camp_index(self, camp):
        return 0 if self._camp_token(camp) == "1" else 1

    def _same_camp(self, camp_a, camp_b):
        return self._camp_token(camp_a) == self._camp_token(camp_b)

    def _clip01(self, value):
        return max(0.0, min(1.0, float(value)))
