#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

from collections import Counter, deque
import logging

import numpy as np

# 配置日志
logger = logging.getLogger(__name__)

# 导入新的三层奖励系统（必须成功）
from agent_ppo.feature.reward_system_v2 import compute_reward_three_layer

# 导入统一的空间工具库（必须成功，不支持降级）
from agent_ppo.feature.spatial_utils import (
    apply_action_to_pos,
    relative_pos,
    get_direction_ahead_in_local_map,
    extract_safe_pos,
    quantize_to_coarse_cell,
    evaluate_direction_quality,
)

# 导入动作仿真器（Priority 1）- 替换固定delta的动作预测逻辑
from agent_ppo.feature.action_simulator import simulate_action

# 导入统一的masked policy分布工具（Priority 2）
from agent_ppo.feature.mask_utils import softmax_numpy, softmax_torch

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
MAP_DIAG = float(np.sqrt(MAP_SIZE**2 + MAP_SIZE**2))
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Coarse cell size / 粗粒度网格尺寸
COARSE_CELL_SIZE = 4.0
# Memory windows / 记忆窗口大小
RECENT_CELL_WINDOW = 12
STUCK_WINDOW = 6

# 所有空间工具函数已统一到 spatial_utils.py，确保单一真实来源


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _signed_norm(v, v_abs_max):
    """Normalize signed value to [-1, 1]."""
    return float(np.clip(float(v) / max(float(v_abs_max), 1e-6), -1.0, 1.0))


def _sorted_entities_by_distance(hero_pos, entities):
    enriched = []
    for entity in entities:
        entity_pos = extract_safe_pos(entity)
        _, _, dist = relative_pos(hero_pos, entity_pos, MAP_SIZE)
        enriched.append((dist, entity, entity_pos))
    enriched.sort(key=lambda item: item[0])
    return enriched


def _extract_hero_hp_info(hero_dict):
    """
    统一提取英雄的血量信息（hp 和 max_hp）。
    
    这是唯一的官方血量读取入口，确保全项目一致性。
    
    读取优先级（hp）：
        1. hero_dict["hp"] - 首选
        2. hero_dict["hero_hp"] - 备选（если hp=0）
        3. 0.0 - 最后缺省（表示英雄可能已死）
    
    读取优先级（max_hp）：
        1. hero_dict["max_hp"] - 标准字段
        2. 100.0 - 缺省值（通常游戏英雄血量上限）
    
    注意：action_quality 模块会在hp为0时使用保守策略评估
    
    Returns:
        (hero_hp, hero_max_hp): 元组，两个值都是 float
    """
    # 读取当前血量，优先级: hp > hero_hp > 0.0
    hero_hp = float(hero_dict.get("hp", 0.0))
    if hero_hp == 0.0:
        hero_hp = float(hero_dict.get("hero_hp", 0.0))
    
    # 读取最大血量，缺省值 100.0（标准值）
    # 如果值异常小（<1.0），取缺省值
    hero_max_hp = float(hero_dict.get("max_hp", 100.0))
    if hero_max_hp < 1.0:
        hero_max_hp = 100.0
    
    return hero_hp, hero_max_hp


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 1.0
        self.last_nearest_treasure_dist_norm = 1.0
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.last_flash_count = 0
        self.last_total_score = 0.0
        self.last_hero_pos = None
        self.visit_counter = Counter()
        self.visited_cells = set()
        self.recent_cells = deque(maxlen=RECENT_CELL_WINDOW)
        self.prev_action = -1
        
        # D: 时序与记忆特征 - 新增
        self.trajectory_history = deque(maxlen=20)  # 历史轨迹（coarse cells）
        self.hero_pos_history = deque(maxlen=20)   # 英雄实际位置历史（{x,z}字典） - 用于action_quality
        self.monster_dist_history = deque(maxlen=15)  # 怪物距离历史
        self.treasure_dist_history = deque(maxlen=15)  # 宝箱距离历史
        self.reward_history = deque(maxlen=15)  # 奖励历史
        self.danger_trend = deque(maxlen=5)  # 危险趋势窗口
        self.exploration_history = Counter()  # 探索记忆
        
        # Action quality evaluation related
        self.hero_max_hp = 100.0  # 默认最大血量（会在feature_process中更新）
        self.danger_trend_value = 0.0  # 当前危险趋势值

    def _extract_legal_action(self, observation):
        legal_act_raw = observation.get("legal_action", observation.get("legal_act", []))
        legal_action = [1] * 16

        if isinstance(legal_act_raw, (list, tuple, np.ndarray)) and len(legal_act_raw) > 0:
            # 检查是否是二进制掩码 (0或1的列表)
            if all(int(a) in [0, 1] for a in legal_act_raw) and len(legal_act_raw) >= 8:
                # 二进制掩码模式
                if len(legal_act_raw) >= 16:
                    legal_action = [int(a) for a in legal_act_raw[:16]]
                else:
                    # 不足16位，默认后续为0
                    legal_action = [int(a) for a in legal_act_raw] + [0] * (16 - len(legal_act_raw))
            elif isinstance(legal_act_raw[0], (bool, np.bool_)):
                # bool 类型
                if len(legal_act_raw) >= 16:
                    for idx in range(16):
                        legal_action[idx] = int(legal_act_raw[idx])
                else:
                    for idx in range(min(8, len(legal_act_raw))):
                        legal_action[idx] = int(legal_act_raw[idx])
                    for idx in range(8, 16):
                        legal_action[idx] = 0
            else:
                # 动作索引模式
                valid_set = {int(a) for a in legal_act_raw if 0 <= int(a) < 16}
                legal_action = [1 if idx in valid_set else 0 for idx in range(16)]

        if sum(legal_action) == 0:
            legal_action = [1] * 8 + [0] * 8

        return legal_action

    def _build_monster_features(self, hero_pos, monsters):
        monster_records = _sorted_entities_by_distance(hero_pos, monsters)
        monster_feats = []
        cur_min_monster_dist_norm = 1.0

        for idx in range(2):
            if idx < len(monster_records):
                raw_dist, monster, monster_pos = monster_records[idx]
                dist_norm = _norm(raw_dist, MAP_DIAG)
                cur_min_monster_dist_norm = min(cur_min_monster_dist_norm, dist_norm)
                speed_norm = _norm(monster.get("speed", 0.0), MAX_MONSTER_SPEED)
                bucket_norm = _norm(monster.get("hero_l2_distance", raw_dist / 30.0), MAX_DIST_BUCKET)
                danger_score = float(np.clip(1.0 - dist_norm + 0.15 * speed_norm, 0.0, 1.0))
                monster_feats.append(
                    np.array(
                        [
                            1.0,
                            _signed_norm(monster_pos["x"] - hero_pos["x"], MAP_SIZE),
                            _signed_norm(monster_pos["z"] - hero_pos["z"], MAP_SIZE),
                            dist_norm,
                            speed_norm,
                            bucket_norm,
                            danger_score,
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                monster_feats.append(np.zeros(7, dtype=np.float32))

        return monster_feats, cur_min_monster_dist_norm

    def _build_treasure_features(self, hero_pos, treasures, cur_min_monster_dist_norm):
        treasure_records = _sorted_entities_by_distance(hero_pos, treasures)
        treasure_feats = []
        nearest_treasure_dist_norm = 1.0

        for idx in range(3):
            if idx < len(treasure_records):
                raw_dist, _, treasure_pos = treasure_records[idx]
                dist_norm = _norm(raw_dist, MAP_DIAG)
                if idx == 0:
                    nearest_treasure_dist_norm = dist_norm
                loot_priority = float(np.clip(cur_min_monster_dist_norm - 0.65 * dist_norm, 0.0, 1.0))
                treasure_feats.append(
                    np.array(
                        [
                            1.0,
                            _signed_norm(treasure_pos["x"] - hero_pos["x"], MAP_SIZE),
                            _signed_norm(treasure_pos["z"] - hero_pos["z"], MAP_SIZE),
                            dist_norm,
                            loot_priority,
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                treasure_feats.append(np.zeros(5, dtype=np.float32))

        return np.concatenate(treasure_feats).astype(np.float32), nearest_treasure_dist_norm

    def _build_buff_features(self, hero_pos, buffs, cur_min_monster_dist_norm):
        buff_records = _sorted_entities_by_distance(hero_pos, buffs)
        if not buff_records:
            return np.zeros(5, dtype=np.float32), 1.0

        raw_dist, _, buff_pos = buff_records[0]
        dist_norm = _norm(raw_dist, MAP_DIAG)
        buff_priority = float(np.clip(cur_min_monster_dist_norm - 0.45 * dist_norm, 0.0, 1.0))
        buff_feat = np.array(
            [
                1.0,
                _signed_norm(buff_pos["x"] - hero_pos["x"], MAP_SIZE),
                _signed_norm(buff_pos["z"] - hero_pos["z"], MAP_SIZE),
                dist_norm,
                buff_priority,
            ],
            dtype=np.float32,
        )
        return buff_feat, dist_norm

    def _build_map_features(self, map_info):
        map_feat = np.zeros(25, dtype=np.float32)
        if map_info is None or not len(map_info) or not len(map_info[0]):
            return map_feat

        center_row = len(map_info) // 2
        center_col = len(map_info[0]) // 2
        flat_idx = 0
        for row in range(center_row - 2, center_row + 3):
            for col in range(center_col - 2, center_col + 3):
                if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                    map_feat[flat_idx] = float(map_info[row][col] != 0)
                flat_idx += 1
        return map_feat

    # ===== 五大类特征计算方法 =====
    
    def _build_hero_enhanced_features(self, hero_pos, hero, env_info, step_no, max_step):
        """A. 自身状态特征 (24D): 位置+速度+技能+进度+其他
        英雄位置、移动速度、技能状态、任务进度、生存状态
        """
        flash_cd = float(hero.get("flash_cooldown", 0.0))
        buff_remain = float(hero.get("buff_remaining_time", 0.0))
        
        treasures_collected = int(env_info.get("treasures_collected", 0))
        total_treasure = int(env_info.get("total_treasure", 1))
        collected_buff = int(env_info.get("collected_buff", 0))
        total_buff = int(env_info.get("total_buff", 1))
        
        # 加速状态 (是否有加速buff)
        is_speeding = float(buff_remain > 0.0)
        
        # 计算速度 (相对于历史位置)
        if self.last_hero_pos:
            _, _, move_dist = relative_pos(hero_pos, self.last_hero_pos, MAP_SIZE)
            speed_norm = _norm(move_dist, 3.0)  # 最大移动速度
        else:
            speed_norm = 0.0
        
        step_norm = _norm(step_no, max_step)
        time_remaining = 1.0 - step_norm
        treasure_progress = float(treasures_collected) / float(total_treasure) if total_treasure > 0 else 0.0
        buff_progress = float(collected_buff) / float(total_buff) if total_buff > 0 else 0.0
        
        # 生存评估 (预计还能活多久)
        estimated_survival_ratio = time_remaining
        
        hero_enhanced = np.array([
            # 位置特征 (2D)
            _norm(hero_pos["x"], MAP_SIZE),
            _norm(hero_pos["z"], MAP_SIZE),
            # 速度特征 (3D)
            speed_norm,
            is_speeding,
            _norm(buff_remain, MAX_BUFF_DURATION),
            # 技能状态 (4D)
            float(flash_cd <= 0.0),  # 闪现可用
            _norm(flash_cd, MAX_FLASH_CD),  # 闪现冷却进度
            float(flash_cd <= 30.0),  # 闪现即将可用
            _norm(flash_cd / max(MAX_FLASH_CD, 1.0), 1.0),  # 闪现可用度
            # 进度特征 (7D)
            step_norm,
            time_remaining,
            treasure_progress,
            buff_progress,
            1.0 - treasure_progress,
            1.0 - buff_progress,
            estimated_survival_ratio,
            # 其他状态 (8D)
            float(treasures_collected),  # 绝对数量
            float(collected_buff),
            _norm(treasures_collected + collected_buff, max(total_treasure + total_buff, 1)),
            float(hero.get("status", 0) == 1),  # 是否活跃
            float(self.last_hero_pos is None),  # 第一步标志
            float(step_no % 50 < 5),  # 周期性标志
            float(step_no > max_step * 0.9),  # 最后10%标志
            0.0,  # 预留
        ], dtype=np.float32)
        
        return hero_enhanced

    def _build_external_entities_enhanced(self, hero_pos, monsters, treasures, buffs, cur_min_monster_dist_norm):
        """B. 外部实体特征 (34D): 怪物(14D) + 宝箱(12D) + buff(8D)
        多个怪物、多个宝箱、增益位置
        """
        # 怪物特征增强 (14D): 2个怪物各7D
        monster_records = _sorted_entities_by_distance(hero_pos, monsters)
        monster_feats = []
        
        for idx in range(2):
            if idx < len(monster_records):
                raw_dist, monster, monster_pos = monster_records[idx]
                dist_norm = _norm(raw_dist, MAP_DIAG)
                speed_norm = _norm(monster.get("speed", 0.0), MAX_MONSTER_SPEED)
                
                # 方向特征
                rel_x = _signed_norm(monster_pos["x"] - hero_pos["x"], MAP_SIZE / 2)
                rel_z = _signed_norm(monster_pos["z"] - hero_pos["z"], MAP_SIZE / 2)
                
                # 威胁评分
                threat_score = max(0.0, 1.0 - dist_norm) * (0.5 + 0.5 * speed_norm)
                
                monster_feats.extend([
                    1.0,  # 怪物存在标志
                    dist_norm,
                    speed_norm,
                    rel_x,
                    rel_z,
                    threat_score,
                    float(idx == 0),  # 最近怪物标志
                ])
            else:
                monster_feats.extend([0.0] * 7)
        
        # 宝箱特征增强 (12D): 3个宝箱各4D
        treasure_records = _sorted_entities_by_distance(hero_pos, treasures)
        treasure_feats = []
        
        for idx in range(3):
            if idx < len(treasure_records):
                raw_dist, _, treasure_pos = treasure_records[idx]
                dist_norm = _norm(raw_dist, MAP_DIAG)
                
                # 宝箱优先级 (离怪物更远更好)
                priority = max(0.0, cur_min_monster_dist_norm - 0.6 * dist_norm)
                
                rel_x = _signed_norm(treasure_pos["x"] - hero_pos["x"], MAP_SIZE / 2)
                rel_z = _signed_norm(treasure_pos["z"] - hero_pos["z"], MAP_SIZE / 2)
                
                treasure_feats.extend([
                    dist_norm,
                    priority,
                    rel_x,
                    rel_z,
                ])
            else:
                treasure_feats.extend([0.0] * 4)
        
        # Buff特征加强 (8D)
        buff_records = _sorted_entities_by_distance(hero_pos, buffs)
        
        if buff_records:
            raw_dist, _, buff_pos = buff_records[0]
            buff_dist_norm = _norm(raw_dist, MAP_DIAG)
            buff_priority = max(0.0, cur_min_monster_dist_norm - 0.5 * buff_dist_norm)
            rel_x = _signed_norm(buff_pos["x"] - hero_pos["x"], MAP_SIZE / 2)
            rel_z = _signed_norm(buff_pos["z"] - hero_pos["z"], MAP_SIZE / 2)
            buff_feats = [1.0, buff_dist_norm, buff_priority, rel_x, rel_z, 0.0, 0.0, 0.0]
        else:
            buff_feats = [0.0] * 8
        
        entities_feat = np.array(monster_feats + treasure_feats + buff_feats, dtype=np.float32)
        return entities_feat

    def _build_map_and_paths_enhanced(self, map_info, hero_pos, monsters, treasures):
        """C. 地图与路径特征 (35D): 局部地图(25D) + 方向质量(8D) + 逃生通路(2D)
        局部障碍、8个方向的安全性和收益、逃生路线
        """
        # 1. 局部地图 (25D) - 5x5网格
        map_feat = self._build_map_features(map_info)
        
        # 2. 8个移动方向的质量评分 (8D)
        # 使用共享的方向质量评估函数
        direction_quality = []
        
        for action_idx in range(8):  # 只看移动，不看闪现
            # 使用统一的方向质量评估函数，避免重复代码
            eval_result = evaluate_direction_quality(
                action_id=action_idx,
                hero_pos=hero_pos,
                monsters=monsters,
                treasures=treasures,
                local_map=map_info,
                map_size=MAP_SIZE,
                extract_pos_fn=extract_safe_pos,
                distance_fn=lambda pos_a, pos_b: relative_pos(pos_a, pos_b, MAP_SIZE)[2],
            )
            
            direction_quality.append(eval_result["direction_quality"])
        
        # 3. 逃生通路评分 (2D)
        # 最安全的方向 + 最多宝箱方向的安全性
        if direction_quality:
            safest_quality = max(direction_quality)
            treasure_best_quality = max(direction_quality) if direction_quality else 0.0
        else:
            safest_quality = 0.5
            treasure_best_quality = 0.5
        
        map_and_paths_feat = np.concatenate([
            map_feat,
            np.array(direction_quality, dtype=np.float32),
            np.array([safest_quality, treasure_best_quality], dtype=np.float32)
        ])
        
        return map_and_paths_feat

    def _build_temporal_memory_features(self, hero_pos, hero_cell, cur_min_monster_dist_norm, 
                                        nearest_treasure_dist_norm, reward_info):
        """D. 时序与记忆特征 (30D): 
        历史轨迹(3D) + 危险趋势(4D) + 收益趋势(3D) + 探索记忆(5D) + 动作历史(5D) + 其他趋势(5D)
        """
        # 记录历史
        self.trajectory_history.append(hero_cell)
        self.monster_dist_history.append(cur_min_monster_dist_norm)
        self.treasure_dist_history.append(nearest_treasure_dist_norm)
        self.reward_history.append(reward_info.get("reward_total", 0.0))
        
        # 更新危险趋势指标（用于action_quality评估）
        if len(self.monster_dist_history) >= 2:
            recent_dists = list(self.monster_dist_history)[-3:]
            self.danger_trend_value = float(recent_dists[-1] - recent_dists[0])  # 正值表示怪物在离开
        else:
            self.danger_trend_value = 0.0
        
        self.danger_trend.append(self.danger_trend_value)
        
        # 1. 历史轨迹相似度 (3D)
        # 最近是否重复、轨迹变化、探索度
        if len(self.trajectory_history) >= 2:
            recent_trajectory = list(self.trajectory_history)[-6:]
            unique_count = len(set(recent_trajectory))
            trajectory_change = float(unique_count) / max(len(recent_trajectory), 1)
            repeat_factor = 1.0 - trajectory_change
        else:
            trajectory_change = 0.5
            repeat_factor = 0.0
        
        trajectory_exploration_ratio = float(len(self.visited_cells)) / max(32 * 32, 1)  # 32x32网格
        
        # 2. 危险趋势 (4D)
        # 怪物距离走势、最小值、最大值、波动度
        if len(self.monster_dist_history) >= 2:
            recent_dists = list(self.monster_dist_history)[-5:]
            dist_trend = (recent_dists[-1] - recent_dists[0]) / 5.0 if recent_dists else 0.0  # 趋势
            dist_min = min(recent_dists)
            dist_max = max(recent_dists)
            dist_volatility = (dist_max - dist_min) / max(1.0, dist_max)
        else:
            dist_trend = 0.0
            dist_min = cur_min_monster_dist_norm
            dist_max = cur_min_monster_dist_norm
            dist_volatility = 0.0
        
        # 3. 收益趋势 (3D)
        # 奖励走势、平均奖励、最大奖励
        if len(self.reward_history) >= 2:
            recent_rewards = list(self.reward_history)[-5:]
            reward_trend = (recent_rewards[-1] - recent_rewards[0]) / 5.0 if recent_rewards else 0.0
            avg_reward = float(np.mean(recent_rewards))
            max_reward = max(recent_rewards)
        else:
            reward_trend = 0.0
            avg_reward = 0.0
            max_reward = 0.0
        
        # 4. 探索记忆 (5D)
        # 访问过区域数、高频区域、冷门区域、覆盖度、多样性
        exploration_area_count = len(self.visited_cells)
        high_freq_visits = sum(1 for count in self.visit_counter.values() if count >= 3)
        coverage_ratio = float(exploration_area_count) / 32.0  # 最多32×32个粗粒度格子
        
        # 5. 动作历史 (5D)
        # 最后5个动作的交互特征
        recent_action_pattern = float(self.prev_action >= 8)  # 最近是否用过闪现
        action_diversity = 0.5  # 动作多样性评分
        
        # 6. 其他趋势 (5D) - 宝箱距离趋势等
        if len(self.treasure_dist_history) >= 2:
            treasure_trend = float(self.treasure_dist_history[-1] - self.treasure_dist_history[0]) / 5.0
            treasure_volatility = (max(self.treasure_dist_history) - min(self.treasure_dist_history)) / 2.0
        else:
            treasure_trend = 0.0
            treasure_volatility = 0.0
        
        temporal_memory_feat = np.array([
            # 历史轨迹 (3D)
            trajectory_change,
            repeat_factor,
            trajectory_exploration_ratio,
            # 危险趋势 (4D)
            _signed_norm(dist_trend, 0.2),
            dist_min,
            dist_max,
            dist_volatility,
            # 收益趋势 (3D)
            _signed_norm(reward_trend, 1.0),
            _norm(avg_reward + 3.0, 6.0),  # 偏移到正范围
            _norm(max_reward + 3.0, 6.0),
            # 探索记忆 (5D)
            _norm(exploration_area_count, 100.0),
            _norm(high_freq_visits, 20.0),
            coverage_ratio,
            len(self.visited_cells) / max(len(self.trajectory_history), 1),  # 多样性
            0.0,  # 预留
            # 动作历史 (5D)
            float(self.prev_action >= 0 and self.prev_action < 8),  # 上一步是移动
            float(self.prev_action >= 8),  # 上一步是闪现
            recent_action_pattern,
            action_diversity,
            0.0,  # 预留
            # 其他趋势 (5D)
            _signed_norm(treasure_trend, 0.3),
            _norm(treasure_volatility, 0.5),
            nearest_treasure_dist_norm,
            float(len(self.treasure_dist_history) > 0),
            0.0,  # 预留
            # 额外空间 (5D) - 为了达到 30D
            0.0,  # 扩展空间 1
            0.0,  # 扩展空间 2
            0.0,  # 扩展空间 3
            0.0,  # 扩展空间 4
            0.0,  # 扩展空间 5
        ], dtype=np.float32)
        
        return temporal_memory_feat

    def _build_decision_auxiliary_features(self, hero_pos, monsters, treasures, buffs, 
                                           cur_min_monster_dist_norm, nearest_treasure_dist_norm,
                                           flash_cd, buff_remain, move_dist):
        """E. 决策辅助特征 (20D):
        风险评分(4D) + 收益评分(4D) + 闪现专项评分(4D) + 综合建议(8D)
        """
        # 1. 风险评分 (4D)
        # 当前危险、最小安全距离、危险趋势、逃脱难度
        current_danger = max(0.0, 1.0 - cur_min_monster_dist_norm)
        min_safe_dist = 0.2  # 安全阈值
        danger_too_close = float(cur_min_monster_dist_norm < min_safe_dist)
        
        if len(self.monster_dist_history) >= 3:
            recent_dists = list(self.monster_dist_history)[-3:]
            danger_trend = float(recent_dists[-1] < recent_dists[0])  # 怪物在靠近
        else:
            danger_trend = 0.0
        
        # 逃脱难度 = 距离近 + 怪物速度快 + 地形差
        escape_difficulty = current_danger
        
        # 2. 收益评分 (4D)
        # 宝箱可达度、buff可达度、收益风险比、总体收益指数
        treasure_reachability = max(0.0, 1.0 - nearest_treasure_dist_norm * 0.5)
        
        if buffs:
            buff_distances = []
            for b in buffs:
                _, _, dist = relative_pos(hero_pos, extract_safe_pos(b), MAP_SIZE)
                buff_distances.append(dist)
            buff_reachability = max(0.0, 1.0 - min(buff_distances) / MAP_DIAG * 0.5)
        else:
            buff_reachability = 0.0
        
        # 收益-风险比
        reward_risk_ratio = treasure_reachability / max(current_danger + 0.1, 1.0)
        
        # 3. 闪现专项评分 (4D)
        # 闪现必要性、闪现收益、闪现安全性、最优闪现方向
        flash_necessity = current_danger if flash_cd <= 0.0 else 0.0
        
        # 闪现能获得的收益
        if flash_cd <= 0.0:
            best_flash_benefit = 0.0
            best_flash_safety = 0.0
            for action_idx in range(8, 16):  # 闪现方向
                flash_pos = apply_action_to_pos(hero_pos, action_idx, MAP_SIZE)
                if monsters:
                    dists = []
                    for m in monsters:
                        _, _, dist = relative_pos(flash_pos, extract_safe_pos(m), MAP_SIZE)
                        dists.append(dist)
                    flash_min_dist = min(dists)
                    flash_safety = _norm(flash_min_dist, MAP_DIAG)
                else:
                    flash_safety = 1.0
                
                if treasures:
                    dists = []
                    for t in treasures:
                        _, _, dist = relative_pos(flash_pos, extract_safe_pos(t), MAP_SIZE)
                        dists.append(dist)
                    flash_treasure_dist = min(dists)
                    benefit = max(0, nearest_treasure_dist_norm - flash_treasure_dist / MAP_DIAG)
                else:
                    benefit = 0.0
                
                best_flash_benefit = max(best_flash_benefit, benefit)
                best_flash_safety = max(best_flash_safety, flash_safety)
        else:
            best_flash_benefit = 0.0
            best_flash_safety = 0.5
        
        flash_ready = float(flash_cd <= 0.0)
        
        # 4. 综合建议 (8D)
        # 推荐动作类型、优先级排序、综合评分、置信度
        should_move_to_treasure = float(treasure_reachability > 0.3 and current_danger < 0.5)
        should_seek_buff = float(buff_reachability > 0.2 and buff_remain <= 0.0)
        should_emergency_flash = float(current_danger > 0.7 and flash_cd <= 0.0)
        should_retreat = float(current_danger > 0.6)
        
        primary_action_confidence = max(
            float(should_move_to_treasure),
            max(float(should_seek_buff), max(float(should_emergency_flash), float(should_retreat)))
        )
        
        decision_auxiliary_feat = np.array([
            # 风险评分 (4D)
            current_danger,
            danger_too_close,
            _signed_norm(danger_trend, 1.0),
            escape_difficulty,
            # 收益评分 (4D)
            treasure_reachability,
            buff_reachability,
            reward_risk_ratio,
            treasure_reachability + buff_reachability,  # 综合收益
            # 闪现专项评分 (4D)
            flash_necessity,
            best_flash_benefit,
            best_flash_safety,
            flash_ready,
            # 综合建议 (8D)
            float(should_move_to_treasure),
            float(should_seek_buff),
            float(should_emergency_flash),
            float(should_retreat),
            primary_action_confidence,
            float(move_dist > 0.1),  # 是否在活跃移动
            0.0,  # 预留
            0.0,  # 预留
        ], dtype=np.float32)
        
        return decision_auxiliary_feat

    def _stuck_score(self, current_cell):
        recent = list(self.recent_cells)
        if current_cell is not None:
            recent = recent + [current_cell]
        if len(recent) < STUCK_WINDOW:
            return 0.0

        recent = recent[-STUCK_WINDOW:]
        unique_cell_count = len(set(recent))
        if unique_cell_count <= 1:
            return 1.0
        if unique_cell_count == 2:
            return 0.75
        if unique_cell_count == 3:
            return 0.35
        return 0.0

    def _make_reward_info(self, **kwargs):
        reward_info = {
            "survive_reward": 0.0,
            "dist_shaping": 0.0,
            "emergency_escape_reward": 0.0,
            "treasure_pickup_reward": 0.0,
            "treasure_approach_reward": 0.0,
            "buff_reward": 0.0,
            "score_gain_reward": 0.0,
            "exploration_reward": 0.0,
            "loiter_penalty": 0.0,
            "flash_reward": 0.0,
            "reward_total": 0.0,
            "min_monster_dist_norm": 1.0,
            "nearest_treasure_dist_norm": 1.0,
            "stuck_score": 0.0,
            "move_dist": 0.0,
        }
        reward_info.update(kwargs)
        return {key: float(value) for key, value in reward_info.items()}

    def _compute_reward(
        self,
        hero_cell,
        env_info,
        cur_min_monster_dist_norm,
        nearest_treasure_dist_norm,
        move_dist,
        last_action,
    ):
        """
        计算奖励 - 使用新的三层结构奖励系统
        
        Try to use the new three-layer reward system if available,
        fall back to the old system if not.
        """
        global USE_NEW_REWARD_SYSTEM
        
        treasures_collected = int(env_info.get("treasures_collected", env_info.get("treasure_collected_count", 0)))
        collected_buff = int(env_info.get("collected_buff", 0))
        total_buff = int(env_info.get("total_buff", 1))
        flash_count = int(env_info.get("flash_count", 0))
        total_treasure = int(env_info.get("total_treasure", 1))
        buff_remain = float(env_info.get("buff_remaining_time", 0.0))
        
        # ====== 使用新的三层奖励系统（严格模式）======
        try:
            # 判断是否死亡
            is_dead = env_info.get("is_dead", False)
            
            reward_total, reward_details = compute_reward_three_layer(
                # 环境信息
                step_no=self.step_no,
                max_step=self.max_step,
                is_dead=is_dead,
                
                # 距离信息
                cur_min_monster_dist_norm=cur_min_monster_dist_norm,
                last_min_monster_dist_norm=self.last_min_monster_dist_norm,
                nearest_treasure_dist_norm=nearest_treasure_dist_norm,
                last_nearest_treasure_dist_norm=self.last_nearest_treasure_dist_norm,
                
                # 收集信息
                treasures_collected=treasures_collected,
                total_treasure=total_treasure,
                collected_buff=collected_buff,
                total_buff=total_buff,
                buff_remain=buff_remain,
                flash_count=flash_count,
                last_flash_count=self.last_flash_count,
                
                # 行为信息
                move_dist=move_dist,
                hero_cell=hero_cell,
                recent_cells=self.recent_cells,
                visit_counter=self.visit_counter,
                prev_action=self.prev_action,
                last_action=last_action,
                
                # 可选：上一步的收集数据
                last_treasures_collected=self.last_treasure_count,
                last_collected_buff=self.last_buff_count,
            )
            
            # 构建奖励信息（包含三层详细拆分）
            reward_info = {
                # 新系统的详细拆分
                "survival_reward": reward_details.get("survival_reward", 0.0),
                "score_objective": reward_details.get("score_objective", 0.0),
                "layer_a": reward_details.get("layer_a", 0.0),
                "distance_shaping": reward_details.get("distance_shaping", 0.0),
                "danger_zone_penalty": reward_details.get("danger_zone_penalty", 0.0),
                "buff_bonus": reward_details.get("buff_bonus", 0.0),
                "flash_escape_bonus": reward_details.get("flash_escape_bonus", 0.0),
                "layer_b": reward_details.get("layer_b", 0.0),
                "movement_penalty": reward_details.get("movement_penalty", 0.0),
                "loop_penalty": reward_details.get("loop_penalty", 0.0),
                "wasteful_flash_penalty": reward_details.get("wasteful_flash_penalty", 0.0),
                "layer_c": reward_details.get("layer_c", 0.0),
                
                # 聚合指标：总奖励和距离信息
                "reward_total": reward_total,
                "min_monster_dist_norm": cur_min_monster_dist_norm,
                "nearest_treasure_dist_norm": nearest_treasure_dist_norm,
            }
            
            return reward_total, reward_info
            
        except Exception as e:
            # 严格模式：新奖励系统报错直接停止，不自动降级
            # 这确保实验口径一致，避免某次step的报错导致整次训练切换系统
            raise RuntimeError(
                f"[REWARD SYSTEM ERROR - STRICT MODE] New reward_system_v2 failed at step {self.step_no}:\n"
                f"  Type: {type(e).__name__}\n"
                f"  Message: {e}\n"
                f"  Reward system errors are fatal in development mode.\n"
                f"  Please fix the issue and retry training.\n"
                f"  Details: {str(e)[-200:]}"
            ) from e

    def _update_memory(
        self,
        hero_pos,
        hero_cell,
        env_info,
        cur_min_monster_dist_norm,
        nearest_treasure_dist_norm,
        last_action,
    ):
        self.last_hero_pos = {"x": hero_pos["x"], "z": hero_pos["z"]}
        self.last_min_monster_dist_norm = cur_min_monster_dist_norm
        self.last_nearest_treasure_dist_norm = nearest_treasure_dist_norm
        self.last_treasure_count = int(env_info.get("treasures_collected", env_info.get("treasure_collected_count", 0)))
        self.last_buff_count = int(env_info.get("collected_buff", 0))
        self.last_flash_count = int(env_info.get("flash_count", 0))
        self.last_total_score = float(env_info.get("total_score", 0.0))

        self.visit_counter[hero_cell] += 1
        self.visited_cells.add(hero_cell)
        self.recent_cells.append(hero_cell)
        self.trajectory_history.append(hero_cell)  # 存储coarse cell用于grid划分
        self.hero_pos_history.append({"x": hero_pos["x"], "z": hero_pos["z"]})  # 存储exact位置用于action_quality
        self.prev_action = int(last_action) if isinstance(last_action, (int, np.integer)) else -1

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, reward and reward info.

        将 env_obs 转换为特征向量、合法动作掩码、即时奖励和奖励拆分信息。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        hero = frame_state["heroes"]
        hero_pos = extract_safe_pos(hero)
        hero_cell = quantize_to_coarse_cell(hero_pos, cell_size=COARSE_CELL_SIZE, map_size=MAP_SIZE)
        flash_cd = float(hero.get("flash_cooldown", 0.0))
        buff_remain = float(hero.get("buff_remaining_time", 0.0))
        
        # 统一提取英雄血量信息（会在后续action_quality计算时使用）
        _, hero_max_hp = _extract_hero_hp_info(hero)
        self.hero_max_hp = hero_max_hp

        move_dx = 0.0
        move_dz = 0.0
        move_dist = 0.0
        if self.last_hero_pos is not None:
            move_dx = hero_pos["x"] - self.last_hero_pos["x"]
            move_dz = hero_pos["z"] - self.last_hero_pos["z"]
            _, _, move_dist = relative_pos(hero_pos, self.last_hero_pos, MAP_SIZE)

        monsters = frame_state.get("monsters", [])
        _, cur_min_monster_dist_norm = self._build_monster_features(hero_pos, monsters)

        organs = frame_state.get("organs", [])
        treasures = [organ for organ in organs if int(organ.get("sub_type", 0)) == 1 and int(organ.get("status", 0)) == 1]
        buffs = [organ for organ in organs if int(organ.get("sub_type", 0)) == 2 and int(organ.get("status", 0)) == 1]
        
        # 计算最近宝箱距离用于后续特征构建
        if treasures:
            dists = []
            for t in treasures:
                _, _, dist = relative_pos(hero_pos, extract_safe_pos(t), MAP_SIZE)
                dists.append(dist)
            nearest_treasure_dist = min(dists)
            nearest_treasure_dist_norm = _norm(nearest_treasure_dist, MAP_DIAG)
        else:
            nearest_treasure_dist_norm = 1.0
        
        # 获取合法动作掩码
        legal_action = self._extract_legal_action(observation)
        stuck_score = self._stuck_score(hero_cell)

        # ===== 新的5大类特征系统 =====
        
        # 首先计算必需的基础信息，用于所有特征构建
        if self.last_hero_pos is None:
            reward_total = 0.0
            reward_info = self._make_reward_info(
                reward_total=reward_total,
                min_monster_dist_norm=cur_min_monster_dist_norm,
                nearest_treasure_dist_norm=nearest_treasure_dist_norm,
                stuck_score=stuck_score,
                move_dist=move_dist,
            )
        else:
            reward_total, reward_info = self._compute_reward(
                hero_cell=hero_cell,
                env_info=env_info,
                cur_min_monster_dist_norm=cur_min_monster_dist_norm,
                nearest_treasure_dist_norm=nearest_treasure_dist_norm,
                move_dist=move_dist,
                last_action=last_action,
            )
        
        # A. 自身状态特征 (24D)
        hero_enhanced_feat = self._build_hero_enhanced_features(
            hero_pos, hero, env_info, self.step_no, self.max_step
        )
        
        # B. 外部实体特征 (34D)：怪物(2×7D) + 宝藏(12D) + Buff(8D)
        entities_feat = self._build_external_entities_enhanced(
            hero_pos, monsters, treasures, buffs, cur_min_monster_dist_norm
        )
        
        # 分解 entities_feat (34D) 为 4 个子特征向量用于模型编码
        # entities_feat 结构：[monster1(7D) + monster2(7D) + treasure(12D) + buff(8D)]
        monster1_feat = entities_feat[0:7]
        monster2_feat = entities_feat[7:14]
        treasure_feat = entities_feat[14:26]
        buff_feat = entities_feat[26:34]
        
        # C. 地图与路径特征 (35D)
        map_and_paths_feat = self._build_map_and_paths_enhanced(
            map_info, hero_pos, monsters, treasures
        )
        
        # D. 时序与记忆特征 (30D)
        temporal_memory_feat = self._build_temporal_memory_features(
            hero_pos, hero_cell, cur_min_monster_dist_norm, 
            nearest_treasure_dist_norm, reward_info
        )
        
        # E. 决策辅助特征 (20D)
        decision_auxiliary_feat = self._build_decision_auxiliary_features(
            hero_pos, monsters, treasures, buffs,
            cur_min_monster_dist_norm, nearest_treasure_dist_norm,
            flash_cd, buff_remain, move_dist
        )
        
        # 按照模型期望的 10 个分组拼接特征
        # 顺序必须与 Config.FEATURE_SPLIT_SHAPE 的 10 项对应
        
        # 更新内存用于下一步
        self._update_memory(
            hero_pos=hero_pos,
            hero_cell=hero_cell,
            env_info=env_info,
            cur_min_monster_dist_norm=cur_min_monster_dist_norm,
            nearest_treasure_dist_norm=nearest_treasure_dist_norm,
            last_action=last_action,
        )
        
        # ===== 动作质量评估（必须成功，严格模式） =====
        # 与reward_system_v2相同，不支持降级到默认值
        from agent_ppo.feature.action_quality import compute_action_quality_features
        
        # 获取英雄当前血量（用于动作质量评估）— 使用统一的提取函数
        hero_hp, _ = _extract_hero_hp_info(hero)
        
        try:
            # 计算动作质量评估 (96D: 16 动作 × 6 维度)
            # 使用hero_pos_history (存储实际{x,z}位置) 而不是trajectory_history (存储coarse_cell)
            action_quality_features = compute_action_quality_features(
                hero_pos=hero_pos,
                hero_hp=hero_hp,
                hero_max_hp=self.hero_max_hp,
                monsters=[m for m in monsters],
                treasures=[t for t in treasures],
                buffs=[b for b in buffs],
                legal_action_mask=np.array(legal_action, dtype=np.float32),
                local_map=map_info,
                recent_positions=list(self.hero_pos_history) if self.hero_pos_history else None,
                flash_cooldown=int(flash_cd),
                danger_trend=self.danger_trend_value,
            )
            reward_info["action_quality_computed"] = True
        except Exception as e:
            # 严格模式：action_quality计算失败则抛出RuntimeError，不支持任何降级处理
            logger.error(f"Action quality computation FAILED (strict mode): {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Action quality features extraction failed: {str(e)}") from e
        
        # 拼接完整的255D特征向量（包含action_quality）
        feature = np.concatenate(
            [
                hero_enhanced_feat,      # self_dim (24D)
                monster1_feat,           # monster1_dim (7D)
                monster2_feat,           # monster2_dim (7D)
                treasure_feat,           # treasure_dim (12D)
                buff_feat,               # buff_dim (8D)
                decision_auxiliary_feat, # decision_auxiliary_dim (20D)
                map_and_paths_feat,      # map_dim (35D)
                temporal_memory_feat,    # temporal_memory_dim (30D)
                np.array(legal_action, dtype=np.float32),  # legal_dim (16D)
                action_quality_features,  # action_quality_dim (96D)
            ]
        )

        return feature, legal_action, [reward_total], reward_info
