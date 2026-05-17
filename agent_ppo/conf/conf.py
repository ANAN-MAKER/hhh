#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
from typing import Dict, List


def _read_positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


class NormConfig:
    MAX_ABS_COORD: float = 60000.0
    MAX_REL_COORD: float = 20000.0
    MAX_DIST: float = 30000.0
    MAX_LEVEL: float = 15.0
    MAX_MONEY: float = 20000.0
    MAX_LEVEL_DIFF: float = 14.0
    MAX_ATTACK_RANGE: float = 12000.0
    MAX_MOVE_SPEED: float = 10000.0
    MAX_ATK_SPEED: float = 5000.0
    MAX_KDA_COUNT: float = 10.0
    MAX_PHY_ATK: float = 1500.0
    MAX_MGC_ATK: float = 1500.0
    MAX_PHY_DEF: float = 1500.0
    MAX_MGC_DEF: float = 1500.0


class GameConfig:
    MODEL_SAVE_INTERVAL: int = 1800
    TIME_SCALE_ARG: int = 0
    MAX_FRAME_NO: int = 20000
    NORMAL_TOWER_SUBTYPE: int = 21
    ACTOR_TYPE_HERO: int = 0
    ACTOR_TYPE_MONSTER: int = 1
    ACTOR_TYPE_ORGAN: int = 2
    SUB_TYPE_LANE_SOLDIER: int = 11
    SUB_TYPE_TOWER: int = 21
    SUB_TYPE_SPRING_TOWER: int = 23
    SUB_TYPE_CRYSTAL: int = 24
    RIVER_SPIRIT_CONFIG_ID: int = 6827
    LANE_SOLDIER_CONFIG_IDS = {6800, 6801, 6802, 6803, 6804, 6805}
    TOWER_CONFIG_IDS = {1111, 1112}
    CRYSTAL_CONFIG_IDS = {1113, 1114}
    SPRING_TOWER_CONFIG_IDS = {44, 46}
    ORGAN_SUB_TYPES = {SUB_TYPE_TOWER, SUB_TYPE_SPRING_TOWER, SUB_TYPE_CRYSTAL}
    ORGAN_CONFIG_IDS = TOWER_CONFIG_IDS | CRYSTAL_CONFIG_IDS | SPRING_TOWER_CONFIG_IDS
    SKILL_SLOT_TYPES: List[int] = [1, 2, 3, 5]
    HERO_ID_TO_INDEX: Dict[int, int] = {
        112: 0,
        133: 1,
    }


class DebugConfig:
    ENABLE_TRACE: bool = False
    TRACE_MODE: str = "all"
    TRACE_TOPK: int = 3
    TRACE_FRAME_INTERVAL: int = 50
    TRACE_OUTPUT_DIR: str = "debug_traces"
    ENABLE_TRACE_MONITOR: bool = False
    ATTACH_REAL_CMD: bool = True
    TRACE_MONITOR_INTERVAL_SECONDS: int = 30
    TRACE_LOG_INTERVAL_FRAMES: int = 20
    TRACE_MAX_EPISODES: int = 3
    TRACE_MAX_RECORDS: int = 2000
    TRACE_MONITOR_SIDE_ONLY: bool = True


class DimConfig:
    SKILL_SLOT_FEATURES_PER_SLOT: int = 2
    SKILL_STATE_DIM: int = len(GameConfig.SKILL_SLOT_TYPES) * SKILL_SLOT_FEATURES_PER_SLOT
    SELF_HERO_DIM: int = 17 + SKILL_STATE_DIM
    ENEMY_HERO_DIM: int = 18 + SKILL_STATE_DIM
    SELF_TOWER_DIM: int = 4
    ENEMY_TOWER_DIM: int = 5
    SOLDIER_SLOT_DIM: int = 6
    SOLDIER_SLOT_COUNT: int = 4
    ENEMY_SOLDIERS_DIM: int = SOLDIER_SLOT_DIM * SOLDIER_SLOT_COUNT
    FRIENDLY_SOLDIERS_DIM: int = SOLDIER_SLOT_DIM * SOLDIER_SLOT_COUNT
    GLOBAL_CONTEXT_DIM: int = 5
    MATCHUP_DIM: int = 4
    NUM_MATCHUPS: int = 4

    FEATURE_GROUP_DIMS: Dict[str, int] = {
        "self_hero": SELF_HERO_DIM,
        "enemy_hero": ENEMY_HERO_DIM,
        "self_tower": SELF_TOWER_DIM,
        "enemy_tower": ENEMY_TOWER_DIM,
        "enemy_soldiers": ENEMY_SOLDIERS_DIM,
        "friendly_soldiers": FRIENDLY_SOLDIERS_DIM,
        "global_context": GLOBAL_CONTEXT_DIM,
        "matchup": MATCHUP_DIM,
    }

    # Active runtime feature vector size is 117:
    # 25 + 26 + 4 + 5 + 24 + 24 + 5 + 4.
    DIM_OF_FEATURE: List[int] = [sum(FEATURE_GROUP_DIMS.values())]


class Config:
    NETWORK_NAME: str = "network"
    LSTM_TIME_STEPS: int = 16
    LSTM_UNIT_SIZE: int = 512
    FEATURE_DIM: int = DimConfig.DIM_OF_FEATURE[0]
    HERO_ID_CONDITION_DIM: int = len(GameConfig.HERO_ID_TO_INDEX)
    NUM_MATCHUPS: int = DimConfig.NUM_MATCHUPS

    LABEL_SIZE_LIST: List[int] = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST: List[bool] = [True, True, True, True, True, True]
    # The target branch is conditioned on the button branch, so its legal mask is flattened to 12 * 9 = 108.
    LEGAL_ACTION_DIM: int = sum(LABEL_SIZE_LIST)
    RAW_LEGAL_ACTION_DIM: int = sum(LABEL_SIZE_LIST[:-1]) + LABEL_SIZE_LIST[0] * LABEL_SIZE_LIST[-1]

    # Sample layout used by FrameCollector and Algorithm:
    # feature+legal_action, reward, advantage, six action branches, six old probabilities,
    # six branch weights, six is_train flags, then initial LSTM cell/hidden state.
    DATA_SPLIT_SHAPE: List[int] = (
        [FEATURE_DIM + LEGAL_ACTION_DIM]
        + [1] * 8
        + LABEL_SIZE_LIST
        + [1] * 7
        + [LSTM_UNIT_SIZE, LSTM_UNIT_SIZE]
    )
    SERI_VEC_SPLIT_SHAPE: List[tuple] = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]
    INIT_LEARNING_RATE_START: float = 1e-4
    TARGET_LR: float = 1e-4
    TARGET_STEP: int = 5000
    BETA_START: float = 0.025
    LOG_EPSILON: float = 1e-6

    CLIP_PARAM: float = 0.2
    MIN_POLICY: float = 0.00001
    PPO_EPOCHS: int = 1
    PPO_MINIBATCH_SIZE: int = 512
    ADVANTAGE_NORM_EPS: float = 1e-8

    data_shapes: List[List[int]] = [
        [(FEATURE_DIM + LEGAL_ACTION_DIM) * LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[0] * LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[1] * LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[2] * LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[3] * LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[4] * LSTM_TIME_STEPS],
        [LABEL_SIZE_LIST[5] * LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_TIME_STEPS],
        [LSTM_UNIT_SIZE],
        [LSTM_UNIT_SIZE],
    ]

    LEGAL_ACTION_SIZE_LIST: List[int] = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA: float = 0.997
    LAMDA: float = 0.95

    USE_GRAD_CLIP: bool = True
    GRAD_CLIP_RANGE: float = 0.5
    GRAD_CLIP_ENCODER: float = 1.0
    GRAD_CLIP_LSTM: float = 0.5
    GRAD_CLIP_HEAD: float = 0.5

    BETA_END: float = 0.005
    BETA_ANNEAL_STEPS: int = 50000
    CLIP_PARAM_START: float = 0.2
    CLIP_PARAM_END: float = 0.1
    CLIP_ANNEAL_STEPS: int = 50000

    SAMPLE_DIM: int = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
    SEED: int = 42
    # Default to one thread for compatibility; override from env when needed.
    TORCH_NUM_THREADS: int = _read_positive_int_env("KAIWU_TORCH_NUM_THREADS", 1)
    TORCH_NUM_INTEROP_THREADS: int = _read_positive_int_env("KAIWU_TORCH_NUM_INTEROP_THREADS", 1)
