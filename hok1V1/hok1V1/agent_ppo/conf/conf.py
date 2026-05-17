#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    HERO_IDS = (112, 133)
    CAMP_BLUE = 1
    CAMP_RED = 2

    BUTTON_NONE = 0
    BUTTON_NOOP = 1
    BUTTON_MOVE = 2
    BUTTON_ATTACK = 3
    BUTTON_SKILL_1 = 4
    BUTTON_SKILL_2 = 5
    BUTTON_SKILL_3 = 6
    BUTTON_HEAL = 7
    BUTTON_SUMMONER = 8
    BUTTON_RECALL = 9
    BUTTON_SKILL_4 = 10
    BUTTON_EQUIPMENT_SKILL = 11
    SKILL_BUTTONS = (BUTTON_SKILL_1, BUTTON_SKILL_2, BUTTON_SKILL_3, BUTTON_SKILL_4)

    TARGET_NONE = 0
    TARGET_ENEMY = 1
    TARGET_SELF = 2
    TARGET_SOLDIERS = (3, 4, 5, 6)
    TARGET_TOWER = 7
    TARGET_MONSTER = 8

    TOWER_SUB_TYPES = (21,)
    CRYSTAL_SUB_TYPES = (24, 25)

    HERO_SIGHT_FALLBACK = 7000.0
    DEFENSE_TOWER_SIGHT_FALLBACK = 8800.0
    SPRING_TOWER_SIGHT_FALLBACK = 8000.0
    MAX_FRAME_NO = 20000.0
    MAX_GOLD_DIFF = 6000.0
    MAX_EXP_DIFF = 6000.0

    UNIT_FEATURE_DIM = 24
    GLOBAL_FEATURE_DIM = 28
    SOLDIER_SLOT_COUNT = 4
    ORGAN_SLOT_COUNT = 2
    TARGET_SLOT_COUNT = 9

    REWARD_STAGE = "A"
    ENABLE_TOWER_DANGER_REWARD = False
    ENABLE_PUSH_ASSIST_REWARD = True
    ENABLE_LANE_FOLLOW_REWARD = True
    ENABLE_FINISH_REWARD = True
    ENABLE_SKILL_REWARD = False
    ENABLE_FULL_BUFF_FEATURE = False
    USE_TARGET_ATTENTION = False
    USE_RECURRENT = True

    REWARD_WEIGHT_DICT = {
        "reward_hp": 2.0,
        "reward_tower": 12.0,
        "reward_gold": 0.008,
        "reward_exp": 0.008,
        "reward_last_hit": 0.5,
        "reward_death": -1.0,
        "reward_kill": 1.0,
        "reward_forward": 0.03,
        "reward_win": 3.0,
        "reward_tower_danger": 1.0,
        "reward_push_assist": 1.0,
        "reward_lane_follow": 1.0,
        "reward_finish": 1.0,
    }
    TIME_SCALE_ARG = 0
    MODEL_SAVE_INTERVAL = 1800


class DimConfig:
    UNIT_FEATURE_DIM = GameConfig.UNIT_FEATURE_DIM
    GLOBAL_FEATURE_DIM = GameConfig.GLOBAL_FEATURE_DIM
    UNIT_SLOT_COUNT = (
        1
        + 1
        + GameConfig.SOLDIER_SLOT_COUNT
        + GameConfig.SOLDIER_SLOT_COUNT
        + GameConfig.ORGAN_SLOT_COUNT
        + GameConfig.ORGAN_SLOT_COUNT
    )
    DIM_OF_FEATURE = [UNIT_SLOT_COUNT * UNIT_FEATURE_DIM + GLOBAL_FEATURE_DIM]


class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    FEATURE_DIM = DimConfig.DIM_OF_FEATURE[0]
    LEGAL_ACTION_DIM = 85
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [True, True, True, True, True, True]

    DATA_SPLIT_SHAPE = [
        FEATURE_DIM + LEGAL_ACTION_DIM,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]

    INIT_LEARNING_RATE_START = 8e-4
    TARGET_LR = 1e-4
    TARGET_STEP = 6000
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001

    UNIT_EMBED_DIM = 64
    GROUP_EMBED_DIM = 96
    TARGET_EMBED_DIM = 64

    data_shapes = [
        [(FEATURE_DIM + LEGAL_ACTION_DIM) * LSTM_TIME_STEPS],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
