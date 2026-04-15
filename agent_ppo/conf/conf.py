#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Feature dimensions / 特征维度（共159维，五大类）
    # A. 自身状态特征 (24D): 英雄位置速度技能进度
    # B. 外部实体特征 (34D): 怪物宝箱buff
    # C. 地图与路径特征 (35D): 局部地图方向质量逃生通路
    # D. 时序与记忆特征 (30D): 历史轨迹危险趋势探索记忆
    # E. 决策辅助特征 (20D): 风险收益闪现评分
    # + 合法动作掩码 (16D)
    FEATURES = [
        24,  # A: hero_self_enhanced 英雄自身增强 (位置2+速度3+技能4+进度7+其他8)
        34,  # B: external_entities 外部实体 (怪物14+宝箱12+buff8)
        35,  # C: map_and_paths 地图与路径 (地图25+方向8+逃生2)
        30,  # D: temporal_memory 时序与记忆 (轨迹3+危险4+收益3+记忆5+动作5+趋势5)
        20,  # E: decision_auxiliary 决策辅助 (风险4+收益4+闪现4+综合8)
        16,  # legal_action_as_feature 合法动作掩码
    ]
    
    # 模型期望的 9 个分组（兼容原有模型架构）
    # 映射: [A分解, B分解*3, E作进度, C作地图, D作规划, 合法]
    FEATURE_SPLIT_SHAPE = [
        24,  # self_dim: A的完整部分（自身状态）
        7,   # monster1_dim: B的怪物部分的一半
        7,   # monster2_dim: B的怪物部分的另一半
        12,  # treasure_dim: B的宝藏部分
        8,   # buff_dim: B的buff部分
        20,  # progress_dim: E作为决策辅助进度信息
        35,  # map_dim: C的完整部分（地图与路径）
        30,  # plan_dim: D作为时序规划信息
        16,  # legal_dim: 合法动作掩码
    ]
    
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)  # 159维
    DIM_OF_OBSERVATION = FEATURE_LEN  # 网络输入维度: 159D

    # Action space / 动作空间：8个移动方向 + 8个闪现方向
    ACTION_NUM = 16

    # Value head / 价值头：单头综合价值
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数（更偏长时序任务）
    GAMMA = 0.995
    LAMDA = 0.97
    INIT_LEARNING_RATE_START = 0.00025
    BETA_START = 0.002
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
