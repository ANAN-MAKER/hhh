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
    
    # 模型期望的 10 个分组（扩展以包含action_quality）
    # 结构: [A自身] + [B怪物×2+宝藏+buff] + [E决策辅助] + [C地图路径] + [D时序记忆] + [合法动作] + [动作质量]
    FEATURE_SPLIT_SHAPE = [
        24,  # A: 英雄自身状态 (位置、速度、技能、进度)
        7,   # B1: 怪物1特征
        7,   # B2: 怪物2特征
        12,  # B3: 宝藏特征
        8,   # B4: Buff特征
        20,  # E: 决策辅助特征 (决策支持信息，非动作质量本身)
        35,  # C: 地图与路径特征 (局部地图+方向质量+逃生通路)
        30,  # D: 时序与记忆特征 (历史轨迹、危险趋势、探索记忆)
        16,  # 合法动作掩码
        96,  # 动作质量评估 (16动作×6维度，独立于决策辅助)
    ]
    
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)  # 255维（从159D扩展）
    DIM_OF_OBSERVATION = FEATURE_LEN  # 网络输入维度: 255D

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
