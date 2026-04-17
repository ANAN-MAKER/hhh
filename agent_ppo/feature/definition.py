#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions, GAE computation for Gorge Chase PPO.
峡谷追猎 PPO 数据类定义与 GAE 计算。
"""

import numpy as np
from common_python.utils.common_func import create_cls, attached
from agent_ppo.conf.conf import Config


# ObsData: feature=145D vector, legal_action=16D mask / 特征向量与合法动作掩码
# === 修改 Task C ===
# 扩展ObsData支持GRU hidden_state
ObsData = create_cls("ObsData", feature=None, legal_action=None, hidden_state=None)

# ActData: action, d_action(greedy), prob, logprob, value / 动作、贪心动作、概率、对数概率、价值
ActData = create_cls("ActData", action=None, d_action=None, prob=None, logprob=None, value=None)

# SampleData: single-frame sample with complete PPO fields
# === 修改 Task C ===
# 扩展SampleData支持完整的GAE计算和logprob存储
SampleData = create_cls(
    "SampleData",
    # 观测与动作信息
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    old_logprob=1,  # 采样时计算的logprob
    old_value=Config.VALUE_NUM,  # 采样时的critic值
    
    # 奖励与环境信息
    reward=Config.VALUE_NUM,
    done=1,
    
    # 下一步信息（用于GAE）
    next_obs=Config.DIM_OF_OBSERVATION,
    next_legal_action=Config.ACTION_NUM,
    next_value=Config.VALUE_NUM,
    
    # PPO训练需要的字段
    advantage=Config.VALUE_NUM,
    return_=Config.VALUE_NUM,  # GAE计算的累积回报
    
    # 辅助字段（兼容旧代码）
    reward_sum=Config.VALUE_NUM,  # 等同于return_
    value=Config.VALUE_NUM,  # 等同于old_value
    prob=Config.ACTION_NUM,  # 采样时的完整概率分布
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    填充 next_value 并使用 GAE 计算优势函数。
    === 修改 Task C ===
    现在处理新的字段：next_obs, next_legal_action, old_logprob等。
    """
    # 处理终局最后一帧的next_value
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].old_value
    
    # 最后一帧的next_value已经被设置为0（见train_workflow设置）
    
    # 计算GAE
    _calc_gae(list_sample_data)
    
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE (Generalized Advantage Estimation).

    计算广义优势估计（GAE）。
    === 修改 Task C ===
    现在基于old_value和next_value计算GAE。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        # 使用old_value而非value（确保与采样阶段一致）
        delta = -sample.old_value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.return_ = gae + sample.old_value  # 累积回报
        # 兼容旧字段
        sample.reward_sum = sample.return_
        sample.value = sample.old_value
