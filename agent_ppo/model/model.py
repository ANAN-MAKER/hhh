#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=1.0):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


class MLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            make_fc_layer(in_features, hidden_features, gain=nn.init.calculate_gain("relu")),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            make_fc_layer(hidden_features, out_features, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = make_fc_layer(hidden_dim, hidden_dim, gain=nn.init.calculate_gain("relu"))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = make_fc_layer(hidden_dim, hidden_dim, gain=nn.init.calculate_gain("relu"))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return self.act(out + residual)


class RiskRewardFusionBlock(nn.Module):
    """Risk-Reward Fusion Block: 风险收益融合块
    
    该模块的目标是把当前已有的各种分散信息显式整合成一个中间语义层：
      - 地图风险 (from map_encoder)
      - 怪物威胁 (from monster_fusion)
      - 宝箱收益 (from treasure_encoder)
      - buff 收益 (from buff_encoder)
      - 时序趋势 (从 plan_encoder 的时序记忆部分)
      - 动作质量评估 (from action_plan, 包含动作质量评估信息)
    
    输出三类中间语义表示：
      1. 当前风险等级表征（risk_level: 高危/中等/安全）
      2. 当前收益机会表征（opportunity_level: 高收益/中等/保守）
      3. 决策风格建议表征（decision_style: 保守/平衡/激进）
    
    这一层让模型内部真的出现清晰的语义理解，而不是只靠大 backbone 黑箱吸收。
    """
    
    def __init__(
        self,
        threat_dim: int = 64,    # from monster_fusion
        opportunity_dim: int = 96,  # treasure_hidden + buff_hidden (64+16=80, 近似96)
        map_dim: int = 64,        # from map_encoder
        temporal_dim: int = 96,   # from plan_encoder
        output_dim: int = 128,    # 输出维度
    ):
        super().__init__()
        
        # ====================================================================
        # 1. 风险等级评估子模块
        # ====================================================================
        # 输入: 威胁信息 + 地图信息 + 时序信息 -> 风险评分
        self.risk_encoder = MLPBlock(
            in_features=threat_dim + map_dim,
            hidden_features=96,
            out_features=64,
        )
        self.risk_gate = nn.Sequential(
            make_fc_layer(64, 32, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(32, 3, gain=0.01),  # 3 维: 高危/中等/安全
        )
        
        # ====================================================================
        # 2. 收益机会评估子模块
        # ====================================================================
        # 输入: 宝箱/buff 信息 + 时序趋势 -> 收益机会评分
        self.opportunity_encoder = MLPBlock(
            in_features=opportunity_dim + temporal_dim,
            hidden_features=128,
            out_features=96,
        )
        self.opportunity_gate = nn.Sequential(
            make_fc_layer(96, 48, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(48, 3, gain=0.01),  # 3 维: 高收益/中等/保守
        )
        
        # ====================================================================
        # 3. 决策风格建议子模块
        # ====================================================================
        # 综合风险和收益，建议决策风格
        self.decision_style_fusion = MLPBlock(
            in_features=64 + 96,  # risk_encoder output + opportunity_encoder output
            hidden_features=96,
            out_features=64,
        )
        self.decision_style_gate = nn.Sequential(
            make_fc_layer(64, 32, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(32, 3, gain=0.01),  # 3 维: 保守/平衡/激进
        )
        
        # ====================================================================
        # 4. 最终融合层
        # ====================================================================
        # 把三类语义和原始编码信息融合输出
        fusion_input_dim = 64 + 96 + 64 + threat_dim + opportunity_dim + map_dim
        self.final_fusion = MLPBlock(
            in_features=fusion_input_dim,
            hidden_features=256,
            out_features=output_dim,
        )
    
    def forward(self, threat_hidden, opportunity_hidden, map_hidden, temporal_hidden):
        """
        Args:
            threat_hidden: 怪物威胁编码 (batch_size, threat_dim)
            opportunity_hidden: 宝箱/buff 编码拼接 (batch_size, opportunity_dim)
            map_hidden: 地图编码 (batch_size, map_dim)
            temporal_hidden: 时序记忆编码 (batch_size, temporal_dim)
        
        Returns:
            fused_context: 融合后的决策上下文 (batch_size, output_dim=128)
        """
        # ====================================================================
        # 风险评估
        # ====================================================================
        risk_input = torch.cat([threat_hidden, map_hidden], dim=1)
        risk_repr = self.risk_encoder(risk_input)  # (batch, 64)
        risk_scores = self.risk_gate(risk_repr)     # (batch, 3)
        
        # ====================================================================
        # 收益评估
        # ====================================================================
        opportunity_input = torch.cat([opportunity_hidden, temporal_hidden], dim=1)
        opportunity_repr = self.opportunity_encoder(opportunity_input)  # (batch, 96)
        opportunity_scores = self.opportunity_gate(opportunity_repr)    # (batch, 3)
        
        # ====================================================================
        # 决策风格
        # ====================================================================
        decision_input = torch.cat([risk_repr, opportunity_repr], dim=1)
        decision_repr = self.decision_style_fusion(decision_input)  # (batch, 64)
        decision_scores = self.decision_style_gate(decision_repr)   # (batch, 3)
        
        # ====================================================================
        # 最终融合
        # ====================================================================
        # 把中间表示和原始编码拼接
        fusion_input = torch.cat([
            risk_repr,           # (batch, 64)
            opportunity_repr,    # (batch, 96)
            decision_repr,       # (batch, 64)
            threat_hidden,       # (batch, threat_dim)
            opportunity_hidden,  # (batch, opportunity_dim)
            map_hidden,          # (batch, map_dim)
        ], dim=1)
        
        fused_context = self.final_fusion(fusion_input)  # (batch, 128)
        
        return fused_context
class Model(nn.Module):
    """Structured encoders + Risk-Reward Fusion + residual backbone + Actor/Critic dual heads.

    分组编码器 + 风险收益融合块 + 残差骨干 + Actor/Critic 双头
    
    信息流：
      环境观测 -> 空间对齐 -> 特征编码 -> 动作质量评估 
      -> 风险收益融合 -> backbone -> policy/value 输出
    
    对外保持：
      - PPO 训练入口不变
      - 16 维动作输出不变
      - legal action 使用方式不变
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_treasure_hunter_v1"
        self.device = device

        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        (
            self.self_dim,
            self.monster1_dim,
            self.monster2_dim,
            self.treasure_dim,
            self.buff_dim,
            self.progress_dim,
            self.map_dim,
            self.plan_dim,
            self.legal_dim,
        ) = Config.FEATURE_SPLIT_SHAPE

        # ====================================================================
        # STAGE 1: 感知阶段 - 各类特征编码器
        # ====================================================================
        self.self_progress_encoder = MLPBlock(self.self_dim + self.progress_dim, 64, 64)
        self.monster_encoder = MLPBlock(self.monster1_dim, 32, 32)
        self.monster_fusion = MLPBlock(64, 64, 64)  # 威胁融合
        self.treasure_encoder = MLPBlock(self.treasure_dim, 64, 64)
        self.buff_encoder = MLPBlock(self.buff_dim, 16, 16)
        self.map_encoder = MLPBlock(self.map_dim, 64, 64)
        self.plan_encoder = MLPBlock(self.plan_dim, 96, 96)  # 时序规划
        self.legal_encoder = MLPBlock(self.legal_dim, 16, 16)

        # ====================================================================
        # STAGE 2: 理解阶段 - 风险收益融合块
        # ====================================================================
        self.risk_reward_fusion = RiskRewardFusionBlock(
            threat_dim=64,         # monster_fusion output
            opportunity_dim=80,    # treasure_encoder + buff_encoder (64 + 16)
            map_dim=64,            # map_encoder output
            temporal_dim=96,       # plan_encoder output
            output_dim=128,        # 中间表示维度
        )

        # ====================================================================
        # STAGE 3: 决策阶段 - 主干网络
        # ====================================================================
        fusion_dim = 128 + 64  # risk_reward_fusion output + self_progress
        self.backbone = nn.Sequential(
            make_fc_layer(fusion_dim, 256, gain=nn.init.calculate_gain("relu")),
            nn.LayerNorm(256),
            nn.GELU(),
            ResidualBlock(256),
            ResidualBlock(256),
            make_fc_layer(256, 128, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
        )

        # ====================================================================
        # STAGE 4: 输出阶段 - 策略头和价值头
        # ====================================================================
        # Actor head / 策略头
        self.actor_head = nn.Sequential(
            make_fc_layer(128, 128, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(128, action_num, gain=0.01),
        )

        # Critic head / 价值头
        self.critic_head = nn.Sequential(
            make_fc_layer(128, 64, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(64, value_num, gain=1.0),
        )

    def forward(self, obs, inference=False):
        """
        前向传播
        
        Args:
            obs: 输入观测 (batch_size, 159)
            inference: 是否推理模式
        
        Returns:
            logits: 动作 logits (batch_size, 16)
            value: 价值估计 (batch_size, 1)
        """
        # ====================================================================
        # 特征分割
        # ====================================================================
        self_feat, monster1, monster2, treasure, buff, progress, map_local, action_plan, legal = torch.split(
            obs, Config.FEATURE_SPLIT_SHAPE, dim=1
        )

        # ====================================================================
        # STAGE 1: 特征编码
        # ====================================================================
        self_progress = self.self_progress_encoder(torch.cat([self_feat, progress], dim=1))
        monster_hidden = torch.cat([self.monster_encoder(monster1), self.monster_encoder(monster2)], dim=1)
        monster_threat = self.monster_fusion(monster_hidden)  # (batch, 64) - 威胁表示
        treasure_hidden = self.treasure_encoder(treasure)     # (batch, 64)
        buff_hidden = self.buff_encoder(buff)                 # (batch, 16)
        map_hidden = self.map_encoder(map_local)              # (batch, 64)
        plan_hidden = self.plan_encoder(action_plan)          # (batch, 96)
        legal_hidden = self.legal_encoder(legal)              # (batch, 16)

        # ====================================================================
        # STAGE 2: 风险收益融合
        # ====================================================================
        opportunity_hidden = torch.cat([treasure_hidden, buff_hidden], dim=1)  # (batch, 80)
        fused_context = self.risk_reward_fusion(
            threat_hidden=monster_threat,
            opportunity_hidden=opportunity_hidden,
            map_hidden=map_hidden,
            temporal_hidden=plan_hidden,
        )  # (batch, 128)

        # ====================================================================
        # STAGE 3: 主干网络
        # ====================================================================
        backbone_input = torch.cat([fused_context, self_progress], dim=1)  # (batch, 192)
        hidden = self.backbone(backbone_input)  # (batch, 128)

        # ====================================================================
        # STAGE 4: 输出头
        # ====================================================================
        logits = self.actor_head(hidden)  # (batch, 16) - 动作 logits
        value = self.critic_head(hidden)  # (batch, 1) - 价值估计

        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
