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
class Model(nn.Module):
    """Structured encoders + residual backbone + Actor/Critic dual heads.

    分组编码器 + 残差骨干 + Actor/Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_treasure_hunter"
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

        self.self_progress_encoder = MLPBlock(self.self_dim + self.progress_dim, 64, 64)
        self.monster_encoder = MLPBlock(self.monster1_dim, 32, 32)
        self.monster_fusion = MLPBlock(64, 64, 64)
        self.treasure_encoder = MLPBlock(self.treasure_dim, 64, 64)
        self.buff_encoder = MLPBlock(self.buff_dim, 16, 16)
        self.map_encoder = MLPBlock(self.map_dim, 64, 64)
        self.plan_encoder = MLPBlock(self.plan_dim, 96, 96)
        self.legal_encoder = MLPBlock(self.legal_dim, 16, 16)

        fusion_dim = 64 + 64 + 64 + 16 + 64 + 96 + 16
        self.backbone = nn.Sequential(
            make_fc_layer(fusion_dim, 256, gain=nn.init.calculate_gain("relu")),
            nn.LayerNorm(256),
            nn.GELU(),
            ResidualBlock(256),
            ResidualBlock(256),
            make_fc_layer(256, 128, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
        )

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
        self_feat, monster1, monster2, treasure, buff, progress, map_local, action_plan, legal = torch.split(
            obs, Config.FEATURE_SPLIT_SHAPE, dim=1
        )

        self_progress = self.self_progress_encoder(torch.cat([self_feat, progress], dim=1))
        monster_hidden = torch.cat([self.monster_encoder(monster1), self.monster_encoder(monster2)], dim=1)
        monster_hidden = self.monster_fusion(monster_hidden)
        treasure_hidden = self.treasure_encoder(treasure)
        buff_hidden = self.buff_encoder(buff)
        map_hidden = self.map_encoder(map_local)
        plan_hidden = self.plan_encoder(action_plan)
        legal_hidden = self.legal_encoder(legal)

        hidden = self.backbone(
            torch.cat(
                [
                    self_progress,
                    monster_hidden,
                    treasure_hidden,
                    buff_hidden,
                    map_hidden,
                    plan_hidden,
                    legal_hidden,
                ],
                dim=1,
            )
        )
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
