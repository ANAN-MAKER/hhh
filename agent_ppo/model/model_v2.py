#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model v2 for Gorge Chase PPO - Ability-Driven Layered Architecture.
峡谷追猎 PPO 神经网络模型 v2 - 能力驱动分层架构。

Architecture Overview / 架构概览:
  [159D Input]
    ↓
  [Stage 1] Input Encoding Layer (6 encoders) → 368D
    ├─ HeroStateEncoder (24D → 64D)
    ├─ LocalMapEncoder (35D → 64D)
    ├─ MonsterThreatEncoder (14D → 64D)
    ├─ ResourceTargetEncoder (20D → 64D)
    ├─ TemporalMemoryEncoder (30D → 96D)
    └─ LegalActionEncoder (16D → 16D)
    ↓
  [Stage 2] Situation Understanding Layer (3 fusion blocks) → 192D
    ├─ ConsolidatedThreatFusion (224D → 128D)
    ├─ OpportunitiesFusion (256D → 128D)
    └─ GlobalSituationFusion (272D → 192D)
    ↓
  [Stage 3] Action Planning Layer (1 planning block) → 128D
    └─ ActionPlanningHead (384D → 128D)
    ↓
  [Stage 4] Policy & Value Output Layer (2 heads)
    ├─ PolicyHead: 128D → 16D (dual-layer: action_type + direction)
    └─ ValueHead: 192D → 1D (multi-dimensional: 5 values → 1 aggregate)
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


class EncoderBlock(nn.Module):
    """MLP encoder with LayerNorm and GELU activation.
    
    简单的编码块：Linear → LayerNorm → GELU → Linear
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            make_fc_layer(in_features, hidden_features, gain=nn.init.calculate_gain("relu")),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            make_fc_layer(hidden_features, out_features, gain=nn.init.calculate_gain("relu")),
        )

    def forward(self, x):
        return self.net(x)


class FusionBlock(nn.Module):
    """Fusion block with dimension reduction.
    
    融合块：拼接 + 降维处理
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            make_fc_layer(in_features, hidden_features, gain=nn.init.calculate_gain("relu")),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            make_fc_layer(hidden_features, out_features, gain=nn.init.calculate_gain("relu")),
        )

    def forward(self, *inputs):
        # Concatenate all input tensors
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = torch.cat(inputs, dim=-1)
        return self.net(x)


# ============================================================================
# Stage 1: Input Encoding Layer
# 第1阶段：输入编码层 - 基础感知类
# ============================================================================

class HeroStateEncoder(nn.Module):
    """Encode hero's self state (position, speed, skills, progress).
    
    编码自身状态 (位置、速度、技能、进度)。
    输入: 24D (hero_self)
    输出: 64D
    """
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=64):
        super().__init__()
        self.encoder = EncoderBlock(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        return self.encoder(x)


class LocalMapEncoder(nn.Module):
    """Encode local map structure and safety.
    
    编码局部地图结构和安全性。
    输入: 35D (map_feature)
    输出: 64D
    """
    def __init__(self, input_dim=35, hidden_dim=64, output_dim=64):
        super().__init__()
        self.encoder = EncoderBlock(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        return self.encoder(x)


class MonsterThreatEncoder(nn.Module):
    """Encode threat from dual monsters.
    
    编码双怪物威胁。
    输入: 14D (monster1: 7D + monster2: 7D)
    输出: 64D
    """
    def __init__(self, monster1_dim=7, monster2_dim=7, hidden_dim=64, output_dim=64):
        super().__init__()
        # Separate encoders for each monster
        self.monster1_encoder = make_fc_layer(monster1_dim, hidden_dim)
        self.monster2_encoder = make_fc_layer(monster2_dim, hidden_dim)
        
        # Fusion of dual threats
        self.fusion = EncoderBlock(2 * hidden_dim, hidden_dim, output_dim)

    def forward(self, monster1, monster2):
        m1_feat = torch.relu(self.monster1_encoder(monster1))
        m2_feat = torch.relu(self.monster2_encoder(monster2))
        combined = torch.cat([m1_feat, m2_feat], dim=-1)
        return self.fusion(combined)


class ResourceTargetEncoder(nn.Module):
    """Encode resource opportunities (treasures and buffs).
    
    编码资源机会 (宝箱和buff)。
    输入: 20D (treasure: 12D + buff: 8D)
    输出: 64D
    """
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=64):
        super().__init__()
        self.encoder = EncoderBlock(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        return self.encoder(x)


class TemporalMemoryEncoder(nn.Module):
    """Encode temporal trends and historical patterns.
    
    编码时间趋势和历史模式。
    输入: 30D (plan_feature - 包含历史轨迹、危险趋势等)
    输出: 96D (比其他编码器更大，因为时序信息重要)
    """
    def __init__(self, input_dim=30, hidden_dim=96, output_dim=96):
        super().__init__()
        self.encoder = EncoderBlock(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        return self.encoder(x)


class LegalActionEncoder(nn.Module):
    """Encode legal action mask (which actions are valid).
    
    编码合法动作掩码。
    输入: 16D (legal_action)
    输出: 16D (可以直接通过或轻度编码)
    """
    def __init__(self, input_dim=16, output_dim=16):
        super().__init__()
        # Light encoding for legal mask
        self.encoder = make_fc_layer(input_dim, output_dim)

    def forward(self, x):
        return self.encoder(x)


# ============================================================================
# Stage 2: Situation Understanding Layer
# 第2阶段：局势理解层 - 局势理解类
# ============================================================================

class ConsolidatedThreatFusion(nn.Module):
    """Fuse threat information from monsters, map, and temporal trends.
    
    融合怪物、地图、时间信息形成威胁评估。
    输入: threat(64D) + map(64D) + temporal(96D) = 224D
    输出: 128D
    """
    def __init__(self, input_dim=224, hidden_dim=160, output_dim=128):
        super().__init__()
        self.fusion = FusionBlock(input_dim, hidden_dim, output_dim)

    def forward(self, threat_feat, map_feat, temporal_feat):
        return self.fusion(threat_feat, map_feat, temporal_feat)


class OpportunitiesFusion(nn.Module):
    """Fuse resource opportunities with self state and threat context.
    
    融合资源机会与自身状态和威胁背景。
    输入: resource(64D) + hero(64D) + threat(128D) = 256D
    输出: 128D
    """
    def __init__(self, input_dim=256, hidden_dim=160, output_dim=128):
        super().__init__()
        self.fusion = FusionBlock(input_dim, hidden_dim, output_dim)

    def forward(self, resource_feat, hero_feat, threat_fusion):
        return self.fusion(resource_feat, hero_feat, threat_fusion)


class GlobalSituationFusion(nn.Module):
    """Fuse all understanding into global situation assessment.
    
    融合所有理解形成全局局势评估。
    输入: threat_fusion(128D) + opportunity_fusion(128D) + legal(16D) = 272D
    输出: 192D (最大的中间表示)
    """
    def __init__(self, input_dim=272, hidden_dim=200, output_dim=192):
        super().__init__()
        self.fusion = FusionBlock(input_dim, hidden_dim, output_dim)

    def forward(self, threat_fusion, opportunity_fusion, legal_feat):
        return self.fusion(threat_fusion, opportunity_fusion, legal_feat)


# ============================================================================
# Stage 3: Action Planning Layer
# 第3阶段：规划评估层
# ============================================================================

class ActionPlanningHead(nn.Module):
    """Evaluate candidate actions based on global situation.
    
    基于全局局势评估候选动作。
    输入: global_situation(192D) + map(64D) + threat(64D) = 384D
    输出: 128D (动作规划特征)
    """
    def __init__(self, input_dim=384, hidden_dim=256, output_dim=128):
        super().__init__()
        self.planning = FusionBlock(input_dim, hidden_dim, output_dim)

    def forward(self, global_situation, map_feat, threat_feat):
        return self.planning(global_situation, map_feat, threat_feat)


# ============================================================================
# Stage 4: Policy & Value Output Layer
# 第4阶段：策略/价值输出层
# ============================================================================

class PolicyHead(nn.Module):
    """Dual-layer policy output: action type + direction.
    
    双层策略输出：动作类型（移动vs闪现）+ 方向选择。
    
    输入: 128D (action_planning)
    输出: 16D (hard constraint mapping)
      - 移动 (0-7): P(move) * P(direction)
      - 闪现 (8-15): P(flash) * P(direction)
    """
    def __init__(self, input_dim=128):
        super().__init__()
        # Layer 1: Action type (move vs flash)
        self.action_type_head = nn.Sequential(
            make_fc_layer(input_dim, 64, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(64, 2, gain=0.01),  # [move_logit, flash_logit]
        )
        
        # Layer 2: Direction (8 directions)
        self.direction_head = nn.Sequential(
            make_fc_layer(input_dim, 64, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(64, 8, gain=0.01),  # 8 direction logits
        )

    def forward(self, action_planning):
        """
        Returns: 16D action probability distribution
        """
        action_type_logits = self.action_type_head(action_planning)  # [batch, 2]
        direction_logits = self.direction_head(action_planning)      # [batch, 8]
        
        # Apply softmax to get probabilities
        action_type_probs = torch.softmax(action_type_logits, dim=-1)  # [batch, 2]
        direction_probs = torch.softmax(direction_logits, dim=-1)      # [batch, 8]
        
        # Combine: P(action) = P(type) * P(direction)
        move_probs = action_type_probs[:, 0:1] * direction_probs      # [batch, 8]
        flash_probs = action_type_probs[:, 1:2] * direction_probs     # [batch, 8]
        
        # Final 16D action probability
        final_probs = torch.cat([move_probs, flash_probs], dim=-1)    # [batch, 16]
        
        return final_probs


class ValueHead(nn.Module):
    """Multi-dimensional value estimation: 5 value aspects → 1 aggregate value.
    
    多维价值评估：5个价值维度 → 1个综合价值。
    
    输入: 192D (global_situation)
    输出: 1D (综合价值标量)
    
    包含5个维度:
      1. Survival value (生存价值)
      2. Scoring value (得分价值)
      3. Buff value (Buff利用价值)
      4. Flash opportunity (闪现机会价值)
      5. Stability value (局势稳定性价值)
    """
    def __init__(self, input_dim=192, hidden_dim=128):
        super().__init__()
        
        # 5 independent value heads
        self.survival_head = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(hidden_dim, 1, gain=1.0),
        )
        
        self.scoring_head = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(hidden_dim, 1, gain=1.0),
        )
        
        self.buff_head = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(hidden_dim, 1, gain=1.0),
        )
        
        self.flash_head = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(hidden_dim, 1, gain=1.0),
        )
        
        self.stability_head = nn.Sequential(
            make_fc_layer(input_dim, hidden_dim, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(hidden_dim, 1, gain=1.0),
        )
        
        # Aggregate function for 5 dimensions → 1 value
        self.aggregate = nn.Sequential(
            make_fc_layer(5, 8, gain=nn.init.calculate_gain("relu")),
            nn.GELU(),
            make_fc_layer(8, 1, gain=1.0),
        )

    def forward(self, global_situation):
        """
        Returns: 1D value scalar
        """
        v_survival = self.survival_head(global_situation)      # [batch, 1]
        v_scoring = self.scoring_head(global_situation)        # [batch, 1]
        v_buff = self.buff_head(global_situation)              # [batch, 1]
        v_flash = self.flash_head(global_situation)            # [batch, 1]
        v_stability = self.stability_head(global_situation)    # [batch, 1]
        
        # Concatenate all value dimensions
        values_concat = torch.cat(
            [v_survival, v_scoring, v_buff, v_flash, v_stability],
            dim=-1
        )  # [batch, 5]
        
        # Aggregate to single value
        final_value = self.aggregate(values_concat)  # [batch, 1]
        
        return final_value


# ============================================================================
# Complete Model
# ============================================================================

class ModelV2(nn.Module):
    """Ability-Driven Layered Architecture for Gorge Chase PPO.
    
    能力驱动分层架构 - 完整模型。
    
    This model restructures from a monolithic black-box into a clear
    capability-driven 4-stage pipeline:
      1. Perception (Input Encoding)
      2. Understanding (Situation Fusion)
      3. Planning (Action Evaluation)
      4. Decision (Policy & Value Output)
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_v2_layered"
        self.device = device

        # Extract feature dimensions from config
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
        # Stage 1: Input Encoding Layer (6 encoders)
        # ====================================================================
        self.hero_encoder = HeroStateEncoder(
            input_dim=self.self_dim,
            hidden_dim=64,
            output_dim=64
        )
        
        self.map_encoder = LocalMapEncoder(
            input_dim=self.map_dim,
            hidden_dim=64,
            output_dim=64
        )
        
        self.threat_encoder = MonsterThreatEncoder(
            monster1_dim=self.monster1_dim,
            monster2_dim=self.monster2_dim,
            hidden_dim=64,
            output_dim=64
        )
        
        self.resource_encoder = ResourceTargetEncoder(
            input_dim=self.treasure_dim + self.buff_dim,
            hidden_dim=64,
            output_dim=64
        )
        
        self.temporal_encoder = TemporalMemoryEncoder(
            input_dim=self.plan_dim,
            hidden_dim=96,
            output_dim=96
        )
        
        self.legal_encoder = LegalActionEncoder(
            input_dim=self.legal_dim,
            output_dim=16
        )

        # ====================================================================
        # Stage 2: Situation Understanding Layer (3 fusion blocks)
        # ====================================================================
        self.threat_fusion = ConsolidatedThreatFusion(
            input_dim=64 + 64 + 96,  # threat + map + temporal
            hidden_dim=160,
            output_dim=128
        )
        
        self.opportunity_fusion = OpportunitiesFusion(
            input_dim=64 + 64 + 128,  # resource + hero + threat_fusion
            hidden_dim=160,
            output_dim=128
        )
        
        self.global_fusion = GlobalSituationFusion(
            input_dim=128 + 128 + 16,  # threat_fusion + opportunity_fusion + legal
            hidden_dim=200,
            output_dim=192
        )

        # ====================================================================
        # Stage 3: Action Planning Layer (1 planning block)
        # ====================================================================
        self.action_planning = ActionPlanningHead(
            input_dim=192 + 64 + 64,  # global_situation + map + threat
            hidden_dim=256,
            output_dim=128
        )

        # ====================================================================
        # Stage 4: Policy & Value Output Layer (2 heads)
        # ====================================================================
        self.policy_head = PolicyHead(input_dim=128)
        self.value_head = ValueHead(input_dim=192, hidden_dim=128)

    def forward(self, obs, inference=False):
        """Forward pass through the complete model.
        
        Args:
            obs: [batch_size, 159] - input features (9 groups)
            inference: bool - inference mode flag (unused for now, for compatibility)
        
        Returns:
            logits: [batch_size, 16] - action probabilities
            value: [batch_size, 1] - state value
        """
        
        # Split input into 9 feature groups
        (
            self_feat,
            monster1,
            monster2,
            treasure,
            buff,
            progress,
            map_local,
            action_plan,
            legal
        ) = torch.split(obs, Config.FEATURE_SPLIT_SHAPE, dim=1)
        
        # ====================================================================
        # Stage 1: Input Encoding
        # ====================================================================
        hero_encoded = self.hero_encoder(self_feat)                 # [batch, 64]
        map_encoded = self.map_encoder(map_local)                   # [batch, 64]
        threat_encoded = self.threat_encoder(monster1, monster2)    # [batch, 64]
        
        # Combine treasure and buff for resource encoding
        resources = torch.cat([treasure, buff], dim=-1)             # [batch, 20]
        resource_encoded = self.resource_encoder(resources)         # [batch, 64]
        
        temporal_encoded = self.temporal_encoder(action_plan)       # [batch, 96]
        legal_encoded = self.legal_encoder(legal)                   # [batch, 16]
        
        # ====================================================================
        # Stage 2: Situation Understanding
        # ====================================================================
        threat_fusion_output = self.threat_fusion(
            threat_encoded,
            map_encoded,
            temporal_encoded
        )  # [batch, 128]
        
        opportunity_fusion_output = self.opportunity_fusion(
            resource_encoded,
            hero_encoded,
            threat_fusion_output
        )  # [batch, 128]
        
        global_situation = self.global_fusion(
            threat_fusion_output,
            opportunity_fusion_output,
            legal_encoded
        )  # [batch, 192]
        
        # ====================================================================
        # Stage 3: Action Planning
        # ====================================================================
        action_planning_output = self.action_planning(
            global_situation,
            map_encoded,
            threat_encoded
        )  # [batch, 128]
        
        # ====================================================================
        # Stage 4: Policy & Value Output
        # ====================================================================
        logits = self.policy_head(action_planning_output)  # [batch, 16]
        value = self.value_head(global_situation)          # [batch, 1]
        
        return logits, value

    def set_train_mode(self):
        """Set model to training mode."""
        self.train()

    def set_eval_mode(self):
        """Set model to evaluation mode."""
        self.eval()
