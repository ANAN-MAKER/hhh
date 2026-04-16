#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Priority 5: Multi-head Value Learning - Auxiliary value heads for better value estimation.

多头价值学习：添加辅助价值头用于更好的价值估计。
"""

import torch
import torch.nn as nn


class MultiHeadValueNet(nn.Module):
    """Multi-head value network with auxiliary tasks."""
    
    def __init__(self, input_dim=256, hidden_dim=64, num_auxiliary_heads=3):
        super(MultiHeadValueNet, self).__init__()
        self.num_auxiliary_heads = num_auxiliary_heads
        
        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Primary value head (main task)
        self.primary_value_head = nn.Linear(hidden_dim // 2, 1)
        
        # Auxiliary value heads (auxiliary tasks)
        # Head 1: Survival value (how long can survive)
        self.auxiliary_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_auxiliary_heads)
        ])
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]
            
        Returns:
            values: Dict with primary and auxiliary values
        """
        # Shared encoding
        shared_repr = self.shared(x)  # [batch, hidden_dim//2]
        
        # Primary value (main objective)
        primary_value = self.primary_value_head(shared_repr)  # [batch, 1]
        
        # Auxiliary values
        auxiliary_values = []
        for head in self.auxiliary_heads:
            aux_value = head(shared_repr)  # [batch, 1]
            auxiliary_values.append(aux_value)
        
        return {
            'primary_value': primary_value.squeeze(-1),  # [batch]
            'auxiliary_values': torch.cat(auxiliary_values, dim=1),  # [batch, num_aux]
            'shared_repr': shared_repr,  # [batch, hidden_dim//2] for analysis
        }


class MultiTaskValueLoss(nn.Module):
    """Compute combined loss from primary and auxiliary value heads."""
    
    def __init__(self, primary_weight=1.0, auxiliary_weight=0.1):
        super(MultiTaskValueLoss, self).__init__()
        self.primary_weight = primary_weight
        self.auxiliary_weight = auxiliary_weight
        
    def forward(self, value_outputs, targets_dict):
        """
        Args:
            value_outputs: Dict from MultiHeadValueNet.forward()
            targets_dict: Dict with 'primary_target' and optionally 'auxiliary_targets'
            
        Returns:
            total_loss: Combined loss from all heads
            loss_breakdown: Dict with individual losses
        """
        # Primary value loss (main)
        primary_value = value_outputs['primary_value']
        primary_target = targets_dict['primary_target']
        primary_loss = torch.mean((primary_value - primary_target) ** 2)
        
        # Auxiliary value losses (supporting)
        auxiliary_losses = []
        if 'auxiliary_targets' in targets_dict:
            auxiliary_targets = targets_dict['auxiliary_targets']  # [batch, num_aux]
            auxiliary_values = value_outputs['auxiliary_values']  # [batch, num_aux]
            
            for i in range(auxiliary_values.shape[1]):
                aux_loss = torch.mean((auxiliary_values[:, i] - auxiliary_targets[:, i]) ** 2)
                auxiliary_losses.append(aux_loss)
        
        # Combine losses
        total_loss = self.primary_weight * primary_loss
        if auxiliary_losses:
            avg_auxiliary_loss = torch.mean(torch.stack(auxiliary_losses))
            total_loss = total_loss + self.auxiliary_weight * avg_auxiliary_loss
        
        return total_loss, {
            'primary_loss': primary_loss.item(),
            'auxiliary_losses': [l.item() for l in auxiliary_losses] if auxiliary_losses else [],
            'total_loss': total_loss.item(),
        }


def create_auxiliary_targets(reward_breakdown):
    """
    Create auxiliary targets from reward breakdown.
    
    Args:
        reward_breakdown: Dict with reward components
        
    Returns:
        auxiliary_targets: [batch, num_aux] tensor for auxiliary heads
    """
    # Example: Create 3 auxiliary targets from reward components
    auxiliary_targets = []
    
    # Auxiliary target 1: Treasure collection probability (0-1)
    treasure_score = reward_breakdown.get('treasure_bonus', 0.0) / 100.0
    treasure_score = max(0.0, min(1.0, treasure_score))
    auxiliary_targets.append(treasure_score)
    
    # Auxiliary target 2: Survival duration (normalized 0-1)
    survival_score = reward_breakdown.get('step_ratio', 0.5)
    auxiliary_targets.append(survival_score)
    
    # Auxiliary target 3: Safety score (0-1)
    danger_penalty = reward_breakdown.get('danger_zone_penalty', 0.0)
    safety_score = 1.0 - min(1.0, abs(danger_penalty) / 10.0)
    auxiliary_targets.append(safety_score)
    
    return torch.tensor(auxiliary_targets, dtype=torch.float32).unsqueeze(0)
