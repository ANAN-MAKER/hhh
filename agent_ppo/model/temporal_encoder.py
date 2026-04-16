#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Priority 4: Temporal Modeling - GRU-based temporal encoding for monster trend prediction.

时序建模：使用GRU进行怪物行为趋势预测。
"""

import torch
import torch.nn as nn
import numpy as np


class TemporalEncoder(nn.Module):
    """GRU-based temporal encoder for sequence modeling."""
    
    def __init__(self, input_dim=10, hidden_dim=16, num_layers=1, dropout=0.0):
        super(TemporalEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU for temporal encoding
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection to temporal features (30D expected)
        self.output_proj = nn.Linear(hidden_dim, 30)
        
    def forward(self, sequence):
        """
        Args:
            sequence: [batch_size, seq_len, input_dim] temporal sequence
            
        Returns:
            temporal_features: [batch_size, 30] encoded temporal features
        """
        # GRU forward pass
        output, _ = self.gru(sequence)  # [batch, seq_len, hidden_dim]
        
        # Use last timestep for final encoding
        last_output = output[:, -1, :]  # [batch, hidden_dim]
        
        # Project to temporal feature dimension
        temporal_features = self.output_proj(last_output)  # [batch, 30]
        
        return temporal_features


class MonsterTrendPredictor(nn.Module):
    """Predict monster behavior trends from historical states."""
    
    def __init__(self, obs_dim=255, trend_dim=8):
        super(MonsterTrendPredictor, self).__init__()
        self.trend_dim = trend_dim
        
        # Simple trend prediction head
        self.trend_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, trend_dim),  # 8D: 4 monsters × 2 features (distance trend, threat level)
        )
        
    def forward(self, obs):
        """
        Args:
            obs: [batch_size, obs_dim] observation
            
        Returns:
            monster_trends: [batch_size, trend_dim] predicted monster trends
        """
        return torch.tanh(self.trend_net(obs))  # Outputs in [-1, 1]


def create_temporal_sequence(state_history, window_size=5):
    """
    Create temporal sequences from state history for GRU input.
    
    Args:
        state_history: List of [10D] state snapshots (e.g., distance metrics)
        window_size: Number of timesteps in sequence
        
    Returns:
        sequence: [1, window_size, 10] shaped tensor for GRU
    """
    if len(state_history) < window_size:
        # Pad with zeros if history too short
        pad_size = window_size - len(state_history)
        sequence = [np.zeros(10)] * pad_size + state_history[-window_size:]
    else:
        sequence = state_history[-window_size:]
    
    return torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0)
