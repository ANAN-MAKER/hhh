#!/usr/bin/env python3
"""Test for Priority 4-5: Temporal Modeling & Multi-head Value Learning"""

import sys
import numpy as np
import torch

sys.path.insert(0, '/workspace/code')

from agent_ppo.model.temporal_encoder import TemporalEncoder, MonsterTrendPredictor, create_temporal_sequence
from agent_ppo.model.multi_head_value import MultiHeadValueNet, MultiTaskValueLoss, create_auxiliary_targets


def test_temporal_encoder():
    """Test TemporalEncoder for time-series modeling."""
    print("=" * 70)
    print("TEST 1: Temporal Encoder (Priority 4)")
    print("=" * 70)
    
    # Create encoder
    encoder = TemporalEncoder(input_dim=10, hidden_dim=16, num_layers=1)
    
    # Create dummy sequence [batch=2, seq_len=5, input_dim=10]
    sequence = torch.randn(2, 5, 10)
    
    # Forward pass
    try:
        output = encoder(sequence)
        
        expected_shape = (2, 30)  # Should output 30D temporal features
        actual_shape = output.shape
        
        shape_ok = actual_shape == torch.Size(expected_shape)
        
        print(f"  Input shape: {sequence.shape}")
        print(f"  Output shape: {actual_shape}")
        print(f"  Expected: {expected_shape}")
        print(f"  Shape correct: {shape_ok}")
        
        return shape_ok
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_monster_trend_predictor():
    """Test monster trend prediction."""
    print("\n" + "=" * 70)
    print("TEST 2: Monster Trend Predictor (Priority 4)")
    print("=" * 70)
    
    # Create predictor
    predictor = MonsterTrendPredictor(obs_dim=255, trend_dim=8)
    
    # Create dummy observation [batch=4, obs_dim=255]
    obs = torch.randn(4, 255)
    
    try:
        trends = predictor(obs)
        
        expected_shape = (4, 8)
        actual_shape = trends.shape
        
        # Check output is in [-1, 1] (tanh output)
        in_range = (trends >= -1.0).all() and (trends <= 1.0).all()
        
        print(f"  Input shape: {obs.shape}")
        print(f"  Output shape: {actual_shape}")
        print(f"  Expected shape: {expected_shape}")
        print(f"  Shape correct: {actual_shape == torch.Size(expected_shape)}")
        print(f"  Output in [-1, 1]: {in_range}")
        
        return actual_shape == torch.Size(expected_shape) and in_range
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_temporal_sequence_creation():
    """Test temporal sequence creation from history."""
    print("\n" + "=" * 70)
    print("TEST 3: Temporal Sequence Creation (Priority 4)")
    print("=" * 70)
    
    # Create mock state history
    state_history = [np.random.randn(10) for _ in range(10)]
    
    try:
        sequence = create_temporal_sequence(state_history, window_size=5)
        
        expected_shape = (1, 5, 10)
        actual_shape = sequence.shape
        
        print(f"  History length: {len(state_history)}")
        print(f"  Window size: 5")
        print(f"  Output shape: {actual_shape}")
        print(f"  Expected: {expected_shape}")
        print(f"  Shape correct: {actual_shape == torch.Size(expected_shape)}")
        
        return actual_shape == torch.Size(expected_shape)
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_multi_head_value_net():
    """Test multi-head value network."""
    print("\n" + "=" * 70)
    print("TEST 4: Multi-Head Value Network (Priority 5)")
    print("=" * 70)
    
    # Create network
    net = MultiHeadValueNet(input_dim=256, hidden_dim=64, num_auxiliary_heads=3)
    
    # Create dummy input [batch=4, input_dim=256]
    x = torch.randn(4, 256)
    
    try:
        outputs = net(x)
        
        # Check outputs
        primary_shape = outputs['primary_value'].shape
        auxiliary_shape = outputs['auxiliary_values'].shape
        repr_shape = outputs['shared_repr'].shape
        
        primary_ok = primary_shape == torch.Size([4])
        auxiliary_ok = auxiliary_shape == torch.Size([4, 3])
        repr_ok = repr_shape == torch.Size([4, 32])
        
        print(f"  Input shape: {x.shape}")
        print(f"  Primary value shape: {primary_shape} (expected [4])")
        print(f"  Auxiliary values shape: {auxiliary_shape} (expected [4, 3])")
        print(f"  Shared repr shape: {repr_shape} (expected [4, 32])")
        print(f"  All shapes correct: {primary_ok and auxiliary_ok and repr_ok}")
        
        return primary_ok and auxiliary_ok and repr_ok
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_multi_task_loss():
    """Test multi-task value loss computation."""
    print("\n" + "=" * 70)
    print("TEST 5: Multi-Task Value Loss (Priority 5)")
    print("=" * 70)
    
    # Create network and loss
    net = MultiHeadValueNet(input_dim=256, hidden_dim=64, num_auxiliary_heads=3)
    loss_fn = MultiTaskValueLoss(primary_weight=1.0, auxiliary_weight=0.1)
    
    # Create dummy data
    x = torch.randn(4, 256)
    
    try:
        # Forward pass
        outputs = net(x)
        
        # Create targets
        targets = {
            'primary_target': torch.randn(4),  # [batch]
            'auxiliary_targets': torch.randn(4, 3),  # [batch, num_aux]
        }
        
        # Compute loss
        total_loss, breakdown = loss_fn(outputs, targets)
        
        loss_is_scalar = total_loss.item() > 0
        breakdown_ok = ('primary_loss' in breakdown and 'auxiliary_losses' in breakdown)
        
        print(f"  Total loss: {total_loss.item():.6f}")
        print(f"  Primary loss: {breakdown['primary_loss']:.6f}")
        print(f"  Auxiliary losses: {breakdown['auxiliary_losses']}")
        print(f"  Loss is positive scalar: {loss_is_scalar}")
        print(f"  Breakdown correct: {breakdown_ok}")
        
        return loss_is_scalar and breakdown_ok
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_auxiliary_targets_creation():
    """Test auxiliary targets creation from reward breakdown."""
    print("\n" + "=" * 70)
    print("TEST 6: Auxiliary Targets Creation (Priority 5)")
    print("=" * 70)
    
    # Create mock reward breakdown
    reward_breakdown = {
        'treasure_bonus': 25.0,
        'step_ratio': 0.7,
        'danger_zone_penalty': 5.0,
    }
    
    try:
        targets = create_auxiliary_targets(reward_breakdown)
        
        shape_ok = targets.shape == torch.Size([1, 3])
        values_in_range = (targets >= 0.0).all() and (targets <= 1.0).all()
        
        print(f"  Reward breakdown: {reward_breakdown}")
        print(f"  Auxiliary targets: {targets}")
        print(f"  Shape [1, 3]: {shape_ok}")
        print(f"  Values in [0, 1]: {values_in_range}")
        
        return shape_ok and values_in_range
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("PRIORITY 4-5: TEMPORAL MODELING & MULTI-HEAD VALUE LEARNING")
    print("=" * 70 + "\n")
    
    results = {
        "temporal_encoder": test_temporal_encoder(),
        "monster_trend_predictor": test_monster_trend_predictor(),
        "temporal_sequence": test_temporal_sequence_creation(),
        "multi_head_value": test_multi_head_value_net(),
        "multi_task_loss": test_multi_task_loss(),
        "auxiliary_targets": test_auxiliary_targets_creation(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All Priority 4-5 tests passed! Temporal & Multi-head implementation ready.")
    else:
        print("\n✗ Some tests failed.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
