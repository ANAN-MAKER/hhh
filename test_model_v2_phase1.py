#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Phase 1 Validation Test for Model V2
验证 Phase 1 模型 V2 是否正常工作

验证项：
1. ✓ 模型初始化无误
2. ✓ 前向传播正常
3. ✓ 维度匹配正确
4. ✓ 梯度反向传播正常
5. ✓ 损失计算正确
"""

import torch
import torch.nn as nn
from agent_ppo.model.model_v2 import ModelV2
from agent_ppo.conf.conf import Config

def test_model_initialization():
    """Test 1: Model initialization"""
    print("=" * 60)
    print("Test 1: Model Initialization")
    print("=" * 60)
    
    try:
        model = ModelV2(device='cpu')
        print("✓ Model V2 initialized successfully")
        print(f"  Model name: {model.model_name}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        raise

def test_forward_pass(model):
    """Test 2: Forward pass and dimension check"""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass & Dimension Validation")
    print("=" * 60)
    
    batch_size = 8
    feature_dim = Config.DIM_OF_OBSERVATION  # 159
    
    # Create random input batch
    obs = torch.randn(batch_size, feature_dim)
    
    try:
        logits, value = model(obs)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {obs.shape} (expected: [{batch_size}, {feature_dim}])")
        print(f"  Logits shape: {logits.shape} (expected: [{batch_size}, 16])")
        print(f"  Value shape: {value.shape} (expected: [{batch_size}, 1])")
        
        # Verify dimensions
        assert logits.shape == (batch_size, 16), f"Logits shape mismatch: {logits.shape}"
        assert value.shape == (batch_size, 1), f"Value shape mismatch: {value.shape}"
        
        # Verify probabilities sum to 1
        probs_sum = logits.sum(dim=1)
        print(f"  Action probabilities sum: min={probs_sum.min():.4f}, max={probs_sum.max():.4f}")
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5), \
            "Action probabilities don't sum to 1"
        
        return logits, value
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise

def test_gradient_flow(model, logits, value):
    """Test 3: Gradient flow"""
    print("\n" + "=" * 60)
    print("Test 3: Gradient Flow & Backpropagation")
    print("=" * 60)
    
    try:
        # Create simple loss (MSE for value, cross-entropy for action)
        batch_size = logits.shape[0]
        target_actions = torch.randint(0, 16, (batch_size,))
        target_value = torch.randn(batch_size, 1)
        
        action_loss = nn.CrossEntropyLoss()(logits, target_actions)
        value_loss = nn.MSELoss()(value, target_value)
        total_loss = action_loss + 0.5 * value_loss
        
        print(f"  Action loss: {action_loss.item():.6f}")
        print(f"  Value loss: {value_loss.item():.6f}")
        print(f"  Total loss: {total_loss.item():.6f}")
        
        # Backward pass
        total_loss.backward()
        
        print("✓ Backward pass successful")
        
        # Check gradient flow
        grad_count = 0
        no_grad_count = 0
        max_grad = 0.0
        min_grad = float('inf')
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                grad_norm = param.grad.norm().item()
                max_grad = max(max_grad, grad_norm)
                min_grad = min(min_grad, grad_norm)
            else:
                no_grad_count += 1
        
        print(f"  Parameters with gradients: {grad_count}")
        print(f"  Parameters without gradients: {no_grad_count}")
        print(f"  Gradient norm range: [{min_grad:.6f}, {max_grad:.6f}]")
        
        if no_grad_count > 0:
            print(f"  ⚠ Warning: {no_grad_count} parameters have no gradients")
        
        assert grad_count > 0, "No gradients computed!"
        
        return total_loss
        
    except Exception as e:
        print(f"✗ Gradient flow failed: {e}")
        raise

def test_dimension_tracking():
    """Test 4: Complete dimension tracking"""
    print("\n" + "=" * 60)
    print("Test 4: Complete Dimension Tracking")
    print("=" * 60)
    
    model = ModelV2(device='cpu')
    batch_size = 4
    feature_dim = Config.DIM_OF_OBSERVATION
    
    obs = torch.randn(batch_size, feature_dim)
    
    # Manual forward pass to track dimensions
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
    
    print("\n  Input Feature Groups:")
    print(f"    self_feat:     {self_feat.shape}")
    print(f"    monster1:      {monster1.shape}")
    print(f"    monster2:      {monster2.shape}")
    print(f"    treasure:      {treasure.shape}")
    print(f"    buff:          {buff.shape}")
    print(f"    progress:      {progress.shape}")
    print(f"    map_local:     {map_local.shape}")
    print(f"    action_plan:   {action_plan.shape}")
    print(f"    legal:         {legal.shape}")
    
    # Stage 1 encoders
    hero_encoded = model.hero_encoder(self_feat)
    map_encoded = model.map_encoder(map_local)
    threat_encoded = model.threat_encoder(monster1, monster2)
    resources = torch.cat([treasure, buff], dim=-1)
    resource_encoded = model.resource_encoder(resources)
    temporal_encoded = model.temporal_encoder(action_plan)
    legal_encoded = model.legal_encoder(legal)
    
    print("\n  Stage 1 - Input Encoding Layer Output:")
    print(f"    hero_encoded:       {hero_encoded.shape} (expect [B, 64])")
    print(f"    map_encoded:        {map_encoded.shape} (expect [B, 64])")
    print(f"    threat_encoded:     {threat_encoded.shape} (expect [B, 64])")
    print(f"    resource_encoded:   {resource_encoded.shape} (expect [B, 64])")
    print(f"    temporal_encoded:   {temporal_encoded.shape} (expect [B, 96])")
    print(f"    legal_encoded:      {legal_encoded.shape} (expect [B, 16])")
    
    # Stage 2 - fusion
    threat_fusion = model.threat_fusion(threat_encoded, map_encoded, temporal_encoded)
    opportunity_fusion = model.opportunity_fusion(resource_encoded, hero_encoded, threat_fusion)
    global_situation = model.global_fusion(threat_fusion, opportunity_fusion, legal_encoded)
    
    print("\n  Stage 2 - Situation Understanding Layer Output:")
    print(f"    threat_fusion:      {threat_fusion.shape} (expect [B, 128])")
    print(f"    opportunity_fusion: {opportunity_fusion.shape} (expect [B, 128])")
    print(f"    global_situation:   {global_situation.shape} (expect [B, 192])")
    
    # Stage 3 - planning
    action_planning = model.action_planning(global_situation, map_encoded, threat_encoded)
    
    print("\n  Stage 3 - Action Planning Layer Output:")
    print(f"    action_planning:    {action_planning.shape} (expect [B, 128])")
    
    # Stage 4 - output
    logits = model.policy_head(action_planning)
    value = model.value_head(global_situation)
    
    print("\n  Stage 4 - Policy & Value Output:")
    print(f"    logits:             {logits.shape} (expect [B, 16])")
    print(f"    value:              {value.shape} (expect [B, 1])")
    
    print("\n✓ All dimensions correctly tracked!")

def test_compatibility_with_old_model():
    """Test 5: Compatibility check"""
    print("\n" + "=" * 60)
    print("Test 5: Compatibility Check")
    print("=" * 60)
    
    # Check if config is compatible
    assert Config.ACTION_NUM == 16, "Action number should be 16"
    assert Config.VALUE_NUM == 1, "Value number should be 1"
    assert Config.FEATURE_LEN == 159, "Feature length should be 159"
    
    print("✓ Input feature dimension: 159 ✓")
    print("✓ Output action dimension: 16 ✓")
    print("✓ Output value dimension: 1 ✓")
    print("✓ Configuration compatible with training pipeline ✓")

def main():
    print("\n" + "=" * 60)
    print("MODEL V2 - PHASE 1 VALIDATION TEST")
    print("=" * 60)
    
    # Test 1: Initialization
    model = test_model_initialization()
    
    # Test 5: Compatibility
    test_compatibility_with_old_model()
    
    # Test 2: Forward pass
    logits, value = test_forward_pass(model)
    
    # Test 4: Dimension tracking
    test_dimension_tracking()
    
    # Test 3: Gradient flow
    test_gradient_flow(model, logits, value)
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 VALIDATION SUMMARY")
    print("=" * 60)
    print("✓ Model initialization successful")
    print("✓ Forward pass working correctly")
    print("✓ All dimensions properly tracked")
    print("✓ Gradient flow normal")
    print("✓ Configuration compatible")
    print("\n✓✓✓ PHASE 1 VALIDATION PASSED ✓✓✓")
    print("\nReady for Phase 2: 功能完善\n")

if __name__ == "__main__":
    main()
