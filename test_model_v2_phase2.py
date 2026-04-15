#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Phase 2: 功能完善 - Unit Tests for Model V2
单元测试框架 + 性能测试

这些测试将在Phase 2中逐步完善
"""

import torch
import torch.nn as nn
import time
from agent_ppo.model.model_v2 import (
    ModelV2,
    HeroStateEncoder,
    LocalMapEncoder,
    MonsterThreatEncoder,
    ResourceTargetEncoder,
    TemporalMemoryEncoder,
    LegalActionEncoder,
    ConsolidatedThreatFusion,
    OpportunitiesFusion,
    GlobalSituationFusion,
    ActionPlanningHead,
    PolicyHead,
    ValueHead,
)
from agent_ppo.conf.conf import Config


class TestEncoders:
    """Test individual encoders from Stage 1."""
    
    @staticmethod
    def test_hero_state_encoder():
        """Test HeroStateEncoder."""
        print("Testing HeroStateEncoder...")
        encoder = HeroStateEncoder(input_dim=24, output_dim=64)
        x = torch.randn(8, 24)
        y = encoder(x)
        assert y.shape == (8, 64), f"Expected shape (8, 64), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        print("  ✓ HeroStateEncoder passed")
    
    @staticmethod
    def test_local_map_encoder():
        """Test LocalMapEncoder."""
        print("Testing LocalMapEncoder...")
        encoder = LocalMapEncoder(input_dim=35, output_dim=64)
        x = torch.randn(8, 35)
        y = encoder(x)
        assert y.shape == (8, 64), f"Expected shape (8, 64), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        print("  ✓ LocalMapEncoder passed")
    
    @staticmethod
    def test_monster_threat_encoder():
        """Test MonsterThreatEncoder."""
        print("Testing MonsterThreatEncoder...")
        encoder = MonsterThreatEncoder(monster1_dim=7, monster2_dim=7, output_dim=64)
        m1 = torch.randn(8, 7)
        m2 = torch.randn(8, 7)
        y = encoder(m1, m2)
        assert y.shape == (8, 64), f"Expected shape (8, 64), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        print("  ✓ MonsterThreatEncoder passed")
    
    @staticmethod
    def test_resource_target_encoder():
        """Test ResourceTargetEncoder."""
        print("Testing ResourceTargetEncoder...")
        encoder = ResourceTargetEncoder(input_dim=20, output_dim=64)
        x = torch.randn(8, 20)  # 12D treasure + 8D buff
        y = encoder(x)
        assert y.shape == (8, 64), f"Expected shape (8, 64), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        print("  ✓ ResourceTargetEncoder passed")
    
    @staticmethod
    def test_temporal_memory_encoder():
        """Test TemporalMemoryEncoder."""
        print("Testing TemporalMemoryEncoder...")
        encoder = TemporalMemoryEncoder(input_dim=30, output_dim=96)
        x = torch.randn(8, 30)
        y = encoder(x)
        assert y.shape == (8, 96), f"Expected shape (8, 96), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        print("  ✓ TemporalMemoryEncoder passed")
    
    @staticmethod
    def test_legal_action_encoder():
        """Test LegalActionEncoder."""
        print("Testing LegalActionEncoder...")
        encoder = LegalActionEncoder(input_dim=16, output_dim=16)
        x = torch.randn(8, 16)
        y = encoder(x)
        assert y.shape == (8, 16), f"Expected shape (8, 16), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        print("  ✓ LegalActionEncoder passed")
    
    @staticmethod
    def run_all():
        print("\n" + "="*60)
        print("STAGE 1: ENCODER TESTS")
        print("="*60)
        TestEncoders.test_hero_state_encoder()
        TestEncoders.test_local_map_encoder()
        TestEncoders.test_monster_threat_encoder()
        TestEncoders.test_resource_target_encoder()
        TestEncoders.test_temporal_memory_encoder()
        TestEncoders.test_legal_action_encoder()
        print("✓ All encoder tests passed!\n")


class TestFusionBlocks:
    """Test fusion blocks from Stage 2."""
    
    @staticmethod
    def test_threat_fusion():
        """Test ConsolidatedThreatFusion."""
        print("Testing ConsolidatedThreatFusion...")
        fusion = ConsolidatedThreatFusion(input_dim=224, output_dim=128)
        threat = torch.randn(8, 64)
        map_feat = torch.randn(8, 64)
        temporal = torch.randn(8, 96)
        y = fusion(threat, map_feat, temporal)
        assert y.shape == (8, 128), f"Expected shape (8, 128), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        print("  ✓ ConsolidatedThreatFusion passed")
    
    @staticmethod
    def test_opportunity_fusion():
        """Test OpportunitiesFusion."""
        print("Testing OpportunitiesFusion...")
        fusion = OpportunitiesFusion(input_dim=256, output_dim=128)
        resource = torch.randn(8, 64)
        hero = torch.randn(8, 64)
        threat_fusion = torch.randn(8, 128)
        y = fusion(resource, hero, threat_fusion)
        assert y.shape == (8, 128), f"Expected shape (8, 128), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        print("  ✓ OpportunitiesFusion passed")
    
    @staticmethod
    def test_global_fusion():
        """Test GlobalSituationFusion."""
        print("Testing GlobalSituationFusion...")
        fusion = GlobalSituationFusion(input_dim=272, output_dim=192)
        threat_fusion = torch.randn(8, 128)
        opp_fusion = torch.randn(8, 128)
        legal = torch.randn(8, 16)
        y = fusion(threat_fusion, opp_fusion, legal)
        assert y.shape == (8, 192), f"Expected shape (8, 192), got {y.shape}"
        assert not torch.isnan(y).any(), "Output contains NaN"
        print("  ✓ GlobalSituationFusion passed")
    
    @staticmethod
    def run_all():
        print("\n" + "="*60)
        print("STAGE 2: FUSION BLOCK TESTS")
        print("="*60)
        TestFusionBlocks.test_threat_fusion()
        TestFusionBlocks.test_opportunity_fusion()
        TestFusionBlocks.test_global_fusion()
        print("✓ All fusion block tests passed!\n")


class TestOutputHeads:
    """Test output heads from Stage 4."""
    
    @staticmethod
    def test_policy_head():
        """Test PolicyHead dual-layer output."""
        print("Testing PolicyHead...")
        head = PolicyHead(input_dim=128)
        x = torch.randn(8, 128)
        y = head(x)
        
        # Check shape
        assert y.shape == (8, 16), f"Expected shape (8, 16), got {y.shape}"
        
        # Check probabilities sum to 1
        probs_sum = y.sum(dim=1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5), \
            f"Probabilities don't sum to 1: {probs_sum}"
        
        # Check all probabilities are valid
        assert (y >= 0).all() and (y <= 1).all(), "Probabilities out of [0, 1] range"
        
        print("  ✓ PolicyHead passed")
    
    @staticmethod
    def test_value_head():
        """Test ValueHead multi-dimensional output."""
        print("Testing ValueHead...")
        head = ValueHead(input_dim=192)
        x = torch.randn(8, 192)
        y = head(x)
        
        # Check shape
        assert y.shape == (8, 1), f"Expected shape (8, 1), got {y.shape}"
        
        # Check no NaN
        assert not torch.isnan(y).any(), "Output contains NaN"
        
        print("  ✓ ValueHead passed")
    
    @staticmethod
    def run_all():
        print("\n" + "="*60)
        print("STAGE 4: OUTPUT HEAD TESTS")
        print("="*60)
        TestOutputHeads.test_policy_head()
        TestOutputHeads.test_value_head()
        print("✓ All output head tests passed!\n")


class PerformanceBenchmark:
    """Performance benchmark tests."""
    
    @staticmethod
    def benchmark_forward_pass(batch_size=32, num_iterations=100):
        """Benchmark forward pass time."""
        print(f"\n" + "="*60)
        print(f"PERFORMANCE BENCHMARK")
        print("="*60)
        print(f"Batch size: {batch_size}")
        print(f"Iterations: {num_iterations}")
        
        model = ModelV2(device='cpu')
        model.eval()
        
        # Warmup
        obs = torch.randn(batch_size, Config.DIM_OF_OBSERVATION)
        with torch.no_grad():
            for _ in range(10):
                _ = model(obs)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                obs = torch.randn(batch_size, Config.DIM_OF_OBSERVATION)
                start = time.time()
                _ = model(obs)
                elapsed = time.time() - start
                times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\nForward pass time (CPU):")
        print(f"  Average: {avg_time*1000:.2f} ms")
        print(f"  Min:     {min_time*1000:.2f} ms")
        print(f"  Max:     {max_time*1000:.2f} ms")
        print(f"  Throughput: {batch_size/avg_time:.0f} samples/sec")
    
    @staticmethod
    def analyze_parameter_distribution():
        """Analyze parameter distribution across modules."""
        print(f"\n" + "="*60)
        print(f"PARAMETER DISTRIBUTION")
        print("="*60)
        
        model = ModelV2(device='cpu')
        
        # Count parameters by module
        modules_params = {}
        for name, module in model.named_modules():
            if hasattr(module, 'parameters'):
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    modules_params[name] = param_count
        
        # Sort by parameter count
        sorted_modules = sorted(modules_params.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop modules by parameter count:")
        for name, count in sorted_modules[:15]:
            print(f"  {name:50s}: {count:10,} params ({count/513968*100:5.1f}%)")
    
    @staticmethod
    def run_all():
        PerformanceBenchmark.benchmark_forward_pass(batch_size=32, num_iterations=100)
        PerformanceBenchmark.analyze_parameter_distribution()


def main():
    print("\n" + "="*60)
    print("MODEL V2 - PHASE 2 TEST SUITE")
    print("="*60)
    
    try:
        # Run all unit tests
        TestEncoders.run_all()
        TestFusionBlocks.run_all()
        TestOutputHeads.run_all()
        
        # Run benchmarks
        PerformanceBenchmark.run_all()
        
        # Summary
        print("\n" + "="*60)
        print("PHASE 2 TEST SUMMARY")
        print("="*60)
        print("✓ All encoder tests passed")
        print("✓ All fusion block tests passed")
        print("✓ All output head tests passed")
        print("✓ Performance benchmarks completed")
        print("\n✓✓✓ PHASE 2 READY ✓✓✓\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
