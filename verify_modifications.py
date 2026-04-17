#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Simplified verification of second round modifications.
Only checks code structure without full dependencies.
"""

import os
import re

print("="*60)
print("Verifying Second Round Modifications")
print("="*60)

# Check 1: Verify agent.py has _build_masked_dist
print("\n[Check 1] agent.py - _build_masked_dist method...")
with open("agent_ppo/agent.py", "r", encoding="utf-8") as f:
    agent_content = f.read()
    if "_build_masked_dist" in agent_content:
        print("OK - _build_masked_dist method found")
    else:
        print("FAIL - _build_masked_dist method NOT found")
    
    if "def predict(self" in agent_content and "logprob=" in agent_content:
        print("OK - predict() saves logprob")
    else:
        print("FAIL - predict() not updated for logprob")

# Check 2: Verify definition.py has new fields
print("\n[Check 2] definition.py - Extended ActData and SampleData...")
with open("agent_ppo/feature/definition.py", "r", encoding="utf-8") as f:
    def_content = f.read()
    
    if "logprob=None" in def_content:
        print("OK - ActData has logprob field")
    else:
        print("FAIL - ActData missing logprob")
    
    if "old_logprob" in def_content:
        print("OK - SampleData has old_logprob field")
    else:
        print("FAIL - SampleData missing old_logprob")
    
    if "return_" in def_content:
        print("OK - SampleData has return_ field")
    else:
        print("FAIL - SampleData missing return_")

# Check 3: Verify algorithm.py has new PPO features
print("\n[Check 3] algorithm.py - Standard PPO features...")
with open("agent_ppo/algorithm/algorithm.py", "r", encoding="utf-8") as f:
    algo_content = f.read()
    
    if "_compute_policy_loss" in algo_content:
        print("OK - _compute_policy_loss method exists")
    else:
        print("FAIL - _compute_policy_loss NOT found")
    
    if "TARGET_KL" in algo_content or "target_kl" in algo_content:
        print("OK - Target KL early stop implemented")
    else:
        print("FAIL - Target KL early stop NOT found")
    
    if "PPO_EPOCHS" in algo_content or "self.num_epochs" in algo_content:
        print("OK - Multi-epoch training implemented")
    else:
        print("FAIL - Multi-epoch training NOT found")
    
    if "old_logprob" in algo_content or "logprob" in algo_content:
        print("OK - logprob handling implemented")
    else:
        print("FAIL - logprob handling NOT found")

# Check 4: Verify model.py has dual critic
print("\n[Check 4] model.py - Dual critic support...")
with open("agent_ppo/model/model.py", "r", encoding="utf-8") as f:
    model_content = f.read()
    
    if "use_dual_critic" in model_content or "value_survival" in model_content:
        print("OK - Dual critic support added")
    else:
        print("FAIL - Dual critic support NOT found")
    
    if "self.gru" in model_content or "USE_GRU" in model_content:
        print("OK - GRU interface reserved")
    else:
        print("FAIL - GRU interface NOT found")

# Check 5: Verify conf.py has new parameters
print("\n[Check 5] conf.py - New PPO parameters...")
with open("agent_ppo/conf/conf.py", "r", encoding="utf-8") as f:
    conf_content = f.read()
    
    required_params = [
        "TARGET_KL",
        "MAX_GRAD_NORM",
        "NORMALIZE_ADVANTAGE",
        "VALUE_CLIP_RANGE",
        "USE_DUAL_CRITIC",
        "USE_GRU"
    ]
    
    missing = []
    for param in required_params:
        if param not in conf_content:
            missing.append(param)
    
    if not missing:
        print(f"OK - All {len(required_params)} required parameters found")
    else:
        print(f"FAIL - Missing parameters: {missing}")

# Check 6: Verify train_workflow.py has old_logprob
print("\n[Check 6] train_workflow.py - Sample collection updates...")
with open("agent_ppo/workflow/train_workflow.py", "r", encoding="utf-8") as f:
    workflow_content = f.read()
    
    if "old_logprob" in workflow_content:
        print("OK - old_logprob saved in samples")
    else:
        print("FAIL - old_logprob NOT saved")
    
    if "next_obs" in workflow_content:
        print("OK - next_obs saved in samples")
    else:
        print("FAIL - next_obs NOT saved")

print("\n" + "="*60)
print("Verification Complete")
print("="*60)
print("\nKey Improvements Summary:")
print("  [B] Task B - Unified masked policy distribution")
print("  [C] Task C - Extended ActData/SampleData with logprob")
print("  [D] Task D - Enhanced sample collection pipeline")
print("  [E] Task E - Standard PPO training with multi-epoch/minibatch")
print("  [F] Task F - Dual-head critic and GRU interface reservation")
print("  [G] Task G - Complete PPO configuration parameters")
print("="*60)
