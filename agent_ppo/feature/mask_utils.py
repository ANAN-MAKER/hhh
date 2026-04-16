#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Unified masked softmax utilities.

统一的掩码softmax工具，确保numpy和torch版本行为完全一致。
用于保证rollout、evaluate、train三个阶段的策略分布一致性。
"""

import numpy as np
import torch


# 数值稳定性常数
LARGE_NEGATIVE = -1e20  # 用于压制非法动作的大负数
SMALL_EPS = 1e-5  # 避免log(0)的小正数


def softmax_numpy(logits: np.ndarray, legal_action: np.ndarray) -> np.ndarray:
    """
    Numpy版本的掩码softmax。用于推理阶段（rollout/evaluate）。
    
    Args:
        logits: [batch_size, action_dim] 或 [action_dim]
        legal_action: [batch_size, action_dim] 或 [action_dim]，float32或bool
        
    Returns:
        prob: 同shape的概率分布，非法动作概率为0
        
    公式：
        1. masked_logits = logits - LARGE_NEGATIVE * (1.0 - legal_action)
        2. 减去max保证数值稳定
        3. softmax = exp(masked_logits - max) 
        4. 乘以legal_action mask（确保非法动作概率为0）
        5. 正规化使和为1
    """
    # 转换数据类型
    logits = np.asarray(logits, dtype=np.float32)
    legal_action = np.asarray(legal_action, dtype=np.float32)
    
    # 将非法动作的logits设置为很小的值
    # -inf * (1.0 - mask) 作用：当mask=0时(非法)，贡献LARGE_NEGATIVE；当mask=1时(合法)，贡献0
    masked_logits = logits + LARGE_NEGATIVE * (1.0 - legal_action)
    
    # 数值稳定性：减去max
    max_logits = np.max(masked_logits, axis=-1, keepdims=True)
    stable_logits = np.clip(masked_logits - max_logits, LARGE_NEGATIVE, 1)
    
    # 指数化和正规化
    exp_logits = np.exp(stable_logits)
    
    # 乘以mask确保非法动作概率为0（处理exp(LARGE_NEGATIVE)≈0的舍入误差）
    masked_exp = exp_logits * legal_action
    
    # 加上小epsilon避免全0分母
    prob = (masked_exp + SMALL_EPS) * legal_action
    prob_sum = np.sum(prob, axis=-1, keepdims=True)
    
    # 正规化（防止除以0）
    prob = prob / (prob_sum + SMALL_EPS)
    
    return prob


def softmax_torch(logits: torch.Tensor, legal_action: torch.Tensor) -> torch.Tensor:
    """
    Torch版本的掩码softmax。用于训练阶段（learn）。
    
    Args:
        logits: [batch_size, action_dim] 张量
        legal_action: [batch_size, action_dim] 张量，float32或bool
        
    Returns:
        prob: 同shape的概率分布，非法动作概率为0
        
    注意：此版本与softmax_numpy行为完全一致，确保分布一致性。
    """
    # 转换数据类型
    logits = logits.float()
    legal_action = legal_action.float()
    
    # 复制numpy逻辑到torch
    # 将非法动作的logits设置为很小的值
    masked_logits = logits + LARGE_NEGATIVE * (1.0 - legal_action)
    
    # 数值稳定性：减去max
    max_logits, _ = torch.max(masked_logits, dim=-1, keepdim=True)
    stable_logits = torch.clamp(masked_logits - max_logits, LARGE_NEGATIVE, 1)
    
    # 指数化
    exp_logits = torch.exp(stable_logits)
    
    # 乘以mask
    masked_exp = exp_logits * legal_action
    
    # 加上小epsilon
    prob = (masked_exp + SMALL_EPS) * legal_action
    prob_sum = torch.sum(prob, dim=-1, keepdim=True)
    
    # 正规化
    prob = prob / (prob_sum + SMALL_EPS)
    
    return prob


def verify_consistency(logits_np: np.ndarray, legal_np: np.ndarray, atol: float = 1e-5) -> bool:
    """
    快速验证numpy和torch版本的一致性。
    
    Args:
        logits_np: numpy数组 [action_dim]
        legal_np: numpy数组 [action_dim]，float32
        atol: 容差
        
    Returns:
        True if numpy和torch版本在容差内一致
    """
    # numpy版本
    prob_np = softmax_numpy(logits_np, legal_np)
    
    # torch版本
    logits_torch = torch.from_numpy(logits_np).unsqueeze(0).float()
    legal_torch = torch.from_numpy(legal_np).unsqueeze(0).float()
    prob_torch = softmax_torch(logits_torch, legal_torch)[0].cpu().numpy()
    
    # 比较
    is_close = np.allclose(prob_np, prob_torch, atol=atol, rtol=1e-5)
    
    if not is_close:
        max_diff = np.max(np.abs(prob_np - prob_torch))
        print(f"WARNING: numpy/torch inconsistency detected. Max diff: {max_diff}")
        print(f"  numpy prob: {prob_np}")
        print(f"  torch prob: {prob_torch}")
    
    return is_close


# 可选：为了向后兼容，保留旧名称的别名
legacy_softmax_numpy = softmax_numpy
legacy_softmax_torch = softmax_torch
