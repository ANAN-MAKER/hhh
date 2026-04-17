#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for Gorge Chase PPO.
峡谷追猎 PPO 算法实现。

损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss

  - value_loss  : Clipped value function loss（裁剪价值函数损失）
  - policy_loss : PPO Clipped surrogate objective（PPO 裁剪替代目标）
  - entropy_loss: Action entropy regularization（动作熵正则化，鼓励探索）
"""

import os
import time

import torch
from agent_ppo.conf.conf import Config
from agent_ppo.feature.mask_utils import softmax_torch


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM
        
        # PPO multi-epoch and minibatch parameters
        self.num_epochs = getattr(Config, 'PPO_EPOCHS', 4)  # Standard PPO: 3-4 epochs
        self.minibatch_size = getattr(Config, 'PPO_MINIBATCH_SIZE', 64)  # Standard PPO: 32-128

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        """Training entry: Standard PPO update on a batch of SampleData.

        训练入口：对一批 SampleData 执行标准 PPO 多轮更新（多epoch、多minibatch、target KL early stop）。
        === 修改 Task E ===
        现在使用logprob而非完整概率向量，支持target KL early stop。
        """
        if not list_sample_data:
            return
        
        # 验证old_logprob字段存在
        if not hasattr(list_sample_data[0], 'old_logprob') or list_sample_data[0].old_logprob is None:
            if self.logger:
                self.logger.warning("old_logprob not found in SampleData, falling back to old behavior")
            return
            
        # 准备数据张量（完整batch）
        obs = torch.stack([torch.tensor(f.obs, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([torch.tensor(f.legal_action, dtype=torch.float32) for f in list_sample_data]).to(self.device)
        act = torch.stack([torch.tensor(f.act, dtype=torch.long) for f in list_sample_data]).to(self.device).view(-1)
        old_logprob = torch.stack([torch.tensor(f.old_logprob, dtype=torch.float32) for f in list_sample_data]).to(self.device).view(-1)
        advantage = torch.stack([torch.tensor(f.advantage, dtype=torch.float32) for f in list_sample_data]).to(self.device).view(-1)
        old_value = torch.stack([torch.tensor(f.old_value, dtype=torch.float32) for f in list_sample_data]).to(self.device).view(-1)
        return_ = torch.stack([torch.tensor(f.return_, dtype=torch.float32) for f in list_sample_data]).to(self.device).view(-1)
        
        # 标准化优势（PPO的关键稳定性技巧）
        if getattr(Config, 'NORMALIZE_ADVANTAGE', True):
            advantage_mean = advantage.mean()
            advantage_std = advantage.std()
            if advantage_std > 1e-8:
                advantage = (advantage - advantage_mean) / (advantage_std + 1e-8)
        
        total_batch_size = len(list_sample_data)
        epoch_losses = []
        approx_kls = []
        
        # 多epoch更新（Standard PPO）
        target_kl = getattr(Config, 'TARGET_KL', 0.02)
        early_stop = False
        
        for epoch in range(self.num_epochs):
            if early_stop:
                break
            
            # 在epoch内创建minibatch
            indices = torch.randperm(total_batch_size, device=self.device)
            epoch_loss_sum = 0.0
            epoch_kl_sum = 0.0
            epoch_loss_count = 0
            
            for batch_start in range(0, total_batch_size, self.minibatch_size):
                batch_end = min(batch_start + self.minibatch_size, total_batch_size)
                batch_indices = indices[batch_start:batch_end]
                
                # 获取minibatch
                batch_obs = obs[batch_indices]
                batch_legal_action = legal_action[batch_indices]
                batch_act = act[batch_indices]
                batch_old_logprob = old_logprob[batch_indices]
                batch_advantage = advantage[batch_indices]
                batch_old_value = old_value[batch_indices]
                batch_return = return_[batch_indices]
                
                # Forward pass
                self.model.set_train_mode()
                self.optimizer.zero_grad()
                
                logits, value_pred = self.model(batch_obs)
                
                # 计算新的logprob
                new_logprob, policy_loss, entropy_loss, approx_kl, clip_frac = self._compute_policy_loss(
                    logits=logits,
                    legal_action=batch_legal_action,
                    action=batch_act,
                    old_logprob=batch_old_logprob,
                    advantage=batch_advantage,
                )
                
                # 计算价值损失
                value_loss = self._compute_value_loss(
                    value_pred=value_pred,
                    old_value=batch_old_value,
                    return_=batch_return,
                )
                
                # 总损失
                total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss
                
                # Backward pass
                total_loss.backward()
                max_grad_norm = getattr(Config, 'MAX_GRAD_NORM', 0.5)
                torch.nn.utils.clip_grad_norm_(self.parameters, max_grad_norm)
                self.optimizer.step()
                
                epoch_loss_sum += total_loss.item()
                epoch_kl_sum += approx_kl.item()
                epoch_loss_count += 1
                self.train_step += 1
                
                # Early stop if KL divergence exceeds target
                if approx_kl > target_kl:
                    early_stop = True
                    break
            
            # 记录epoch平均loss和KL
            if epoch_loss_count > 0:
                avg_epoch_loss = epoch_loss_sum / epoch_loss_count
                avg_epoch_kl = epoch_kl_sum / epoch_loss_count
                epoch_losses.append(avg_epoch_loss)
                approx_kls.append(avg_epoch_kl)
        
        # 多epoch完成后的监控上报
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            avg_kl = sum(approx_kls) / max(len(approx_kls), 1)
            results = {
                "train_step": self.train_step,
                "num_samples": total_batch_size,
                "epochs_done": len(epoch_losses),
                "minibatch_size": self.minibatch_size,
                "avg_epoch_loss": round(avg_loss, 4),
                "approx_kl": round(avg_kl, 4),
                "early_stop": early_stop,
            }
            if self.logger:
                self.logger.info(
                    f"[train] step:{results['train_step']} samples:{results['num_samples']} "
                    f"epochs:{results['epochs_done']} avg_loss:{results['avg_epoch_loss']} "
                    f"approx_kl:{results['approx_kl']} early_stop:{results['early_stop']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_policy_loss(self, logits, legal_action, action, old_logprob, advantage):
        """Compute PPO policy loss with logprob.
        
        使用logprob计算PPO策略损失。
        === 修改 Task E ===
        改用logprob而非概率向量，确保与采样阶段一致的分布定义。
        """
        import torch.distributions as dist_module
        
        # 构造masked分布
        masked_logits = logits + (legal_action - 1.0) * 1e10
        dist = dist_module.Categorical(logits=masked_logits)
        
        # 计算新的logprob
        new_logprob = dist.log_prob(action)  # shape: (batch_size,)
        
        # Policy loss (PPO Clipped surrogate)
        # ratio = exp(new_logprob - old_logprob)
        ratio = torch.exp(new_logprob - old_logprob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy loss
        entropy_loss = dist.entropy().mean()
        
        # 诊断指标
        approx_kl = (old_logprob - new_logprob).mean()
        clip_frac = (torch.abs(ratio - 1.0) > self.clip_param).float().mean()
        
        return new_logprob, policy_loss, entropy_loss, approx_kl, clip_frac
    
    def _compute_value_loss(self, value_pred, old_value, return_):
        """Compute clipped value loss.
        
        计算裁剪价值损失。
        """
        value_clip_range = getattr(Config, 'VALUE_CLIP_RANGE', 0.2)
        
        # Clipped value loss
        value_pred_clipped = old_value + (value_pred - old_value).clamp(-value_clip_range, value_clip_range)
        value_loss_1 = (value_pred - return_).pow(2)
        value_loss_2 = (value_pred_clipped - return_).pow(2)
        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
        
        return value_loss

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """Deprecated: Use _compute_policy_loss and _compute_value_loss instead.
        
        已弃用：改用_compute_policy_loss和_compute_value_loss。
        """
        # Masked softmax / 合法动作掩码 softmax
        prob_dist = self._masked_softmax(logits, legal_action)

        # Policy loss (PPO Clip) / 策略损失
        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True).clamp(1e-9)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # Value loss (Clipped) / 价值损失
        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(tdret - vp),
                torch.square(tdret - value_clip),
            ).mean()
        )

        # Entropy loss / 熵损失
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        # PPO stability diagnostics / PPO 稳定性指标
        approx_kl = (torch.log(old_action_prob) - torch.log(new_prob)).mean()
        clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean()

        # Total loss / 总损失
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        info = {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }
        return total_loss, info

    def _masked_softmax(self, logits, legal_action):
        """Softmax with legal action masking (suppress illegal actions).

        合法动作掩码下的 softmax（将非法动作概率压为极小值）。
        统一使用mask_utils.softmax_torch确保与numpy版本一致。
        """
        return softmax_torch(logits, legal_action)
