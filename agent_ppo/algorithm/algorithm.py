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
        """Training entry: PPO update on a batch of SampleData (multi-epoch, multi-minibatch).

        训练入口：对一批 SampleData 执行多轮 PPO 更新（多epoch、多minibatch）。
        """
        if not list_sample_data:
            return
            
        # 准备数据张量（完整batch）
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)
        
        # 标准化优势（importance for PPO stability）
        advantage_mean = advantage.mean()
        advantage_std = advantage.std()
        if advantage_std > 1e-8:
            advantage = (advantage - advantage_mean) / (advantage_std + 1e-8)
        
        total_batch_size = len(list_sample_data)
        epoch_losses = []
        
        # 多epoch更新（Standard PPO）
        for epoch in range(self.num_epochs):
            # 在epoch内创建minibatch
            indices = torch.randperm(total_batch_size)
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            
            for batch_start in range(0, total_batch_size, self.minibatch_size):
                batch_end = min(batch_start + self.minibatch_size, total_batch_size)
                batch_indices = indices[batch_start:batch_end]
                
                # 获取minibatch
                batch_obs = obs[batch_indices]
                batch_legal_action = legal_action[batch_indices]
                batch_act = act[batch_indices]
                batch_old_prob = old_prob[batch_indices]
                batch_advantage = advantage[batch_indices]
                batch_old_value = old_value[batch_indices]
                batch_reward_sum = reward_sum[batch_indices]
                
                # Forward pass
                self.model.set_train_mode()
                self.optimizer.zero_grad()
                
                logits, value_pred = self.model(batch_obs)
                
                # Compute loss
                total_loss, info = self._compute_loss(
                    logits=logits,
                    value_pred=value_pred,
                    legal_action=batch_legal_action,
                    old_action=batch_act,
                    old_prob=batch_old_prob,
                    advantage=batch_advantage,
                    old_value=batch_old_value,
                    reward_sum=batch_reward_sum,
                    reward=reward[batch_indices],
                )
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
                self.optimizer.step()
                
                epoch_loss_sum += total_loss.item()
                epoch_loss_count += 1
                self.train_step += 1
            
            # 记录epoch平均loss
            avg_epoch_loss = epoch_loss_sum / max(epoch_loss_count, 1)
            epoch_losses.append(avg_epoch_loss)
        
        # 多epoch完成后的监控上报
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            results = {
                "train_step": self.train_step,
                "num_samples": total_batch_size,
                "epochs_done": self.num_epochs,
                "minibatch_size": self.minibatch_size,
                "avg_epoch_loss": round(avg_loss, 4),
                "final_epoch_loss": round(epoch_losses[-1] if epoch_losses else 0.0, 4),
            }
            self.logger.info(
                f"[train] step:{results['train_step']} samples:{results['num_samples']} "
                f"epochs:{results['epochs_done']} avg_loss:{results['avg_epoch_loss']} "
                f"final_loss:{results['final_epoch_loss']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

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
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 损失（策略损失 + 价值损失 + 熵正则化）。
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
