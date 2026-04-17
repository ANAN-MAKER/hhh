#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
峡谷追猎 PPO Agent 主类。
"""

import os

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.feature.mask_utils import softmax_numpy
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.

        将原始观测转换为 ObsData 和 remain_info。
        """
        feature, legal_action, reward, reward_info = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {
            "reward": reward,
            "reward_info": reward_info,
        }
        return obs_data, remain_info

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（探索）。
        === 修改 Task B ===
        现在使用统一的masked distribution，保存logprob供训练使用。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value = self._run_model(feature, legal_action)
        
        # 构造masked分布（统一实现）
        dist, prob = self._build_masked_dist(logits, legal_action)
        
        # 采样动作并计算logprob
        action = dist.sample().item()
        logprob = dist.log_prob(torch.tensor([action], device=self.device)).item()
        
        # 贪心动作（用于多步rollout或统计）
        d_action = int(torch.argmax(dist.logits).item())

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                logprob=logprob,
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        === 修改 Task B ===
        现在使用同一套_build_masked_dist（argmax模式）。
        """
        obs_data, _ = self.observation_process(env_obs)
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        logits, value = self._run_model(feature, legal_action)
        
        # 构造masked分布（统一实现）
        dist, prob = self._build_masked_dist(logits, legal_action)
        
        # 贪心选择
        action = torch.argmax(dist.logits).item()

        return action

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        if path is None:
            return

        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if not os.path.exists(model_file_path):
            if self.logger:
                self.logger.warning(f"model file {model_file_path} not found, use current parameters")
            return

        try:
            state_dict = torch.load(model_file_path, map_location=self.device)
            current_state = self.model.state_dict()
            compatible_state = {
                key: value
                for key, value in state_dict.items()
                if key in current_state and current_state[key].shape == value.shape
            }
            current_state.update(compatible_state)
            self.model.load_state_dict(current_state)

            skipped_keys = sorted(set(state_dict.keys()) - set(compatible_state.keys()))
            if self.logger:
                if skipped_keys:
                    self.logger.warning(
                        f"load model {model_file_path} partially, skipped {len(skipped_keys)} incompatible params"
                    )
                else:
                    self.logger.info(f"load model {model_file_path} successfully")
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"load model {model_file_path} failed, keep current parameters: {exc}")

    def action_process(self, act_data, is_stochastic=True):
        """Unpack ActData to int action and update last_action.

        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        """Run model inference, return logits, value.

        执行模型推理，返回 logits 和 value。
        === 修改 Task B ===
        不再在此计算概率，改由_build_masked_dist负责。
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        return logits_np, value_np

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking (numpy).

        合法动作掩码下的 softmax（numpy 版）。
        === 修改 Task B ===
        现已弃用。改用_build_masked_dist基于PyTorch实现统一分布。
        保留此方法仅供旧代码兼容。
        """
        return softmax_numpy(input_hidden, legal_action)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution.

        按概率分布采样动作。
        === 修改 Task B ===
        现已弃用。改用PyTorch Categorical分布采样。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
    
    def _build_masked_dist(self, logits_np, legal_action):
        """Build masked action distribution using PyTorch.
        
        使用PyTorch构造masked动作分布。
        === 修改 Task B ===
        统一的动作分布实现，同时供采样和训练使用。
        
        Args:
            logits_np: numpy array of shape (16,)
            legal_action: list of 0/1 indicating legal actions
            
        Returns:
            (dist, prob_np): PyTorch distribution and numpy probability
        """
        import torch.distributions as dist_module
        
        logits_tensor = torch.tensor(logits_np, dtype=torch.float32).to(self.device)
        legal_action_tensor = torch.tensor(legal_action, dtype=torch.float32).to(self.device)
        
        # Mask illegal actions with large negative values
        masked_logits = logits_tensor + (legal_action_tensor - 1.0) * 1e10
        
        # Create Categorical distribution
        dist = dist_module.Categorical(logits=masked_logits)
        
        # Get probabilities
        probs = torch.nn.functional.softmax(masked_logits, dim=0)
        prob_np = probs.cpu().detach().numpy()
        
        return dist, prob_np
