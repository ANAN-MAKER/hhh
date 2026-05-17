#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import numpy as np
import os
import time
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None) -> None:
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_step: int = 0

        self.logger = logger
        self.monitor = monitor

        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.ppo_epochs = Config.PPO_EPOCHS
        self.minibatch_size = Config.PPO_MINIBATCH_SIZE

        self.last_report_monitor_time: float = 0

    def learn(self, list_sample_data) -> Dict[str, Any]:
        """
        list_sample_data: list[SampleData]
        SampleData对象列表
        """
        _input_datas = torch.stack([sample.sample for sample in list_sample_data]).to(self.device)
        results = {}
        batch_size = _input_datas.shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)
        self._normalize_advantages_in_place(_input_datas)
        last_total_loss = None
        last_info_list = None

        self.model.set_train_mode()
        self._set_annealed_hyperparams()

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                batch_indices = permutation[start : start + minibatch_size]
                batch_tensor = _input_datas[batch_indices]
                data_list = self._split_batch_tensor(batch_tensor)
                format_inputs = self._format_model_inputs(data_list)

                self.optimizer.zero_grad()
                rst_list = self.model(format_inputs)
                total_loss, info_list = self.model.compute_loss(data_list, rst_list)
                total_loss.backward()
                self._clip_gradients()
                self.optimizer.step()

                last_total_loss = total_loss
                last_info_list = info_list

        if last_total_loss is None or last_info_list is None:
            raise RuntimeError("No optimizer update was performed in Algorithm.learn")

        self.train_step += 1
        self.scheduler.step()

        results["total_loss"] = last_total_loss.item()
        _info_list = []
        for info in last_info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now
        return results

    def _split_batch_tensor(self, batch_tensor: torch.Tensor) -> List[torch.Tensor]:
        data_list = list(batch_tensor.split(self.cut_points, dim=1))
        for index, data in enumerate(data_list):
            data_list[index] = data.reshape(-1).float()
        return data_list

    def _format_model_inputs(self, data_list: List[torch.Tensor]) -> List[torch.Tensor]:
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, _ = seri_vec.split(
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)
        return [feature_vec, lstm_hidden_state, lstm_cell_state]

    def _normalize_advantages_in_place(self, batch_tensor: torch.Tensor) -> None:
        advantage_offset = self.cut_points[0] + self.cut_points[1]
        advantage_width = self.cut_points[2]
        advantages = batch_tensor[:, advantage_offset : advantage_offset + advantage_width]
        mean = torch.mean(advantages)
        std = torch.std(advantages, unbiased=False)
        advantages.sub_(mean).div_(torch.clamp(std, min=Config.ADVANTAGE_NORM_EPS))

    def _set_annealed_hyperparams(self) -> None:
        anneal_progress = min(1.0, self.train_step / Config.BETA_ANNEAL_STEPS)
        self.model.var_beta = Config.BETA_START + (Config.BETA_END - Config.BETA_START) * anneal_progress
        self.model.clip_param = (
            Config.CLIP_PARAM_START
            + (Config.CLIP_PARAM_END - Config.CLIP_PARAM_START) * anneal_progress
        )

    def _clip_gradients(self) -> None:
        if not Config.USE_GRAD_CLIP:
            return
        encoder_params = []
        lstm_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            elif "lstm" in name:
                lstm_params.append(param)
            else:
                head_params.append(param)
        if encoder_params:
            torch.nn.utils.clip_grad_norm_(encoder_params, Config.GRAD_CLIP_ENCODER)
        if lstm_params:
            torch.nn.utils.clip_grad_norm_(lstm_params, Config.GRAD_CLIP_LSTM)
        if head_params:
            torch.nn.utils.clip_grad_norm_(head_params, Config.GRAD_CLIP_HEAD)
