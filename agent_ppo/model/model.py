#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config, DimConfig


def masked_softmax_torch(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        legal_mask = legal_mask.unsqueeze(0)
        squeeze_result = True
    else:
        squeeze_result = False

    legal_mask = legal_mask.float()
    valid_rows = torch.sum(legal_mask, dim=1, keepdim=True) > 0
    large_negative = torch.tensor(1e20, device=logits.device, dtype=logits.dtype)
    tmp = logits - large_negative * (1.0 - legal_mask)
    tmp_max = torch.max(tmp, dim=1, keepdim=True).values
    tmp = torch.clamp(tmp - tmp_max, -large_negative, 1)
    exp_logits = (torch.exp(tmp) + epsilon) * legal_mask
    denom = torch.clamp(torch.sum(exp_logits, dim=1, keepdim=True), min=1e-8)
    probs = exp_logits / denom
    if torch.any(~valid_rows):
        uniform = torch.full_like(probs, 1.0 / probs.shape[1])
        probs = torch.where(valid_rows, probs, uniform)
    if squeeze_result:
        return probs.squeeze(0)
    return probs


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.var_beta = Config.BETA_START
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])
        self.self_hero_dim = DimConfig.SELF_HERO_DIM
        self.enemy_hero_dim = DimConfig.ENEMY_HERO_DIM
        self.self_tower_dim = DimConfig.SELF_TOWER_DIM
        self.enemy_tower_dim = DimConfig.ENEMY_TOWER_DIM
        self.soldier_slot_dim = DimConfig.SOLDIER_SLOT_DIM
        self.soldier_slot_count = DimConfig.SOLDIER_SLOT_COUNT
        self.global_context_dim = DimConfig.GLOBAL_CONTEXT_DIM
        self.matchup_dim = DimConfig.MATCHUP_DIM

        self.self_hero_encoder = MLP([self.self_hero_dim, 64, 64], "self_hero_encoder", non_linearity_last=True)
        self.enemy_hero_encoder = MLP([self.enemy_hero_dim, 64, 64], "enemy_hero_encoder", non_linearity_last=True)
        self.self_tower_encoder = MLP([self.self_tower_dim, 32, 32], "self_tower_encoder", non_linearity_last=True)
        self.enemy_tower_encoder = MLP([self.enemy_tower_dim, 32, 32], "enemy_tower_encoder", non_linearity_last=True)
        self.soldier_encoder = MLP([self.soldier_slot_dim, 32, 32], "soldier_encoder", non_linearity_last=True)
        self.global_context_encoder = MLP(
            [self.global_context_dim, 32, 32],
            "global_context_encoder",
            non_linearity_last=True,
        )
        self.matchup_encoder = MLP([self.matchup_dim, 16], "matchup_encoder", non_linearity_last=True)

        fusion_input_dim = 64 + 64 + 32 + 32 + (self.soldier_slot_count * 2 * 32) + 32 + 16
        self.fusion_mlp = MLP([fusion_input_dim, 256], "fusion_mlp", non_linearity_last=True)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.label_heads = nn.ModuleList(
            [
                MLP([self.lstm_unit_size, 256, self.label_size_list[lidx]], "label_head_{}".format(lidx))
                for lidx in range(len(self.label_size_list))
            ]
        )
        self.value_head = MLP([self.lstm_unit_size, 256, 1], "value_head")

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        batch_size = lstm_hidden_init.shape[0]
        seq_len = 1 if inference else self.lstm_time_steps

        grouped = self._split_feature_groups(feature_vec)
        encoded_feature = self._encode_entities(grouped)

        if encoded_feature.shape[0] != batch_size * seq_len:
            raise RuntimeError(
                "LSTM batch mismatch: encoded_feature={0}, batch_size={1}, seq_len={2}".format(
                    encoded_feature.shape[0],
                    batch_size,
                    seq_len,
                )
            )

        lstm_input = encoded_feature.reshape(batch_size, seq_len, -1)
        lstm_output, (lstm_hidden_output, lstm_cell_output) = self.lstm(
            lstm_input,
            (lstm_hidden_init.unsqueeze(0), lstm_cell_init.unsqueeze(0)),
        )
        flat_lstm_output = lstm_output.reshape(batch_size * seq_len, self.lstm_unit_size)

        result_list = []
        for lidx in range(len(self.label_size_list)):
            result_list.append(self.label_heads[lidx](flat_lstm_output))
        value_result = self.value_head(flat_lstm_output)
        result_list.append(value_result)

        if inference:
            logits = torch.flatten(torch.cat(result_list[:-1], dim=1), start_dim=1)
            return [logits, value_result, lstm_cell_output, lstm_hidden_output]

        return result_list

    def _split_feature_groups(self, feature_vec):
        offset = 0

        def take(size):
            nonlocal offset
            chunk = feature_vec[:, offset : offset + size]
            offset += size
            return chunk

        self_hero = take(self.self_hero_dim)
        enemy_hero = take(self.enemy_hero_dim)
        self_tower = take(self.self_tower_dim)
        enemy_tower = take(self.enemy_tower_dim)
        enemy_soldiers = take(self.soldier_slot_dim * self.soldier_slot_count).reshape(
            -1, self.soldier_slot_count, self.soldier_slot_dim
        )
        friendly_soldiers = take(self.soldier_slot_dim * self.soldier_slot_count).reshape(
            -1, self.soldier_slot_count, self.soldier_slot_dim
        )
        global_context = take(self.global_context_dim)
        matchup = take(self.matchup_dim)

        if offset != self.feature_dim:
            raise RuntimeError("Feature slice mismatch: expect {0}, got {1}".format(self.feature_dim, offset))

        return {
            "self_hero": self_hero,
            "enemy_hero": enemy_hero,
            "self_tower": self_tower,
            "enemy_tower": enemy_tower,
            "enemy_soldiers": enemy_soldiers,
            "friendly_soldiers": friendly_soldiers,
            "global_context": global_context,
            "matchup": matchup,
        }

    def _encode_entities(self, grouped):
        batch_size = grouped["self_hero"].shape[0]

        self_hero = self.self_hero_encoder(grouped["self_hero"])
        enemy_hero = self.enemy_hero_encoder(grouped["enemy_hero"])
        self_tower = self.self_tower_encoder(grouped["self_tower"])
        enemy_tower = self.enemy_tower_encoder(grouped["enemy_tower"])
        global_context = self.global_context_encoder(grouped["global_context"])
        matchup_encoded = self.matchup_encoder(grouped["matchup"])

        enemy_soldiers = self._encode_soldier_slots(grouped["enemy_soldiers"], batch_size)
        friendly_soldiers = self._encode_soldier_slots(grouped["friendly_soldiers"], batch_size)

        fused = torch.cat(
            [
                self_hero,
                enemy_hero,
                self_tower,
                enemy_tower,
                enemy_soldiers,
                friendly_soldiers,
                global_context,
                matchup_encoded,
            ],
            dim=1,
        )
        return self.fusion_mlp(fused)

    def _encode_soldier_slots(self, soldier_tensor, batch_size):
        flat = soldier_tensor.reshape(-1, self.soldier_slot_dim)
        encoded = self.soldier_encoder(flat)
        return encoded.reshape(batch_size, self.soldier_slot_count * 32)

    def _extract_loss_inputs(self, data_list: List[torch.Tensor]) -> Dict[str, Any]:
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [weight.squeeze(dim=1) for weight in usq_weight_list]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)
        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        return {
            "reward": reward,
            "advantage": advantage,
            "frame_is_train": frame_is_train,
            "label_list": label_list,
            "weight_list": weight_list,
            "old_label_probability_list": old_label_probability_list,
            "legal_action_flag_list": legal_action_flag_list,
        }

    @staticmethod
    def _compute_value_loss(value_result: torch.Tensor, reward: torch.Tensor, advantage: torch.Tensor, clip_param: float) -> torch.Tensor:
        fc_value_result = value_result.squeeze(dim=1)
        v_target = reward
        v_old_value = reward - advantage
        value_diff = fc_value_result - v_old_value
        clipped_v = v_old_value + torch.clamp(value_diff, -clip_param, clip_param)
        value_loss_unclipped = torch.square(v_target - fc_value_result)
        value_loss_clipped = torch.square(v_target - clipped_v)
        return 0.5 * torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))

    def _compute_policy_loss(
        self,
        label_result: List[torch.Tensor],
        label_list: List[torch.Tensor],
        old_label_probability_list: List[torch.Tensor],
        legal_action_flag_list: List[torch.Tensor],
        advantage: torch.Tensor,
        weight_list: List[torch.Tensor],
        frame_is_train: torch.Tensor,
        device: torch.device,
        epsilon: float,
        zero: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        policy_cost = zero
        label_probability_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if not self.is_reinforce_task_list[task_index]:
                continue

            one_hot_actions = nn.functional.one_hot(
                label_list[task_index].long(),
                self.label_size_list[task_index],
            )
            legal_mask = legal_action_flag_list[task_index]
            label_logits = label_result[task_index]
            label_probability = masked_softmax_torch(
                label_logits,
                legal_mask,
                self.min_policy,
            )
            label_probability_list.append(label_probability)

            policy_p = (one_hot_actions * label_probability).sum(1)
            policy_log_p = torch.log(policy_p + epsilon)
            old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
            old_policy_log_p = torch.log(old_policy_p)

            ratio = torch.exp(policy_log_p - old_policy_log_p).clamp(0.0, 3.0)
            surr1 = ratio * advantage
            surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            denominator = torch.maximum(
                torch.sum(weight_list[task_index].float() * frame_is_train),
                torch.tensor(1.0, device=device),
            )
            temp_policy_loss = -torch.sum(
                torch.minimum(surr1, surr2) * weight_list[task_index].float() * frame_is_train
            ) / denominator
            policy_cost = policy_cost + temp_policy_loss
        return policy_cost, label_probability_list

    def _compute_entropy_loss(
        self,
        label_probability_list: List[torch.Tensor],
        legal_action_flag_list: List[torch.Tensor],
        weight_list: List[torch.Tensor],
        frame_is_train: torch.Tensor,
        device: torch.device,
        epsilon: float,
        zero: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                denominator = torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train),
                    torch.tensor(1.0, device=device),
                )
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )
                temp_entropy_loss = -torch.sum(
                    temp_entropy_loss * weight_list[task_index].float() * frame_is_train
                ) / denominator
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index += 1
            else:
                entropy_loss_list.append(zero)

        entropy_cost = zero
        for entropy_element in entropy_loss_list:
            entropy_cost = entropy_cost + entropy_element
        return entropy_cost, entropy_loss_list

    def compute_loss(self, data_list: List[torch.Tensor], rst_list: List[torch.Tensor]) -> Tuple[torch.Tensor, list]:
        inputs = self._extract_loss_inputs(data_list)

        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        device = inputs["reward"].device
        epsilon = 1e-5
        zero = torch.tensor(0.0, device=device)

        self.value_cost = self._compute_value_loss(
            value_result, inputs["reward"], inputs["advantage"], self.clip_param
        )

        self.policy_cost, label_probability_list = self._compute_policy_loss(
            label_result,
            inputs["label_list"],
            inputs["old_label_probability_list"],
            inputs["legal_action_flag_list"],
            inputs["advantage"],
            inputs["weight_list"],
            inputs["frame_is_train"],
            device,
            epsilon,
            zero,
        )

        self.entropy_cost, self.entropy_cost_list = self._compute_entropy_loss(
            label_probability_list,
            inputs["legal_action_flag_list"],
            inputs["weight_list"],
            inputs["frame_is_train"],
            device,
            epsilon,
            zero,
        )

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [self.loss, [self.value_cost, self.policy_cost, self.entropy_cost]]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for index in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[index], fc_feat_dim_list[index + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, index + 1), fc_layer)
            if index + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, index + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
