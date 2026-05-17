#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from kaiwudrl.interface.agent import BaseAgent
from torch.optim.lr_scheduler import LambdaLR

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.decision_trace import DecisionTracer
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.feature_process import FeatureProcess
from agent_ppo.feature.reward_process import GameRewardManager
from agent_ppo.model.model import Model, masked_softmax_torch

torch.set_num_threads(Config.TORCH_NUM_THREADS)
torch.set_num_interop_threads(Config.TORCH_NUM_INTEROP_THREADS)

_SEED = getattr(Config, "SEED", 42)
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


SUMMONER_SKILL_MAP = {
    80102: "heal",
    80109: "sprint",
    80104: "punish",
    80108: "execute",
    80110: "frenzy",
    80105: "interference",
    80103: "stun",
    80107: "purify",
    80121: "weaken",
    80115: "flash",
}
SUMMONER_SKILL_IDS = list(SUMMONER_SKILL_MAP.keys())
POLICY_BRANCH_NAMES = ["button", "move_x", "move_z", "skill_x", "skill_z", "target"]


class Agent(BaseAgent):
    def __init__(self, agent_type: str = "player", device=None, logger=None, monitor=None) -> None:
        self.cur_model_name: str = ""
        self.device = device
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        self.lstm_unit_size: int = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.lstm_cell = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.label_size_list: List[int] = Config.LABEL_SIZE_LIST
        self.legal_action_size: List[int] = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        self.hero_camp: int = 0
        self.player_id: int = 0
        self.debug_mode: str = "predict"
        self.debug_opponent_type: str = "unknown"
        self.current_frame_no: int = -1
        self._legal_action_debug_count: int = 0

        self.lr: float = Config.INIT_LEARNING_RATE_START
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor
        self.feature_processes = None
        self.trace_writer = DecisionTracer(logger=logger, monitor=monitor)
        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step: int) -> float:
        if step > self.target_step:
            return self.target_lr / self.lr
        return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data: Dict[str, Any]) -> Dict[int, int]:
        my_heroes = config_data.get("my_heroes", [])
        select_skills = {}
        for my_hero_id in my_heroes:
            select_skills[my_hero_id] = 80115
        return select_skills

    def _recommend_summoner_skill(self, my_hero_id: int, opponent_hero_id: int) -> int:
        if my_hero_id == 112:
            if opponent_hero_id == 112:
                return random.choice([80115, 80103])
            else:
                return random.choice([80107, 80115])
        else:
            if opponent_hero_id == 112:
                return random.choice([80107, 80115])
            else:
                return random.choice([80115, 80110])

    def configure_debug_context(self, mode: str, opponent_type: str) -> None:
        self.debug_mode = mode
        self.debug_opponent_type = opponent_type
        self.trace_writer.set_context(mode, opponent_type)

    def reset(self, observation: Dict[str, Any]) -> None:
        self.hero_camp = observation.get("camp", observation.get("player_camp"))
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.lstm_cell = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(self.hero_camp)
        self._legal_action_debug_count = 0
        self.trace_writer.start_episode(observation, self.hero_camp, self.player_id)

    def _model_inference(self, list_obs_data: List[ObsData]) -> List[ActData]:
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        feature_vec = torch.as_tensor(np.asarray(feature), dtype=torch.float32, device=self.device).reshape(
            -1, self.seri_vec_split_shape[0][0]
        )
        lstm_hidden_state = torch.as_tensor(np.asarray(lstm_hidden), dtype=torch.float32, device=self.device).reshape(
            -1, self.lstm_unit_size
        )
        lstm_cell_state = torch.as_tensor(np.asarray(lstm_cell), dtype=torch.float32, device=self.device).reshape(
            -1, self.lstm_unit_size
        )

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model([feature_vec, lstm_hidden_state, lstm_cell_state], inference=True)

        logits, value, next_lstm_cell, next_lstm_hidden = [output.detach().cpu().numpy() for output in output_list]
        next_lstm_cell = next_lstm_cell.squeeze(axis=0)
        next_lstm_hidden = next_lstm_hidden.squeeze(axis=0)

        list_act_data = []
        for index in range(len(legal_action)):
            self._validate_raw_legal_action(legal_action[index], log_probe=False)
            prob, d_prob, action, d_action, policy_debug = self._sample_masked_action(logits[index], legal_action[index])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value[index],
                    lstm_cell=next_lstm_cell[index],
                    lstm_hidden=next_lstm_hidden[index],
                    policy_debug=policy_debug,
                )
            )
        return list_act_data

    def predict(self, observation: Dict[str, Any]) -> List[int]:
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        self.trace_writer.record(observation, obs_data, act_data, action, is_stochastic=True)
        return action

    def exploit(self, observation: Dict[str, Any]) -> List[int]:
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, False)
        self.trace_writer.record(observation, obs_data, act_data, action, is_stochastic=False)
        return action

    def observation_process(self, observation: Dict[str, Any]) -> ObsData:
        self.current_frame_no = int(observation.get("frame_state", {}).get("frame_no", -1))
        self._validate_raw_legal_action(observation["legal_action"], frame_no=self.current_frame_no)
        feature_packet = self.feature_processes.process_observation(observation)
        return ObsData(
            feature=feature_packet["feature_vector"],
            legal_action=observation["legal_action"],
            lstm_cell=self.lstm_cell,
            lstm_hidden=self.lstm_hidden,
            feature_groups=feature_packet["feature_groups"],
            visible_context=feature_packet["visible_context"],
            feature_metadata=feature_packet["feature_metadata"],
        )

    def action_process(self, observation: Dict[str, Any], act_data: ActData, is_stochastic: bool) -> List[int]:
        if is_stochastic:
            return act_data.action
        return act_data.d_action

    def learn(self, list_sample_data) -> Dict[str, Any]:
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path: Optional[str] = None, id: str = "1") -> None:
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        model_state_dict_cpu = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path: Optional[str] = None, id: str = "1") -> None:
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return
        if not os.path.exists(model_file_path):
            if self.logger:
                self.logger.warning(f"model {model_file_path} not found, use fresh initialized model")
            return

        try:
            state_dict = torch.load(model_file_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            if "size mismatch" in str(exc):
                raise RuntimeError(
                    "模型输入特征维度已从 101 升级到 117，旧 checkpoint 与当前 PPO 输入不兼容，需要重新训练后再加载。"
                ) from exc
            raise

        self.cur_model_name = model_file_path
        self.logger.info(f"load model {model_file_path} successfully")

    def load_opponent_agent(self, id: str = "1") -> None:
        pass

    def finalize_trace_episode(self, observation: Optional[Dict[str, Any]] = None) -> None:
        self.trace_writer.finalize_episode(observation)

    def update_status(self, obs_data: ObsData, act_data: ActData) -> None:
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _sample_masked_action(self, logits: np.ndarray, legal_action: List[int]) -> tuple:
        self._validate_raw_legal_action(legal_action, log_probe=False)
        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        branch_probabilities = {}
        selected_probabilities = {"sampled": {}, "deterministic": {}}

        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(np.array(legal_action), label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])

        for branch_index in range(0, len(self.label_size_list) - 1):
            branch_name = POLICY_BRANCH_NAMES[branch_index]
            probs = self._legal_soft_max(logits_split[branch_index], legal_actions[branch_index])
            branch_probabilities[branch_name] = probs.tolist()
            prob_list += list(probs)
            d_prob_list += list(probs)

            sample_action = self._legal_sample(probs, use_max=False)
            deterministic_action = self._legal_sample(probs, use_max=True)
            action_list.append(sample_action)
            d_action_list.append(deterministic_action)
            selected_probabilities["sampled"][branch_name] = float(probs[sample_action])
            selected_probabilities["deterministic"][branch_name] = float(probs[deterministic_action])

        target_branch_index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[target_branch_index],
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )

        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]].reshape(self.label_size_list[0], 1)
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        target_probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        branch_probabilities["target"] = {"sampled": target_probs.tolist()}
        prob_list += list(target_probs)
        sampled_target = self._legal_sample(target_probs, use_max=False)
        action_list.append(sampled_target)
        selected_probabilities["sampled"]["target"] = float(target_probs[sampled_target])

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]].reshape(self.label_size_list[0], 1)
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        deterministic_target_probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        branch_probabilities["target"]["deterministic"] = deterministic_target_probs.tolist()
        d_prob_list += list(deterministic_target_probs)
        deterministic_target = self._legal_sample(deterministic_target_probs, use_max=True)
        d_action_list.append(deterministic_target)
        selected_probabilities["deterministic"]["target"] = float(deterministic_target_probs[deterministic_target])

        policy_debug = {
            "branch_probabilities": branch_probabilities,
            "selected_probabilities": selected_probabilities,
            "target_legal_mask": {
                "sampled": target_legal_action.tolist(),
                "deterministic": target_legal_action_d.tolist(),
            },
        }

        return [prob_list], [d_prob_list], action_list, d_action_list, policy_debug

    def _validate_raw_legal_action(
        self,
        legal_action: List[int],
        frame_no: Optional[int] = None,
        log_probe: bool = True,
    ) -> None:
        actual_len = len(legal_action)
        expected_len = Config.RAW_LEGAL_ACTION_DIM
        if frame_no is None:
            frame_no = self.current_frame_no
        if log_probe and self._legal_action_debug_count < 5:
            if self.logger:
                self.logger.info(
                    f"raw legal_action length check: len={actual_len} expected={expected_len} "
                    f"label_size={self.label_size_list} frame_probe={self._legal_action_debug_count + 1} "
                    f"frame_no={frame_no} player_id={self.player_id}"
                )
            self._legal_action_debug_count += 1
        if actual_len != expected_len:
            raise ValueError(
                "raw legal_action length mismatch: actual={0}, expected={1}, frame_no={2}, "
                "agent_id/player_id={3}, label_size={4}".format(
                    actual_len,
                    expected_len,
                    frame_no,
                    self.player_id,
                    self.label_size_list,
                )
            )

    def _legal_soft_max(self, input_hidden, legal_action):
        logits = torch.as_tensor(input_hidden, dtype=torch.float32)
        mask = torch.as_tensor(legal_action, dtype=torch.float32)
        probs = masked_softmax_torch(logits, mask, Config.MIN_POLICY)
        return probs.detach().cpu().numpy()

    def _legal_sample(self, probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))
