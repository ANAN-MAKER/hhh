#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time

from agent_ppo.conf.conf import GameConfig
from agent_ppo.feature.definition import (
    FrameCollector,
    NONE_ACTION,
    build_frame,
    lineup_iterator_roundrobin_camp_heroes,
    sample_process,
)
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.env_conf_manager import EnvConfManager
from tools.metrics_utils import get_training_metrics
from tools.model_pool_utils import get_valid_model_pool


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    do_learns = [True, True]
    last_save_model_time = time.time()

    env_conf_manager = EnvConfManager(
        config_path="agent_ppo/conf/train_env_conf.toml",
        logger=logger,
    )
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes([112])

    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            for index, (do_learn, agent) in enumerate(zip(do_learns, agents)):
                if do_learn and len(g_data[index]) > 0:
                    agent.send_sample_data(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agents, logger, monitor, env_conf_manager, lineup_iterator):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator
        self.agent_num = len(agents)
        self.episode_cnt = 0

    def _call_init_config(self, usr_conf):
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)
        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
            if agent_idx == 0:
                my_hero_ids = blue_hero_ids
                opponent_hero_ids = red_hero_ids
                camp_key = camp_keys[0]
            else:
                my_hero_ids = red_hero_ids
                opponent_hero_ids = blue_hero_ids
                camp_key = camp_keys[1]

            config_data = {
                "my_camp": camp_key,
                "my_heroes": my_hero_ids,
                "opponent_heroes": opponent_hero_ids,
            }
            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.logger.info(f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}")

    def run_episodes(self):
        while True:
            training_metrics = get_training_metrics()
            if training_metrics:
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            lineup = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)
            self._call_init_config(usr_conf)

            env_obs = self.env.reset(usr_conf=usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs["observation"]
            self.reset_agents(observation, is_eval)
            frame_collector = FrameCollector(self.agent_num)

            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0.0] * self.agent_num
            reward_detail_sum_list = [self._new_reward_detail_sum() for _ in range(self.agent_num)]
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            for agent_index, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(agent_index)]["frame_state"])
                    observation[str(agent_index)]["reward"] = reward
                    self._accumulate_reward(reward, reward_sum_list, reward_detail_sum_list, agent_index)

            while True:
                actions = [NONE_ACTION] * self.agent_num

                for agent_index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if is_eval:
                            actions[agent_index] = agent.exploit(observation[str(agent_index)])
                        else:
                            actions[agent_index] = agent.predict(observation[str(agent_index)])

                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(agent_index)])
                            frame_collector.save_frame(frame, agent_id=agent_index)

                env_reward, env_obs = self.env.step(actions)
                if handle_disaster_recovery(env_obs, self.logger):
                    for agent in self.agents:
                        agent.finalize_trace_episode()
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                for agent_index, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(
                            observation[str(agent_index)]["frame_state"],
                            terminated=terminated,
                            truncated=truncated,
                            win=observation[str(agent_index)].get("win"),
                        )
                        observation[str(agent_index)]["reward"] = reward
                        self._accumulate_reward(reward, reward_sum_list, reward_detail_sum_list, agent_index)

                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, "
                        f"truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )

                    for agent_index, agent in enumerate(self.agents):
                        agent.finalize_trace_episode(observation[str(agent_index)])

                    for agent_index, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=agent_index,
                                reward=observation[str(agent_index)]["reward"]["reward_sum"],
                            )

                    if self.monitor:
                        monitor_data = self._build_episode_monitor_data(
                            observation=observation[str(monitor_side)],
                            reward_sum=reward_sum_list[monitor_side],
                            reward_detail_sum=reward_detail_sum_list[monitor_side],
                            frame_no=frame_no,
                            is_eval=is_eval,
                        )
                        self.monitor.put_data({os.getpid(): monitor_data})

                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        yield list_agents_samples
                    break

    def reset_agents(self, observation, is_eval):
        opponent_agent = self.env_conf_manager.get_opponent_agent()
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        for agent_index, agent in enumerate(self.agents):
            agent.configure_debug_context(
                mode="exploit" if is_eval else "predict",
                opponent_type=opponent_agent,
            )

            if agent_index == monitor_side:
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    self.do_predicts[agent_index] = False
                    self.do_samples[agent_index] = False
                elif opponent_agent == "selfplay":
                    agent.load_model(id="latest")
                else:
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    if is_train_test:
                        self.logger.info("Run train_test, cannot get opponent agent, so replace with latest model")
                        agent.load_model(id="latest")
                    else:
                        agent.load_opponent_agent(id=opponent_agent)
                    self.do_samples[agent_index] = False

            agent.reset(observation[str(agent_index)])

    def _new_reward_detail_sum(self):
        return {
            "terminal_reward": 0.0,
            "pushing_reward": 0.0,
            "combat_reward": 0.0,
            "farming_reward": 0.0,
            "tempo_reward": 0.0,
        }

    def _accumulate_reward(self, reward, reward_sum_list, reward_detail_sum_list, index):
        reward_sum_list[index] += reward.get("reward_sum", 0.0)
        for key in reward_detail_sum_list[index]:
            reward_detail_sum_list[index][key] += reward.get(key, 0.0)

    def _build_episode_monitor_data(self, observation, reward_sum, reward_detail_sum, frame_no, is_eval):
        frame_state = observation["frame_state"]
        player_id = observation["player_id"]
        main_hero = None
        for hero in frame_state.get("hero_states", []):
            if hero.get("runtime_id") == player_id or hero.get("player_id") == player_id:
                main_hero = hero
                break

        hero_camp = main_hero.get("camp") if main_hero is not None else observation.get("player_camp")
        enemy_tower_hp = None
        self_tower_hp = None
        for npc in frame_state.get("npc_states", []):
            if not self._is_normal_tower(npc):
                continue
            if npc.get("camp") == hero_camp:
                self_tower_hp = float(npc.get("hp", 0))
            else:
                enemy_tower_hp = float(npc.get("hp", 0))

        win = observation.get("win")
        if win is None:
            win_rate = 0.5
        else:
            win_rate = 1.0 if bool(win) else 0.0

        monitor_data = {
            "episode_cnt": self.episode_cnt,
            "reward_sum": round(reward_sum, 4),
            "terminal_reward": round(reward_detail_sum["terminal_reward"], 4),
            "pushing_reward": round(reward_detail_sum["pushing_reward"], 4),
            "combat_reward": round(reward_detail_sum["combat_reward"], 4),
            "farming_reward": round(reward_detail_sum["farming_reward"], 4),
            "tempo_reward": round(reward_detail_sum["tempo_reward"], 4),
            "win_rate": round(win_rate, 4),
            "frame": frame_no,
        }
        if self_tower_hp is not None:
            monitor_data["self_tower_hp"] = round(self_tower_hp, 2)
        if enemy_tower_hp is not None:
            monitor_data["enemy_tower_hp"] = round(enemy_tower_hp, 2)
        if frame_no > 0 and main_hero is not None:
            money_value = self._first_existing_value(main_hero, ("money", "money_cnt", "gold"))
            if money_value is not None:
                monitor_data["money_per_frame"] = round(float(money_value) / float(frame_no), 4)
            if "kill_cnt" in main_hero:
                monitor_data["kill"] = float(main_hero.get("kill_cnt", 0) or 0)
            if "dead_cnt" in main_hero:
                monitor_data["death"] = float(main_hero.get("dead_cnt", 0) or 0)
            if "total_hurt_to_hero" in main_hero:
                monitor_data["hurt_to_hero"] = round(float(main_hero.get("total_hurt_to_hero", 0) or 0), 2)
            if "total_be_hurt_by_hero" in main_hero:
                monitor_data["hurt_by_hero"] = round(float(main_hero.get("total_be_hurt_by_hero", 0) or 0), 2)
        if is_eval:
            monitor_data["reward"] = round(reward_sum, 4)
        return monitor_data

    def _is_normal_tower(self, npc):
        sub_type = npc.get("sub_type")
        config_id = self._safe_int(npc.get("config_id"))
        if config_id in GameConfig.TOWER_CONFIG_IDS:
            return True
        if isinstance(sub_type, int):
            return sub_type == GameConfig.SUB_TYPE_TOWER
        sub_type_text = str(sub_type).upper()
        return "TOWER" in sub_type_text and "SPRING" not in sub_type_text

    def _safe_int(self, value, default=-1):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _first_existing_value(self, payload, keys):
        for key in keys:
            if key in payload:
                return payload.get(key)
        return None
