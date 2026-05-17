#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
import random
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
    lineup_iterator_roundrobin_camp_heroes,
)
from agent_ppo.conf.conf import GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_manager = EnvConfManager(
        config_path="agent_ppo/conf/train_env_conf.toml",
        logger=logger,
    )

    # Lineup iterator (112:Luban, 133:DiRenjie)
    # 阵容迭代器 (112:鲁班， 133:狄仁杰)
    # Stage A trains Luban only; hero one-hot dimensions still reserve DiRenjie for later expansion.
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes([112])

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
    )

    while True:
        # Run episodes and collect data
        # 运行对局并收集数据
        for g_data in episode_runner.run_episodes():
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
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
        self.last_report_monitor_time = 0

    def _call_init_config(self, usr_conf):
        """Call init_config on both agents to get summoner skill selections,
        then inject the results into usr_conf.
        调用双方 agent 的 init_config 获取召唤师技能选择，并注入 usr_conf。
        """
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        # camp_keys[i] is the camp key for agents[i] based on monitor_side
        # monitor_side 的 agent 对应 blue/red 取决于 monitor_side 配置
        monitor_side = self.env_conf_manager.get_monitor_side()
        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
            # Determine which camp this agent controls
            # 确定该 agent 控制哪个阵营
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
            self.logger.info(
                f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}"
            )

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            lineup = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)

            # Call init_config on agents to get summoner skill selections
            # 调用 agent 的 init_config 获取召唤师技能选择，注入 usr_conf
            self._call_init_config(usr_conf)

            # Start a new environment
            # 启动新对局，返回初始环境状态

            env_obs = self.env.reset(usr_conf=usr_conf)
            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs["observation"]
            extra_info = env_obs["extra_info"]

            # Reset agents
            # 重置智能体
            self.reset_agents(observation)

            # Reset environment frame collector
            # 重置环境帧收集器
            frame_collector = FrameCollector(self.agent_num)

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0] * self.agent_num
            episode_stats = [self._new_episode_stats() for _ in range(self.agent_num)]
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]

            while True:
                # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
                # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if not is_eval:
                            actions[index] = agent.predict(observation[str(index)])
                        else:
                            actions[index] = agent.exploit(observation[str(index)])

                        # Only sample when do_sample=True and is_eval=False
                        # 评估对局数据不采样，不是训练中最新模型产生的数据不采样
                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                    self._update_episode_stats(episode_stats[index], observation[str(index)], actions[index])

                # Step forward
                # 推进环境到下一帧，得到新的状态
                env_reward, env_obs = self.env.step(actions)
                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                extra_info = env_obs["extra_info"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                # Reward generation
                # 计算回报，作为当前环境状态observation的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"], actions[i])
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]
                        self._update_episode_outcome_stats(episode_stats[i], observation[str(i)])

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )
                    # Reward for saving the last state of the environment
                    # 保存环境最后状态的reward
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"]["reward_sum"],
                            )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = self._build_monitor_data(
                            observation[str(monitor_side)],
                            reward_sum_list[monitor_side],
                            actions[monitor_side],
                            is_eval,
                            self.episode_cnt,
                            episode_stats[monitor_side],
                        )
                        if self.monitor:
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        yield list_agents_samples
                    break

    def _build_monitor_data(self, observation, reward_sum, action, is_eval, episode_cnt, episode_stats):
        frame_state = observation["frame_state"]
        camp = observation.get("camp", -1)
        main_hero = self._get_hero(frame_state, camp)
        enemy_hero = self._get_enemy_hero(frame_state, camp)
        main_tower = self._get_tower(frame_state, camp)
        enemy_tower = self._get_enemy_tower(frame_state, camp)
        reward_items = observation.get("reward", {})
        main_config = int(self._value(main_hero, "config_id"))
        enemy_config = int(self._value(enemy_hero, "config_id"))
        frame_no = max(float(frame_state.get("frame_no", 0.0) or 0.0), 1.0)
        frames = max(float(episode_stats.get("frames", 0.0)), 1.0)
        self_hp_ratio = self._hp_ratio(main_hero)
        enemy_hp_ratio = self._hp_ratio(enemy_hero)
        self_tower_hp_ratio = self._hp_ratio(main_tower)
        enemy_tower_hp_ratio = self._hp_ratio(enemy_tower)
        self_money = self._resource(main_hero, "money")
        enemy_money = self._resource(enemy_hero, "money")

        enemy_tower_dead = enemy_tower is not None and self._value(enemy_tower, "hp") <= 0
        main_tower_dead = main_tower is not None and self._value(main_tower, "hp") <= 0
        if enemy_tower_dead and not main_tower_dead:
            win_rate = 1.0
        elif main_tower_dead and not enemy_tower_dead:
            win_rate = 0.0
        else:
            win_rate = 0.5 if is_eval else 0.0

        monitor_data = {
            "episode_cnt": episode_cnt,
            "reward": round(float(reward_sum), 2),
            "episode_win": win_rate,
            "controlled_hero_id": main_config,
            "opponent_hero_id": enemy_config,
            "self_tower_hp": round(self._value(main_tower, "hp"), 2),
            "enemy_tower_hp": round(self._value(enemy_tower, "hp"), 2),
            "self_tower_hp_ratio": round(self_tower_hp_ratio, 4),
            "enemy_tower_hp_ratio": round(enemy_tower_hp_ratio, 4),
            "tower_hp_advantage": round(self_tower_hp_ratio - enemy_tower_hp_ratio, 4),
            "self_hp_ratio": round(self_hp_ratio, 4),
            "enemy_hp_ratio": round(enemy_hp_ratio, 4),
            "hp_advantage": round(self_hp_ratio - enemy_hp_ratio, 4),
            "self_money": round(self_money, 2),
            "enemy_money": round(enemy_money, 2),
            "gold_advantage": round(self_money - enemy_money, 2),
            "money_per_frame": round(self._resource(main_hero, "money") / frame_no, 4),
            "kill": 1.0 if enemy_hero and self._value(enemy_hero, "hp") <= 0 else 0.0,
            "death": 1.0 if main_hero and self._value(main_hero, "hp") <= 0 else 0.0,
            "lane_progress": round(self._lane_progress(main_hero, main_tower, enemy_tower), 4),
            "my_soldier_count": len(self._soldiers(frame_state, camp)),
            "enemy_soldier_count": len(self._enemy_soldiers(frame_state, camp)),
            "move_action_rate": round(episode_stats["move_action"] / frames, 4),
            "attack_action_rate": round(episode_stats["attack_action"] / frames, 4),
            "recall_action_rate": round(episode_stats["recall_action"] / frames, 4),
            "hero_target_rate": round(episode_stats["hero_target"] / frames, 4),
            "creep_target_rate": round(episode_stats["creep_target"] / frames, 4),
            "tower_target_rate": round(episode_stats["tower_target"] / frames, 4),
            "enemy_tower_range_rate": round(episode_stats["enemy_tower_range"] / frames, 4),
            "no_minion_tower_dive_rate": round(episode_stats["no_minion_tower_dive"] / frames, 4),
            "safe_push_window_rate": round(episode_stats["safe_push_window"] / frames, 4),
            "tower_attack_with_minion_rate": round(episode_stats["tower_attack_with_minion"] / frames, 4),
            "reward_hp": round(reward_items.get("reward_hp", 0.0), 4),
            "reward_tower": round(reward_items.get("reward_tower", 0.0), 4),
            "reward_gold": round(reward_items.get("reward_gold", 0.0), 4),
            "reward_exp": round(reward_items.get("reward_exp", 0.0), 4),
            "reward_last_hit": round(reward_items.get("reward_last_hit", 0.0), 4),
            "reward_death": round(reward_items.get("reward_death", 0.0), 4),
            "reward_kill": round(reward_items.get("reward_kill", 0.0), 4),
            "reward_forward": round(reward_items.get("reward_forward", 0.0), 4),
            "reward_win": round(reward_items.get("reward_win", 0.0), 4),
            "reward_tower_danger": round(reward_items.get("reward_tower_danger", 0.0), 4),
            "reward_push_assist": round(reward_items.get("reward_push_assist", 0.0), 4),
            "reward_lane_follow": round(reward_items.get("reward_lane_follow", 0.0), 4),
            "reward_finish": round(reward_items.get("reward_finish", 0.0), 4),
            "reward_stage_a": 1.0 if reward_items.get("reward_stage") == GameConfig.REWARD_STAGE else 0.0,
            "skill_action_rate": round(episode_stats["skill_action"] / frames, 4),
            "summoner_action_rate": round(episode_stats["summoner_action"] / frames, 4),
            "low_movement_rate": round(episode_stats["low_movement"] / frames, 4),
            "enemy_tower_hp_drop": round(episode_stats["enemy_tower_hp_drop"], 2),
            "hurt_to_hero_delta": round(episode_stats["hurt_to_hero_delta"], 2),
            "gold_from_hero": round(episode_stats["gold_from_hero"], 2),
            "gold_from_soldier": round(episode_stats["gold_from_soldier"], 2),
            "gold_from_other": round(episode_stats["gold_from_other"], 2),
            "gold_unattributed": round(episode_stats["gold_unattributed"], 2),
            "gold_total_delta": round(episode_stats["gold_total_delta"], 2),
            "soldier_kill_count": round(episode_stats["soldier_kill_count"], 2),
            "hero_kill_count": round(episode_stats["hero_kill_count"], 2),
        }
        if is_eval:
            monitor_data["win_rate_model_266689"] = win_rate
            monitor_data["win_rate_eval_model"] = win_rate
        else:
            monitor_data["win_rate_train_common_ai"] = win_rate
        return monitor_data

    def _new_episode_stats(self):
        return {
            "frames": 0.0,
            "move_action": 0.0,
            "attack_action": 0.0,
            "recall_action": 0.0,
            "hero_target": 0.0,
            "creep_target": 0.0,
            "tower_target": 0.0,
            "enemy_tower_range": 0.0,
            "no_minion_tower_dive": 0.0,
            "safe_push_window": 0.0,
            "tower_attack_with_minion": 0.0,
            "skill_action": 0.0,
            "summoner_action": 0.0,
            "low_movement": 0.0,
            "last_hero_pos": None,
            "last_enemy_tower_hp": None,
            "last_hurt_to_hero": None,
            "last_self_money": None,
            "enemy_tower_hp_drop": 0.0,
            "hurt_to_hero_delta": 0.0,
            "gold_from_hero": 0.0,
            "gold_from_soldier": 0.0,
            "gold_from_other": 0.0,
            "gold_unattributed": 0.0,
            "gold_total_delta": 0.0,
            "soldier_kill_count": 0.0,
            "hero_kill_count": 0.0,
        }

    def _update_episode_stats(self, stats, observation, action):
        frame_state = observation["frame_state"]
        camp = observation.get("camp", -1)
        main_hero = self._get_hero(frame_state, camp)
        enemy_hero = self._get_enemy_hero(frame_state, camp)
        enemy_tower = self._get_enemy_tower(frame_state, camp)
        my_soldiers = self._soldiers(frame_state, camp)
        button = int(action[0]) if action is not None and len(action) > 0 else -1
        target = int(action[5]) if action is not None and len(action) > 5 else -1
        in_enemy_tower_range = self._in_tower_range(main_hero, enemy_tower)
        has_minion_under_tower = self._has_ally_minion_under_enemy_tower(my_soldiers, enemy_tower)
        safe_push_window = has_minion_under_tower and self._hp_ratio(main_hero) >= 0.30 and self._hp_ratio(enemy_hero) < 0.70

        stats["frames"] += 1.0
        stats["move_action"] += 1.0 if button == GameConfig.BUTTON_MOVE else 0.0
        stats["attack_action"] += 1.0 if button == GameConfig.BUTTON_ATTACK else 0.0
        stats["recall_action"] += 1.0 if button == GameConfig.BUTTON_RECALL else 0.0
        stats["skill_action"] += 1.0 if button in GameConfig.SKILL_BUTTONS else 0.0
        stats["summoner_action"] += 1.0 if button == GameConfig.BUTTON_SUMMONER else 0.0
        stats["hero_target"] += 1.0 if target == GameConfig.TARGET_ENEMY else 0.0
        stats["creep_target"] += 1.0 if target in GameConfig.TARGET_SOLDIERS else 0.0
        stats["tower_target"] += 1.0 if target == GameConfig.TARGET_TOWER else 0.0
        stats["enemy_tower_range"] += 1.0 if in_enemy_tower_range else 0.0
        stats["no_minion_tower_dive"] += 1.0 if in_enemy_tower_range and not has_minion_under_tower else 0.0
        stats["safe_push_window"] += 1.0 if safe_push_window else 0.0
        stats["tower_attack_with_minion"] += 1.0 if target == GameConfig.TARGET_TOWER and has_minion_under_tower else 0.0
        hero_pos = self._position(main_hero)
        if stats["last_hero_pos"] is not None and self._distance(hero_pos, stats["last_hero_pos"]) < 300.0:
            stats["low_movement"] += 1.0
        stats["last_hero_pos"] = hero_pos

    def _update_episode_outcome_stats(self, stats, observation):
        frame_state = observation["frame_state"]
        camp = observation.get("camp", -1)
        main_hero = self._get_hero(frame_state, camp)
        enemy_tower = self._get_enemy_tower(frame_state, camp)

        enemy_tower_hp = self._value(enemy_tower, "hp")
        if stats["last_enemy_tower_hp"] is not None:
            stats["enemy_tower_hp_drop"] += max(0.0, stats["last_enemy_tower_hp"] - enemy_tower_hp)
        stats["last_enemy_tower_hp"] = enemy_tower_hp

        hurt_to_hero = self._value(main_hero, "total_hurt_to_hero")
        if stats["last_hurt_to_hero"] is not None:
            stats["hurt_to_hero_delta"] += max(0.0, hurt_to_hero - stats["last_hurt_to_hero"])
        stats["last_hurt_to_hero"] = hurt_to_hero

        self_money = self._resource(main_hero, "money")
        money_delta = 0.0
        if stats["last_self_money"] is not None:
            money_delta = max(0.0, self_money - stats["last_self_money"])
            stats["gold_total_delta"] += money_delta
        stats["last_self_money"] = self_money

        attributed_money = 0.0
        frame_action = frame_state.get("frame_action", {}) or {}
        for dead_action in frame_action.get("dead_action", []):
            death = dead_action.get("death", {}) if isinstance(dead_action, dict) else {}
            killer = dead_action.get("killer", {}) if isinstance(dead_action, dict) else {}
            if not killer or killer.get("camp") != camp:
                continue
            income = killer.get("income_info", {}) or {}
            money = float(income.get("money", 0.0) or 0.0)
            sub_type = int(death.get("sub_type", -1) or -1)
            actor_type = int(death.get("actor_type", -1) or -1)
            if actor_type == 1 or death.get("config_id") in GameConfig.HERO_IDS:
                stats["gold_from_hero"] += money
                stats["hero_kill_count"] += 1.0
                attributed_money += money
            elif sub_type not in GameConfig.TOWER_SUB_TYPES and sub_type not in GameConfig.CRYSTAL_SUB_TYPES:
                stats["gold_from_soldier"] += money
                stats["soldier_kill_count"] += 1.0
                attributed_money += money
            else:
                stats["gold_from_other"] += money
                attributed_money += money
        if money_delta > attributed_money:
            stats["gold_unattributed"] += money_delta - attributed_money

    def _lane_progress(self, hero, main_tower, enemy_tower):
        if not hero or not main_tower or not enemy_tower:
            return 0.0
        start = self._position(main_tower)
        end = self._position(enemy_tower)
        point = self._position(hero)
        vx, vz = end[0] - start[0], end[1] - start[1]
        lane_len_sq = max(vx * vx + vz * vz, 1.0)
        return max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vz) / lane_len_sq))

    def _get_hero(self, frame_state, camp):
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") == camp:
                return hero
        return None

    def _get_enemy_hero(self, frame_state, camp):
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") != camp:
                return hero
        return None

    def _get_tower(self, frame_state, camp):
        for npc in frame_state.get("npc_states", []):
            if npc.get("camp") == camp and int(self._value(npc, "sub_type")) in GameConfig.TOWER_SUB_TYPES:
                return npc
        return None

    def _get_enemy_tower(self, frame_state, camp):
        for npc in frame_state.get("npc_states", []):
            if npc.get("camp") != camp and int(self._value(npc, "sub_type")) in GameConfig.TOWER_SUB_TYPES:
                return npc
        return None

    def _soldiers(self, frame_state, camp):
        return [
            npc
            for npc in frame_state.get("npc_states", [])
            if isinstance(npc, dict)
            and npc.get("camp") == camp
            and not self._is_organ(npc)
            and self._value(npc, "hp") > 0
        ]

    def _enemy_soldiers(self, frame_state, camp):
        return [
            npc
            for npc in frame_state.get("npc_states", [])
            if isinstance(npc, dict)
            and npc.get("camp") != camp
            and not self._is_organ(npc)
            and self._value(npc, "hp") > 0
        ]

    def _is_organ(self, unit):
        sub_type = int(self._value(unit, "sub_type"))
        return sub_type in GameConfig.TOWER_SUB_TYPES or sub_type in GameConfig.CRYSTAL_SUB_TYPES

    def _hp_ratio(self, unit):
        max_hp = self._value(unit, "max_hp")
        if max_hp <= 0:
            return 0.0
        return max(0.0, min(1.0, self._value(unit, "hp") / max_hp))

    def _in_tower_range(self, hero, tower):
        if not hero or not tower:
            return False
        tower_range = self._value(tower, "attack_range") or 9000.0
        return self._distance(self._position(hero), self._position(tower)) <= tower_range

    def _has_ally_minion_under_enemy_tower(self, soldiers, enemy_tower):
        if not enemy_tower:
            return False
        tower_range = self._value(enemy_tower, "attack_range") or 9000.0
        tower_pos = self._position(enemy_tower)
        return any(self._distance(self._position(soldier), tower_pos) <= tower_range for soldier in soldiers)

    def _distance(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _position(self, unit):
        if not unit:
            return 0.0, 0.0
        location = unit.get("location", {})
        return float(location.get("x", 0.0) or 0.0), float(location.get("z", 0.0) or 0.0)

    def _value(self, unit, key):
        if not unit:
            return 0.0
        return float(unit.get(key, 0.0) or 0.0)

    def _resource(self, unit, key):
        if key == "money":
            return self._value_any(unit, ("money", "gold", "money_cnt"))
        return self._value(unit, key)

    def _value_any(self, unit, keys):
        for key in keys:
            value = self._value(unit, key)
            if value:
                return value
        return 0.0

    def reset_agents(self, observation):
        opponent_agent = self.env_conf_manager.get_opponent_agent()
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # do_predicts 指定哪些智能体要进行模型预测
        # The 'do_samples' specifies which agents are to perform training sampling.
        # do_samples 指定哪些智能体要进行训练采样
        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        # Load model according to the configuration
        # 根据对局配置加载模型
        for i, agent in enumerate(self.agents):
            # Report the latest model in the training camp to the monitor
            # 训练中最新模型所在阵营上报监控
            if i == monitor_side:
                # monitor_side uses the latest model
                # monitor_side 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    agent.load_model(id="latest")
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        if is_train_test:
                            # Run train_test, cannot get opponent agent, so replace with latest model
                            # 运行 train_test 时, 无法获取到对手模型，因此将替换为最新模型
                            self.logger.info(f"Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False
            # Reset agent
            # 重置agent
            agent.reset(observation[str(i)])
