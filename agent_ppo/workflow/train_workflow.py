#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO.
峡谷追猎 PPO 训练工作流。
"""

import os
import time

import numpy as np
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read user config / 读取用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Reset env / 重置环境
            env_obs = self.env.reset(self.usr_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            
            # 三层奖励统计 (Layer A, B, C)
            layer_a_total = 0.0  # 主目标层
            layer_b_total = 0.0  # 引导层
            layer_c_total = 0.0  # 约束层
            
            # A层内部拆分
            survival_reward_total = 0.0
            score_objective_total = 0.0
            
            # B层内部拆分
            distance_shaping_total = 0.0
            danger_zone_penalty_total = 0.0
            buff_bonus_total = 0.0
            flash_escape_bonus_total = 0.0
            
            # C层内部拆分
            movement_penalty_total = 0.0
            loop_penalty_total = 0.0
            wasteful_flash_penalty_total = 0.0
            
            # 其他统计
            exploration_bonus = 0.0
            anti_loiter_penalty = 0.0
            treasure_bonus = 0.0
            escape_bonus = 0.0
            flash_bonus = 0.0
            total_move_dist = 0.0
            total_min_monster_dist = 0.0
            total_nearest_treasure_dist = 0.0
            total_stuck_score = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # Step env / 与环境交互
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # Step reward / 每步即时奖励
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                reward_info = _remain_info.get("reward_info", {})
                total_reward += float(reward[0])
                
                # 统计三层奖励
                layer_a_total += float(reward_info.get("layer_a", 0.0))
                layer_b_total += float(reward_info.get("layer_b", 0.0))
                layer_c_total += float(reward_info.get("layer_c", 0.0))
                
                # A层内部拆分
                survival_reward_total += float(reward_info.get("survival_reward", 0.0))
                score_objective_total += float(reward_info.get("score_objective", 0.0))
                
                # B层内部拆分
                distance_shaping_total += float(reward_info.get("distance_shaping", 0.0))
                danger_zone_penalty_total += float(reward_info.get("danger_zone_penalty", 0.0))
                buff_bonus_total += float(reward_info.get("buff_bonus", 0.0))
                flash_escape_bonus_total += float(reward_info.get("flash_escape_bonus", 0.0))
                
                # C层内部拆分
                movement_penalty_total += float(reward_info.get("movement_penalty", 0.0))
                loop_penalty_total += float(reward_info.get("loop_penalty", 0.0))
                wasteful_flash_penalty_total += float(reward_info.get("wasteful_flash_penalty", 0.0))
                
                # 其他统计（向后兼容）
                exploration_bonus += float(reward_info.get("exploration_reward", 0.0))
                anti_loiter_penalty += float(reward_info.get("loiter_penalty", 0.0))
                treasure_bonus += float(reward_info.get("treasure_pickup_reward", 0.0))
                treasure_bonus += float(reward_info.get("treasure_approach_reward", 0.0))
                escape_bonus += float(reward_info.get("dist_shaping", 0.0))
                escape_bonus += float(reward_info.get("emergency_escape_reward", 0.0))
                flash_bonus += float(reward_info.get("flash_reward", 0.0))
                total_move_dist += float(reward_info.get("move_dist", 0.0))
                total_min_monster_dist += float(reward_info.get("min_monster_dist_norm", 1.0))
                total_nearest_treasure_dist += float(reward_info.get("nearest_treasure_dist_norm", 1.0))
                total_stuck_score += float(reward_info.get("stuck_score", 0.0))

                # Terminal reward / 终局奖励
                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = float(env_info.get("total_score", 0.0))
                    max_step = max(int(env_info.get("max_step", step)), 1)
                    total_treasure = max(int(env_info.get("total_treasure", 0)), 0)
                    treasures_collected = int(env_info.get("treasures_collected", 0))
                    collected_buff = int(env_info.get("collected_buff", 0))
                    flash_count = int(env_info.get("flash_count", 0))
                    treasure_ratio = treasures_collected / max(total_treasure, 1) if total_treasure > 0 else 0.0
                    step_ratio = min(step / max_step, 1.0)
                    score_upper_bound = max(max_step * 1.5 + total_treasure * 100.0, 1.0)
                    score_ratio = float(np.clip(total_score / score_upper_bound, 0.0, 1.0))

                    if terminated:
                        final_reward[0] = -12.0 + 2.0 * treasure_ratio + 2.0 * step_ratio + 1.5 * score_ratio
                        result_str = "FAIL"
                        win_flag = 0.0
                    else:
                        final_reward[0] = 12.0 + 3.0 * treasure_ratio + 2.0 * step_ratio + 2.0 * score_ratio
                        result_str = "WIN"
                        win_flag = 1.0

                    self.logger.info(
                        f"[GAMEOVER] episode:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} score:{total_score:.1f} "
                        f"treasure:{treasures_collected}/{total_treasure} "
                        f"total_reward:{total_reward:.3f} "
                        f"| Layer_A:{layer_a_total:.3f} Layer_B:{layer_b_total:.3f} Layer_C:{layer_c_total:.3f}"
                    )

                # Build sample frame / 构造样本帧
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                # Episode end / 对局结束
                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    # Monitor report / 监控上报
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        avg_divisor = max(step, 1)
                        monitor_data = {
                            # 基础统计
                            "episode_reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_score": round(total_score, 4),
                            "episode_step_ratio": round(step_ratio, 4),
                            "episode_treasure_count": treasures_collected,
                            "episode_treasure_ratio": round(treasure_ratio, 4),
                            "episode_buff_count": collected_buff,
                            "episode_flash_count": flash_count,
                            "win_flag": round(win_flag, 4),
                            
                            # 三层奖励拆分
                            "layer_a_total": round(layer_a_total, 4),  # 主目标层总奖励
                            "layer_b_total": round(layer_b_total, 4),  # 引导层总奖励
                            "layer_c_total": round(layer_c_total, 4),  # 约束层总奖励
                            
                            # A层内部
                            "survival_reward_total": round(survival_reward_total, 4),
                            "score_objective_total": round(score_objective_total, 4),
                            
                            # B层内部
                            "distance_shaping_total": round(distance_shaping_total, 4),
                            "danger_zone_penalty_total": round(danger_zone_penalty_total, 4),
                            "buff_bonus_total": round(buff_bonus_total, 4),
                            "flash_escape_bonus_total": round(flash_escape_bonus_total, 4),
                            
                            # C层内部
                            "movement_penalty_total": round(movement_penalty_total, 4),
                            "loop_penalty_total": round(loop_penalty_total, 4),
                            "wasteful_flash_penalty_total": round(wasteful_flash_penalty_total, 4),
                            
                            # 平均值
                            "avg_step_reward": round(total_reward / avg_divisor, 4),
                            "avg_move_dist": round(total_move_dist / avg_divisor, 4),
                            "avg_min_monster_dist": round(total_min_monster_dist / avg_divisor, 4),
                            "avg_nearest_treasure_dist": round(total_nearest_treasure_dist / avg_divisor, 4),
                            "avg_stuck_score": round(total_stuck_score / avg_divisor, 4),
                            
                            # 向后兼容的奖励项
                            "exploration_bonus": round(exploration_bonus, 4),
                            "anti_loiter_penalty": round(anti_loiter_penalty, 4),
                            "treasure_bonus": round(treasure_bonus, 4),
                            "escape_bonus": round(escape_bonus, 4),
                            "flash_bonus": round(flash_bonus, 4),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info
