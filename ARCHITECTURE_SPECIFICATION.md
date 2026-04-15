#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Module Architecture and Naming Convention Guide.
模块架构与命名规范指南。

该文档定义了整个项目的模块组织、命名规范和职责边界。
确保代码的可维护性和可扩展性。

═══════════════════════════════════════════════════════════════════════════════
完整的信息流管道
═══════════════════════════════════════════════════════════════════════════════

  环境观测 (128x128 map + entities)
      ↓
  [空间坐标对齐层] (spatial_utils.py)
      - normalize_global_pos()
      - relative_pos_normalized()
      - action_to_delta()
      - consistency_check()
  ↓
  特征编码 (preprocessor.py → 159D features)
      - 自身状态编码 (24D)
      - 外部实体编码 (34D)
      - 地图与路径编码 (35D)
      - 时序与记忆编码 (30D)
      - 决策辅助编码 (20D)
  ↓
  [动作质量评估块] (action_quality.py)
      - safety_score: 非法 vs 危险 vs 安全
      - treasure_gain_score: 靠近宝箱
      - buff_gain_score: 靠近 buff
      - terrain_score: 方向开阔性
      - revisit_penalty: 绮圈风险
      - flash_value_score: 闪现价值
      → 输出: 16×6=96D 动作规划特征 (已嵌入 preprocessor)
  ↓
  模型前向传播 (model.py)
      ↓
      STAGE 1: 特征编码 (MLPBlock encoders)
          - self_progress_encoder (24+20D → 64D)
          - monster_encoder + fusion (14D → 64D)
          - treasure_encoder (12D → 64D)
          - buff_encoder (8D → 16D)
          - map_encoder (35D → 64D)
          - plan_encoder (30D → 96D)
          - legal_encoder (16D → 16D)
      ↓
      STAGE 2: 风险收益融合块 (RiskRewardFusionBlock)
          - 风险等级评估 (威胁+地图 → risk_scores)
          - 收益机会评估 (宝箱/buff+时序 → opportunity_scores)
          - 决策风格建议 (综合风险+收益 → decision_scores)
          → 输出: 128D 融合决策上下文
      ↓
      STAGE 3: 主干网络 (Backbone with ResidualBlocks)
          → 输出: 128D 隐层
      ↓
      STAGE 4: 输出头
          - Policy Head: 128D → 16D (动作 logits)
          - Value Head: 128D → 1D (价值估计)
  ↓
  奖励计算 (reward_system_v2.py + reward_spec.py)
      - A 层: 生存 + 得分 (主目标)
      - B 层: 距离成形 + 距离 + buff + 闪现 (引导)
      - C 层: 低效 + 绮圈 + 陷阱 + 浪费 (约束)
      → 返回: RewardInfo (标准化结构)
  ↓
  训练监控 (training_monitor.py + workflow.py)
      - 收集每步 RewardInfo
      - 按 A/B/C 三层聚合
      - 生成 EpisodeStatistics
      - 输出标准化日志

═══════════════════════════════════════════════════════════════════════════════
模块级命名规范
═══════════════════════════════════════════════════════════════════════════════

【FEATURE 模块 (agent_ppo/feature/)】

① preprocessor.py
   职责: 原始观测 → 159D 特征向量（包含动作质量评估）
   输出:
     - obs_data.feature: (159,) float array
     - obs_data.legal_action: (16,) binary array
   核心接口:
     - Preprocessor.forward(env_obs) → (feature, legal_action)
     - 内部调用 action_quality.compute_action_quality_features()

② spatial_utils.py
   职责: 统一所有空间表达和方向映射
   核心函数:
     - normalize_global_pos(x, z) → (x_norm, z_norm)
     - relative_pos_normalized(hero_pos, target_pos) → (dx, dz, dist)
     - action_to_delta(action_id) → (dx, dz)
     - consistency_check() → Dict[str, bool]
   使用场景:
     - preprocessor 中的坐标处理
     - 特征工程中的方向编码
     - 动作质量评估中的位置预测

③ action_quality.py
   职责: 评估 16 个动作的质量（6 维度）
   核心类:
     - ActionQualityEvaluator
       方法: evaluate_all_actions() → (16, 6) ndarray
       返回: 16×6=96D flatten 后送入模型
   输出维度:
     - [0]: safety_score (远离怪物)
     - [1]: treasure_gain_score (靠近宝箱)
     - [2]: buff_gain_score (靠近 buff)
     - [3]: terrain_score (开阔性)
     - [4]: revisit_penalty (绮圈风险)
     - [5]: flash_value_score (闪现价值)

④ reward_system_v2.py
   职责: 三层奖励计算（已存在，不需修改）
   关键方法:
     - RewardCalculator.calculate_survival_objective()
     - RewardCalculator.calculate_score_objective()
     - RewardCalculator.calculate_distance_shaping()
     - 等等（B/C 层方法）

⑤ reward_spec.py (NEW)
   职责: 奖励信息标准化结构和规范定义
   核心类:
     - RewardInfo: 单步奖励（包含 A/B/C 三层详细分项）
     - EpisodeStatistics: 对局统计（累计和聚合）
   辅助函数:
     - create_reward_info()
     - format_reward_step_log()
     - format_reward_episode_log()

⑥ definition.py
   职责: 数据结构定义（保持不变）
   关键类:
     - ObsData: 观测数据
     - ActData: 动作数据
     - SampleData: 训练样本数据

【MODEL 模块 (agent_ppo/model/)】

① model.py
   职责: 神经网络架构（已升级为 4 阶段）
   关键类:
     - MLPBlock: MLP 编码单元
     - ResidualBlock: 残差块
     - RiskRewardFusionBlock (NEW): 风险收益融合块
     - Model: 完整网络

   STAGE 1 编码器 (各自命名清晰):
     - self_progress_encoder: 自身状态+进度编码 → 64D
     - monster_encoder + monster_fusion: 威胁编码 → 64D
     - treasure_encoder: 宝箱编码 → 64D
     - buff_encoder: buff 编码 → 16D
     - map_encoder: 地图编码 → 64D
     - plan_encoder: 时序规划编码 → 96D
     - legal_encoder: 合法动作编码 → 16D

   STAGE 2 融合块:
     - risk_reward_fusion: RiskRewardFusionBlock 实例 → 128D

   STAGE 3 主干:
     - backbone: 残差网络 → 128D

   STAGE 4 输出头:
     - actor_head: 策略头 → 16D (动作 logits)
     - critic_head: 价值头 → 1D (价值)

【WORKFLOW 模块 (agent_ppo/workflow/)】

① train_workflow.py
   职责: 主训练循环（对外接口保持不变）
   核心类:
     - EpisodeRunner
   关键方法:
     - run_episodes(): 对局生成器

② training_monitor.py (NEW)
   职责: 训练监控与统计聚合
   核心类:
     - TrainingMonitor
       方法:
         - record_step_reward(reward_info, env_info)
         - finalize_episode() → EpisodeStatistics
         - get_window_statistics() → Dict
         - log_episode_summary()

【CONF 模块 (agent_ppo/conf/)】

① conf.py
   职责: 配置常量（需要复查，确保命名反映职责）
   关键常数:
     - FEATURE_SPLIT_SHAPE: [24, 7, 7, 12, 8, 20, 35, 30, 16]
       含义: [self, monster1, monster2, treasure, buff, progress, map, plan, legal]
     - ACTION_NUM: 16
     - VALUE_NUM: 1
     - GAMMA, LAMDA, 等 PPO 超参

═══════════════════════════════════════════════════════════════════════════════
对象命名规范
═══════════════════════════════════════════════════════════════════════════════

🔹 编码器类命名 (在模型中)
   <何物>_<目标> 格式
   例如:
     - self_progress_encoder: 自身状态+第三阶段信息的编码器
     - monster_encoder: 怪物信息编码器
     - monster_fusion: 双怪怪物融合器
     - treasure_encoder: 宝箱信息编码器
     - buff_encoder: buff 信息编码器
     - map_encoder: 地图信息编码器
     - plan_encoder: 规划和时序信息编码器
     - legal_encoder: 合法动作掩码编码器

🔹 特征命名 (在 preprocessor 中)
   <含义>_<特征> 格式
   例如:
     - self_feat: 自身状态特征 (24D)
     - monster_hidden: 怪物编码输出 (64D)
     - map_hidden: 地图编码输出 (64D)
     - treasure_hidden: 宝箱编码输出 (64D)
     - action_quality_features: 动作质量评估特征 (96D)

🔹 奖励字段命名 (在 RewardInfo 中)
   <奖励类型>_<具体项> 格式
   例如:
     - survival_base_reward: 基础生存奖励
     - completion_bonus: 完成奖励
     - treasure_pickup_reward: 宝箱拾取奖励
     - dist_shaping: 距离成形
     - danger_zone_penalty: 危险区惩罚
     - buff_acquisition_reward: buff 获取奖励
     - flash_escape_reward: 闪现脱险奖励
     - loiter_penalty: 绮圈惩罚

🔹 统计指标命名 (在 EpisodeStatistics 中)
   layer_<字母>_<描述> 格式
   例如:
     - layer_a_cumulative: A 层累计奖励
     - layer_b_cumulative: B 层累计奖励
     - layer_c_cumulative: C 层累计惩罚
     - avg_min_monster_dist: 平均最小怪物距离
     - reward_per_step: 每步平均奖励

═══════════════════════════════════════════════════════════════════════════════
文件与职责对应表
═══════════════════════════════════════════════════════════════════════════════

📂 agent_ppo/feature/
  ├── preprocessor.py              → 特征工程（调用 spatial_utils, action_quality）
  ├── spatial_utils.py             → 空间对齐（坐标、方向统一）
  ├── action_quality.py            → 动作质量评估
  ├── reward_system_v2.py          → 三层奖励计算
  ├── reward_spec.py               → 奖励规范与结构定义
  ├── definition.py                → 数据类定义
  └── conf/
      └── conf.py                  → 超参和维度定义

📂 agent_ppo/model/
  └── model.py                     → 神经网络（4 阶段架构）

📂 agent_ppo/workflow/
  ├── train_workflow.py            → 主训练循环
  └── training_monitor.py          → 监控统计

═══════════════════════════════════════════════════════════════════════════════
检查清单：确保每个模块职责清晰
═══════════════════════════════════════════════════════════════════════════════

✓ spatial_utils.py
  ├─ 提供全球坐标归一化
  ├─ 提供相对坐标变换
  ├─ 提供动作方向映射
  ├─ 提供局部地图方向解释
  └─ 提供一致性检查函数

✓ action_quality.py
  ├─ 评估 16 个动作的"质量"
  ├─ 输出 16×6=96D 特征
  ├─ 不做硬 mask（只提供建议）
  └─ 不改环境 legal_action

✓ model.py 中的 RiskRewardFusionBlock
  ├─ 综合威胁、机会、地图、时序
  ├─ 输出风险等级、收益机会、决策风格
  ├─ 不改 PPO loss
  └─ 不改输出维度

✓ reward_spec.py + reward_system_v2.py
  ├─ 定义 A/B/C 三层结构
  ├─ 返回标准化 RewardInfo
  ├─ 计算层级聚合值
  └─ 提供日志格式化函数

✓ training_monitor.py
  ├─ 收集每步 RewardInfo
  ├─ 聚合成 EpisodeStatistics
  ├─ 提供窗口统计
  └─ 输出清晰日志

═══════════════════════════════════════════════════════════════════════════════
后续扩展的开放接口
═══════════════════════════════════════════════════════════════════════════════

这个架构为以下扩展预留了清晰的接口：

1. **多头价值估计** (未来可在 critic_head 中添加多维价值)
   - 当前: value_head → 1D
   - 未来: 可扩展成 [survival_value, score_value, safety_value, ...]

2. **层级动作输出** (未来可改成两层动作选择)
   - 当前: policy_head → 16D (flat)
   - 未来: 可改成 [action_type_logits, direction_logits]

3. **注意力机制** (未来可在融合块中添加)
   - 当前: RiskRewardFusionBlock 用 MLP + Gate
   - 未来: 可升级到多头注意力

4. **长程依赖** (未来可添加 RNN/Transformer 层)
   - 当前: 没有显式的时序建模
   - 未来: 可在 plan_encoder 中加入 GRU/Transformer

5. **课程学习** (未来可在 workflow 中添加)
   - 当前: 一致的难度
   - 未来: 可根据 win_rate 动态调整环境难度

═══════════════════════════════════════════════════════════════════════════════
版本号和发布说明
═══════════════════════════════════════════════════════════════════════════════

当前版本: v1.0 (Unified Architecture with Clear Semantics)
发布日期: 2026-04-15

关键改进:
  - ✓ 统一空间坐标系 (spatial_utils.py)
  - ✓ 模型中加入风险收益融合块 (RiskRewardFusionBlock)
  - ✓ 标准化奖励结构 (reward_spec.py)
  - ✓ 统一训练监控 (training_monitor.py)
  - ✓ 清晰的模块命名和职责边界

已验证:
  - 外部接口保持不变 (159D 输入 → 16D 动作 + 1D 价值)
  - PPO 训练流程不变
  - legal action 使用方式不变
  - 向后兼容

下个版本计划 (v1.1):
  - 多维价值头 (multi-dimensional value)
  - 详细的超参调优指南
  - 可视化监控面板
"""


# 这是纯文档文件，无可执行代码

if __name__ == "__main__":
    print(__doc__)
