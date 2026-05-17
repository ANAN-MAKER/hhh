# Tencent AI Arena HoK 1v1 Environment：Mojia Map 整理版

> 来源：Tencent AI Arena Competition 官方文档  
> 页面主题：Honor of Kings Environments / 1v1 Environment: Mojia Map  
> 整理日期：2026-05-17

---

## 1. 文档定位

该页面主要介绍 Tencent AI Arena 中 `hok` 包提供的王者荣耀环境，重点说明环境的：

- 观测空间（Observation Space）
- 动作空间（Action Space）
- 合法动作约束（Legal Action）
- 奖励函数配置（Reward）

需要注意：

- 虽然训练阶段可能提供多个环境，但比赛初赛阶段主要评测 `HoK1v1`。
- 所有环境都提供：
  - 默认空动作：`env.get_noop_action()`
  - 随机动作：`env.get_random_action()`
- 环境的观测和动作空间统一使用 `numpy.array()` 表示。

---

## 2. Honor of Kings 游戏背景

《Honor of Kings》（王者荣耀）是一款 MOBA 游戏。玩家控制一个英雄，通过移动、释放技能、攻击敌方单位来获得收益，并最终推掉敌方防御塔和水晶。

在游戏过程中，玩家主要通过以下信息进行决策：

- 主屏幕中的局部视野
- 左上角小地图
- 英雄状态
- 小兵状态
- 防御塔和水晶状态
- 敌方单位是否可见

游戏目标可以概括为：

> 在保护己方防御塔和水晶的同时，摧毁敌方防御塔和基地水晶。

---

## 3. 1v1 Environment：Mojia Map

### 3.1 环境名称

`HoK1v1` / Mojia Map / 墨家机关道

### 3.2 基本设定

Mojia Map 是一个 1v1 环境：

- 地图中有两个阵营。
- 每个阵营有一个英雄。
- 英雄由智能体控制。
- 目标是摧毁对方防御塔和基地，同时保护己方建筑。

### 3.3 适合研究的问题

该环境适合用于强化学习中的：

- 单智能体决策
- 战斗策略学习
- 走位与技能释放策略
- 资源获取策略
- 推塔策略
- 稀疏奖励与稠密奖励结合的训练

---

## 4. Observation Space：观测空间

观测空间提供当前游戏状态的信息，智能体根据这些状态决定下一步动作。

### 4.1 观测空间组成

`HoK1v1` 中，一个英雄的观测空间主要由 5 个部分组成：

| 组件 | 含义 | 主要内容 |
|---|---|---|
| `public_hero_feature` | 公开英雄特征 | 英雄是否存活、英雄 ID、血量、技能状态、技能相关信息等 |
| `private_hero_feature` | 私有英雄特征 | 英雄击杀等更具体的私有信息 |
| `soldier_vec_feature` | 小兵特征 | 小兵位置、血量、是否存活、所属阵营等 |
| `organ_feature` | 建筑特征 | 防御塔、水晶等建筑状态 |
| `vec_feature_global` | 全局特征 | 游戏时间等全局信息 |

### 4.2 观测空间注意点

官方文档中特别说明：

1. 当敌方单位对己方不可见时，其状态不会出现在当前观测中。
2. 不同英雄使用相同的观测空间，这有利于跨英雄泛化。
3. 比赛中不允许修改观测空间，这样可以减少参赛者在特征工程上的差异。

### 4.3 观测空间可以怎么理解

可以把观测空间理解成智能体的“游戏感知输入”。它不是原始游戏画面，而是整理后的结构化状态向量。

简单理解：

```text
Observation = 英雄状态 + 敌我信息 + 小兵信息 + 建筑信息 + 全局时间信息
```

### 4.4 主要字段示例

官方表格字段非常长，下面整理成可读版。

#### 4.4.1 英雄状态相关字段

| 字段 | 含义 |
|---|---|
| `is_hero_alive` | 英雄是否存活 |
| `hero_level` | 英雄等级 |
| `hp` | 当前生命值 |
| `hp_rate` | 当前生命值比例 |
| `max_hp` | 最大生命值 |
| `hp_recover` | 生命恢复 |
| `ep` | 当前法力值 / 能量值 |
| `ep_rate` | 法力值 / 能量值比例 |
| `max_ep` | 最大法力值 / 能量值 |
| `ep_recover` | 法力恢复 |
| `phy_atk` | 物理攻击 |
| `mgc_atk` | 法术攻击 |
| `phy_def` | 物理防御 |
| `mgc_def` | 法术防御 |
| `kill_cnt` | 击杀数 |
| `dead_cnt` | 死亡数 |
| `money_cnt` | 金币数量 |
| `exp` | 经验值 |
| `money` | 当前金币 |
| `revive_time` | 复活时间 |
| `kill_income` | 击杀该英雄可获得收益 |

#### 4.4.2 位置和距离字段

| 字段 | 含义 |
|---|---|
| `location_x` | 全局 x 坐标 |
| `location_z` | 全局 z 坐标 |
| `dist_from_all_heros` | 与敌方英雄距离 |
| `hero_move_speed` | 英雄移动速度 |
| `hero_attack_range` | 英雄攻击范围 |
| `hero_attack_speed` | 英雄攻击速度 |

#### 4.4.3 技能状态字段

| 字段 | 含义 |
|---|---|
| `skill_1_useable` | 一技能是否可用 |
| `hero_skill_1_cd` | 一技能冷却时间 |
| `skill_2_useable` | 二技能是否可用 |
| `hero_skill_2_cd` | 二技能冷却时间 |
| `skill_3_useable` | 三技能是否可用 |
| `hero_skill_3_cd` | 三技能冷却时间 |
| `skill_4_useable` | 四技能是否可用，部分英雄才有 |
| `hero_skill_4_cd` | 四技能冷却时间 |
| `heal_skill_cd` | 恢复技能冷却时间 |
| `summon_skill_cd` | 召唤师技能冷却时间 |

#### 4.4.4 Buff / 控制相关字段

| 字段 | 含义 |
|---|---|
| `good_skill_buff_on_hero_itself` | 英雄自身是否有技能带来的增益 |
| `avoid_skill_control` | 是否处于免控等控制抵抗状态 |
| `blood_return` | 是否有回血类效果 |
| `ctrl_reduce` | 韧性 / 控制减免 |
| `cd_reduce` | 冷却缩减 |

#### 4.4.5 小兵特征字段

| 字段 | 含义 |
|---|---|
| `is_in_exp_range` | 是否在主英雄经验范围 / 视野范围内 |
| `is_soldier_alive` | 小兵是否存活 |
| `belong_to_main_camp` | 是否属于己方阵营 |
| `soldier_type` | 小兵类型，如普通兵、超级兵等 |
| `location_x` | 小兵全局 x 坐标 |
| `location_z` | 小兵全局 z 坐标 |
| `relative_location_x` | 与主英雄的相对 x 坐标 |
| `relative_location_z` | 与主英雄的相对 z 坐标 |
| `dist_from_each_hero` | 与双方英雄的距离 |
| `hp` | 小兵生命值 |
| `hp_rate` | 小兵生命值比例 |
| `max_hp` | 小兵最大生命值 |
| `atk` | 小兵攻击力 |
| `kill_income` | 击杀小兵收益 |
| `buff_marks` | 是否存在 buff 标记 |

#### 4.4.6 建筑特征字段

| 字段 | 含义 |
|---|---|
| `is_alive` | 建筑是否存活 |
| `belong_to_main_camp` | 是否属于己方阵营 |
| `type` | 建筑类型，如己方水晶、己方一塔、敌方水晶、敌方一塔等 |
| `location_x` | 建筑全局 x 坐标 |
| `location_z` | 建筑全局 z 坐标 |
| `relative_location_x` | 与主英雄的相对 x 坐标 |
| `relative_location_z` | 与主英雄的相对 z 坐标 |
| `distance_from_each_hero` | 与双方英雄距离 |
| `hp` | 建筑生命值 |
| `hp_rate` | 建筑生命值比例 |
| `max_hp` | 建筑最大生命值 |
| `atk` | 建筑攻击力 |
| `kill_income` | 推掉该建筑的收益 |
| `attack_range` | 建筑攻击范围 |

#### 4.4.7 全局特征字段

| 字段 | 含义 |
|---|---|
| `g_game_time` | 游戏时间分段。官方说明中将前 10 分钟均分成 5 段，超过 10 分钟视为最后一段 |

### 4.5 观测空间维度理解

官方长表中最后一段显示全局特征范围为 `486 ~ 491`，因此可以将整理版理解为：

```text
HoK1v1 observation vector ≈ 491 维
```

注意：这里是根据页面表格整理出来的总维度理解，实际使用时仍应以 `hok` 包返回的 observation 结构为准。

---

## 5. Action Space：动作空间

### 5.1 动作空间总体结构

Mojia Map 的原生动作空间是分层结构。一个动作大致由以下部分组成：

```text
Action = Button + Move Offset + Skill Offset + Target
```

也可以理解为：

1. 选择要做什么动作，例如移动、普攻、释放技能。
2. 选择作用目标，例如自己、敌方英雄、小兵、防御塔。
3. 选择动作方向，例如移动方向或技能释放方向。

### 5.2 动作类别整理

| Action Class | Type | Description | Dimension |
|---|---|---|---:|
| Button | None | 无动作 | 1 |
| Button | Move | 移动英雄 | 1 |
| Button | Normal Attack | 释放普通攻击 | 1 |
| Button | Skill 1 | 释放一技能 | 1 |
| Button | Skill 2 | 释放二技能 | 1 |
| Button | Skill 3 | 释放三技能 | 1 |
| Button | Heal Skill | 释放恢复技能 | 1 |
| Button | Chosen Skill | 释放选定技能 | 1 |
| Button | Recall | 回城，如果不被打断，几秒后回到泉水 | 1 |
| Button | Skill 4 | 释放四技能，仅部分英雄有效 | 1 |
| Button | Equipment Skill | 释放装备提供的技能 | 1 |
| Move | Move X | x 轴移动方向 | 16 |
| Move | Move Z | z 轴移动方向 | 16 |
| Skill | Skill X | 技能 x 轴释放方向 | 16 |
| Skill | Skill Z | 技能 z 轴释放方向 | 16 |
| Target | None | 空目标 | 1 |
| Target | Self | 自己 | 1 |
| Target | Enemy | 敌方英雄 | 1 |
| Target | Soldier | 最近 4 个小兵 | 4 |
| Target | Tower | 最近防御塔 | 1 |

---

## 6. Sub Action Mask：子动作掩码

### 6.1 作用

`Sub action mask` 用来描述：

> 当顶层动作 Button 被选定后，哪些子动作是允许的。

例如：

- 如果选择的是移动动作，那么移动方向相关子动作有效。
- 如果选择的是方向性技能，那么技能方向相关子动作有效。
- 如果某个技能不是方向性技能，则不需要选择技能方向。

### 6.2 为什么需要子动作掩码

如果没有子动作掩码，智能体可能会产生很多无意义组合，例如：

- 选择普攻，但同时给出技能方向。
- 选择回城，但同时指定攻击目标。
- 选择不可用技能，却仍然输出技能释放方向。

子动作掩码可以减少无效动作空间，提高训练效率。

---

## 7. Legal Action：合法动作

### 7.1 作用

`legal_action` 表示当前状态下英雄允许执行哪些动作。

它会结合游戏规则与当前状态，过滤掉不合理动作。

### 7.2 过滤内容

官方文档中提到，合法动作主要用于消除以下不合理情况：

1. 技能或攻击不可用。
   - 例如技能还在冷却时，不应该释放该技能。
2. 英雄被控制。
   - 例如被敌方技能或装备效果控制时，某些动作不能执行。
3. 英雄或装备自身的特殊限制。
   - 不同英雄、不同装备可能导致合法动作不同。

### 7.3 合法动作维度

官方给出的合法动作维度为：

```text
12 (Button) + 16 * 2 (Move) + 16 * 2 (Skill) + 8 (Target) * 12 (Button)
```

计算如下：

```text
12 + 32 + 32 + 96 = 172
```

因此可以理解为：

```text
legal_action 约为 172 维
```

---

## 8. Reward：奖励函数

Mojia Map 的奖励函数同时包含稀疏奖励和稠密奖励。

### 8.1 奖励类别

官方文档将奖励分为四类：

| 类别 | 含义 | 奖励特点 |
|---|---|---|
| Farming related | 金币和经验获取 | 稠密奖励 |
| KDA related | 击杀和死亡 | 稀疏奖励 |
| Damage related | 英雄生命值变化 | 稠密奖励 |
| Pushing related | 攻击敌方防御塔和水晶 | 稠密奖励 |

### 8.2 奖励项整理

| Reward | Weight | Type | Description |
|---|---:|---|---|
| `hp_point` | 2 | dense | 英雄生命值比例 |
| `tower_hp_point` | 5 | dense | 防御塔生命值比例 |
| `money` / `gold` | 0.006 | dense | 获取的总金币 |
| `ep_rate` | 0.75 | dense | 法力值比例 |
| `death` | -1 | sparse | 被击杀 |
| `kill` | -0.6 | sparse | 击杀敌方英雄 |
| `exp` | 0.006 | dense | 获取的经验 |
| `last_hit` | 0.5 | sparse | 对小兵完成最后一击 |

> 注意：官方表格中 `kill` 的权重写为 `-0.6`。实际训练时应结合环境源码或比赛配置确认该符号是否与最终奖励方向一致。

---

## 9. 从强化学习角度如何理解这个环境

### 9.1 状态 State

状态就是观测向量，包含：

```text
英雄状态 + 小兵状态 + 建筑状态 + 全局时间
```

### 9.2 动作 Action

动作是分层离散动作，包含：

```text
动作按钮 + 移动方向 + 技能方向 + 目标选择
```

### 9.3 奖励 Reward

奖励由多个目标共同组成：

```text
生存 + 发育 + 击杀 + 推塔 + 补刀
```

### 9.4 策略 Policy 要学习的内容

智能体需要学会：

- 什么时候补兵
- 什么时候进攻敌方英雄
- 什么时候后撤
- 什么时候释放技能
- 技能朝哪个方向释放
- 攻击小兵、英雄、防御塔之间如何权衡
- 如何在发育、击杀、推塔之间平衡

---

## 10. 训练时需要特别关注的问题

### 10.1 动作空间不是简单离散动作

动作由多个子动作组成，因此模型输出时通常要处理：

- Button 分类
- Move X / Move Z 分类
- Skill X / Skill Z 分类
- Target 分类
- Sub action mask
- Legal action mask

### 10.2 Mask 很重要

训练时如果不使用合法动作 mask，模型会频繁输出无效动作，例如：

- 技能冷却中仍释放技能
- 被控制时尝试移动或释放技能
- 非方向性技能却预测方向
- 没有目标时强行攻击

### 10.3 奖励设计影响策略倾向

不同奖励权重会改变智能体行为：

- 金币 / 经验奖励高：更偏向发育。
- 击杀奖励高：更偏向打架。
- 推塔奖励高：更偏向推进。
- 死亡惩罚高：更偏向保守。

### 10.4 观测不是完整全图信息

敌方不可见单位不会出现在当前观测中，因此智能体需要处理信息不完全的问题。

这意味着策略不能只依赖“全图真值”，而要学会在部分可观测条件下决策。

---

## 11. 一句话总结

`HoK1v1` 的 Mojia Map 是一个结构化的王者荣耀 1v1 强化学习环境。智能体接收约 491 维的状态向量，输出由动作按钮、移动方向、技能方向和目标选择组成的分层动作，并通过生存、发育、击杀、推塔、补刀等奖励信号学习对战策略。

---

## 12. 快速复习版

```text
环境：HoK1v1 / Mojia Map / 墨家机关道
目标：摧毁敌方防御塔和基地，保护己方建筑
观测：英雄 + 小兵 + 建筑 + 全局时间
动作：Button + Move + Skill + Target
Mask：Sub action mask + Legal action
奖励：发育、KDA、伤害、推塔、补刀
核心难点：分层动作空间 + 合法动作约束 + 部分可观测 + 多目标奖励
```
