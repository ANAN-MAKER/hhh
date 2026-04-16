#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Spatial coordinate system alignment and transformation utilities.
统一的空间/坐标系对齐模块。

该模块统一管理游戏中的所有空间表达，包括：
  1. 全局坐标系 (Global Coordinate System)
  2. 局部地图索引 (Local Map Indexing)
  3. 相对坐标变换 (Relative Position Transform)
  4. 动作方向映射 (Action Direction Mapping)
  5. 方向一致性验证 (Direction Consistency Check)

地图坐标定义（严格对标环境）：
  - 原点: 左上角 (0, 0)
  - X 轴: 向右为正，范围 [0, 128)
  - Z 轴: 向下为正，范围 [0, 128)
  - 动作: 8个移动方向(0-7) + 8个闪现方向(8-15)
  
方向标准化定义（Action ID）- 对齐开发指南（z向下为正）：
  - 0: 右   (1, 0)    | 8: 右闪  (10, 0)
  - 1: 右上 (1, -1)   | 9: 右上闪 (8, -8)
  - 2: 上   (0, -1)   | 10: 上闪 (0, -10)
  - 3: 左上 (-1, -1)  | 11: 左上闪 (-8, -8)
  - 4: 左   (-1, 0)   | 12: 左闪 (-10, 0)
  - 5: 左下 (-1, 1)   | 13: 左下闪 (-8, 8)
  - 6: 下   (0, 1)    | 14: 下闪 (0, 10)
  - 7: 右下 (1, 1)    | 15: 右下闪 (8, 8)
"""

import numpy as np
from typing import Tuple, Dict, Optional


# ============================================================================
# 常数定义（全局统一，不允许分散定义）
# ============================================================================

MAP_SIZE = 128.0
MAP_DIAG = float(np.sqrt(MAP_SIZE**2 + MAP_SIZE**2))  # ~181

# 标准动作方向映射（与环境严格一致 - 对齐开发指南）
# ACTION_ID -> (dx, dz) 单位位移向量
# z向下为正，x向右为正
ACTION_ID_TO_DELTA = {
    # 8个移动方向 (normal movement, 单位步长)
    0: (1.0, 0.0),      # 右
    1: (1.0, -1.0),     # 右上 (* 修正：z向上为负)
    2: (0.0, -1.0),     # 上 (* 修正：z向上为负)
    3: (-1.0, -1.0),    # 左上 (* 修正：z向上为负)
    4: (-1.0, 0.0),     # 左
    5: (-1.0, 1.0),     # 左下
    6: (0.0, 1.0),      # 下
    7: (1.0, 1.0),      # 右下
    # 8个闪现方向 (flash, 大幅移动)
    8: (10.0, 0.0),     # 右闪
    9: (8.0, -8.0),     # 右上闪 (* 修正：z向上为负)
    10: (0.0, -10.0),   # 上闪 (* 修正：z向上为负)
    11: (-8.0, -8.0),   # 左上闪 (* 修正：z向上为负)
    12: (-10.0, 0.0),   # 左闪
    13: (-8.0, 8.0),    # 左下闪
    14: (0.0, 10.0),    # 下闪
    15: (8.0, 8.0),     # 右下闪
}

# 逆向动作映射（完全相反的方向）
ACTION_ID_TO_OPPOSITE = {
    0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3,
    8: 12, 9: 13, 10: 14, 11: 15, 12: 8, 13: 9, 14: 10, 15: 11,
}

# 方向名称映射（用于 debug 输出）
ACTION_ID_TO_NAME = {
    0: "right", 1: "down-right", 2: "down", 3: "down-left",
    4: "left", 5: "up-left", 6: "up", 7: "up-right",
    8: "flash-right", 9: "flash-down-right", 10: "flash-down", 11: "flash-down-left",
    12: "flash-left", 13: "flash-up-left", 14: "flash-up", 15: "flash-up-right",
}

# 局部地图中的方向对应关系（从中心指向边界）
# 用于判断"某个方向在局部地图中看起来如何"
LOCAL_MAP_DIRECTION_SLOTS = {
    # 8个主要方向，对应局部地图中的采样位置
    # 局部地图是21x21，map[row][col]，中心(10,10)
    # row: 向下(z+)增加 | col: 向右(x+)增加
    0: (10, 15),    # 右(x+): 中心行，右边界
    1: (5, 15),     # 右上(x+,z-): 右上角
    2: (5, 10),     # 上(z-): 上边界，中心列
    3: (5, 5),      # 左上(x-,z-): 左上角
    4: (10, 5),     # 左(x-): 中心行，左边界
    5: (15, 5),     # 左下(x-,z+): 左下角
    6: (15, 10),    # 下(z+): 下边界，中心列
    7: (15, 15),    # 右下(x+,z+): 右下角
}


# ============================================================================
# 1. 坐标归一化函数
# ============================================================================

def normalize_global_pos(
    x: float,
    z: float,
    map_size: float = MAP_SIZE,
    output_range: str = "[0,1]",
) -> Tuple[float, float]:
    """
    将全局坐标归一化到指定范围。
    
    Args:
        x: 全局 X 坐标 [0, MAP_SIZE)
        z: 全局 Z 坐标 [0, MAP_SIZE)
        map_size: 地图尺寸，默认 128
        output_range: 输出范围，"[0,1]" 或 "[-1,1]"
    
    Returns:
        (x_norm, z_norm): 归一化后的坐标
    
    Example:
        >>> normalize_global_pos(64, 64)  # 地图中心
        (0.5, 0.5)
        
        >>> normalize_global_pos(64, 64, output_range="[-1,1]")
        (0.0, 0.0)
    """
    x = float(np.clip(x, 0.0, map_size))
    z = float(np.clip(z, 0.0, map_size))
    
    if output_range == "[0,1]":
        x_norm = x / map_size
        z_norm = z / map_size
    elif output_range == "[-1,1]":
        x_norm = 2.0 * x / map_size - 1.0
        z_norm = 2.0 * z / map_size - 1.0
    else:
        raise ValueError(f"Invalid output_range: {output_range}")
    
    return float(x_norm), float(z_norm)


def denormalize_global_pos(
    x_norm: float,
    z_norm: float,
    map_size: float = MAP_SIZE,
    input_range: str = "[0,1]",
) -> Tuple[float, float]:
    """
    将归一化坐标恢复到全局坐标。
    
    Args:
        x_norm: 归一化 X 坐标
        z_norm: 归一化 Z 坐标
        map_size: 地图尺寸，默认 128
        input_range: 输入范围，"[0,1]" 或 "[-1,1]"
    
    Returns:
        (x, z): 全局坐标
    """
    if input_range == "[0,1]":
        x = x_norm * map_size
        z = z_norm * map_size
    elif input_range == "[-1,1]":
        x = (x_norm + 1.0) * map_size / 2.0
        z = (z_norm + 1.0) * map_size / 2.0
    else:
        raise ValueError(f"Invalid input_range: {input_range}")
    
    x = float(np.clip(x, 0.0, map_size))
    z = float(np.clip(z, 0.0, map_size))
    return x, z


# ============================================================================
# 2. 相对坐标变换函数
# ============================================================================

def relative_pos(
    hero_pos: Dict[str, float],
    target_pos: Dict[str, float],
    map_size: float = MAP_SIZE,
) -> Tuple[float, float, float]:
    """
    计算目标相对于英雄的相对坐标及距离。
    
    Args:
        hero_pos: 英雄位置 {"x": float, "z": float}
        target_pos: 目标位置 {"x": float, "z": float}
        map_size: 地图尺寸
    
    Returns:
        (dx, dz, dist): 相对坐标差值和欧氏距离
    
    Example:
        >>> hero_pos = {"x": 50, "z": 50}
        >>> treasure_pos = {"x": 60, "z": 50}
        >>> dx, dz, dist = relative_pos(hero_pos, treasure_pos)
        >>> print(dx, dz, dist)
        10.0 0.0 10.0
    """
    hero_x = float(hero_pos.get("x", 0.0))
    hero_z = float(hero_pos.get("z", 0.0))
    target_x = float(target_pos.get("x", 0.0))
    target_z = float(target_pos.get("z", 0.0))
    
    dx = target_x - hero_x
    dz = target_z - hero_z
    dist = float(np.sqrt(dx**2 + dz**2))
    
    return dx, dz, dist


def relative_pos_normalized(
    hero_pos: Dict[str, float],
    target_pos: Dict[str, float],
    map_size: float = MAP_SIZE,
) -> Tuple[float, float, float]:
    """
    计算目标的归一化相对坐标及距离。
    
    Args:
        hero_pos: 英雄位置
        target_pos: 目标位置
        map_size: 地图尺寸
    
    Returns:
        (dx_norm, dz_norm, dist_norm): 归一化的相对坐标和距离
        
    说明:
        - dx_norm, dz_norm: 相对坐标除以地图尺寸，范围 [-1, 1]
        - dist_norm: 距离除以地图对角线长度，范围 [0, 1]
    """
    dx, dz, dist = relative_pos(hero_pos, target_pos, map_size)
    
    dx_norm = float(np.clip(dx / map_size, -1.0, 1.0))
    dz_norm = float(np.clip(dz / map_size, -1.0, 1.0))
    dist_norm = float(dist / MAP_DIAG)
    
    return dx_norm, dz_norm, dist_norm


# ============================================================================
# 3. 动作方向映射函数
# ============================================================================

def action_to_delta(action_id: int) -> Tuple[float, float]:
    """
    根据动作 ID 获取对应的位移向量。
    
    Args:
        action_id: 动作 ID，范围 [0, 15]
          - 0-7: 移动动作
          - 8-15: 闪现动作
    
    Returns:
        (dx, dz): 位移向量
    
    Raises:
        ValueError: 如果动作 ID 无效
    
    Example:
        >>> action_to_delta(0)   # 右
        (1.0, 0.0)
        
        >>> action_to_delta(8)   # 右闪
        (10.0, 0.0)
    """
    action_id = int(action_id)
    if action_id not in ACTION_ID_TO_DELTA:
        raise ValueError(f"Invalid action_id: {action_id}, must be in [0, 15]")
    
    return ACTION_ID_TO_DELTA[action_id]


def action_to_normalized_delta(
    action_id: int,
    map_size: float = MAP_SIZE,
) -> Tuple[float, float]:
    """
    获取归一化的动作位移向量。
    
    Args:
        action_id: 动作 ID
        map_size: 地图尺寸
    
    Returns:
        (dx_norm, dz_norm): 归一化的位移向量
    """
    dx, dz = action_to_delta(action_id)
    dx_norm = dx / map_size
    dz_norm = dz / map_size
    return float(dx_norm), float(dz_norm)


def apply_action_to_pos(
    hero_pos: Dict[str, float],
    action_id: int,
    map_size: float = MAP_SIZE,
) -> Dict[str, float]:
    """
    计算执行某个动作后的英雄新位置。
    
    Args:
        hero_pos: 英雄当前位置
        action_id: 动作 ID
        map_size: 地图尺寸
    
    Returns:
        新位置 {"x": float, "z": float}，已 clip 到地图范围内
    """
    hero_x = float(hero_pos.get("x", 0.0))
    hero_z = float(hero_pos.get("z", 0.0))
    dx, dz = action_to_delta(action_id)
    
    new_x = float(np.clip(hero_x + dx, 0.0, map_size))
    new_z = float(np.clip(hero_z + dz, 0.0, map_size))
    
    return {"x": new_x, "z": new_z}


def action_is_movement(action_id: int) -> bool:
    """检查是否是移动动作（不是闪现）。"""
    return 0 <= int(action_id) < 8


def action_is_flash(action_id: int) -> bool:
    """检查是否是闪现动作。"""
    return 8 <= int(action_id) < 16


# ============================================================================
# 4. 局部地图方向解释函数
# ============================================================================

def get_direction_ahead_in_local_map(
    action_id: int,
    local_map: Optional[np.ndarray] = None,
    depth: int = 3,
) -> Dict[str, float]:
    """
    判断某个动作方向在局部地图中前方的情况。
    
    (注: 该函数支持将来扩展以支持实际的局部地图分析)
    
    Args:
        action_id: 动作 ID
        local_map: 局部地图 (21x21)，可选。如果提供，会分析实际障碍
        depth: 前方预测深度（步数）
    
    Returns:
        {
            "openness": float,  # 该方向的开阔度 [0, 1]
            "obstacle_risk": float,  # 遇到障碍的风险 [0, 1]
            "distance_before_obstacle": int,  # 预计几步后会遇障碍
        }
    """
    if action_is_movement(action_id):
        # 移动动作：通常更容易与地形互动
        base_openness = 0.7
    else:
        # 闪现动作：长距离移动，通常更难遇障碍
        base_openness = 0.9
    
    result = {
        "openness": base_openness,
        "obstacle_risk": 1.0 - base_openness,
        "distance_before_obstacle": depth if base_openness > 0.5 else 1,
    }
    
    # 如果提供了局部地图，可在此扩展实际分析逻辑
    if local_map is not None:
        # TODO: 根据局部地图和动作方向，计算更精确的开阔度
        pass
    
    return result


def get_local_map_sample_position(
    action_id: int,
    local_map_size: int = 21,
) -> Optional[Tuple[int, int]]:
    """
    获取某个动作方向在局部地图中对应的采样位置。
    
    Args:
        action_id: 动作 ID (仅支持 0-7 的 8 个方向)
        local_map_size: 局部地图尺寸，默认 21x21
    
    Returns:
        (row, col) 在局部地图中的索引，或 None (如果无对应）
    """
    if action_id < 0 or action_id >= 8:
        return None
    
    # 假设局部地图中心是 (10, 10)，边界是 (0-20, 0-20)
    if action_id in LOCAL_MAP_DIRECTION_SLOTS:
        row, col = LOCAL_MAP_DIRECTION_SLOTS[action_id]
        row = int(np.clip(row, 0, local_map_size - 1))
        col = int(np.clip(col, 0, local_map_size - 1))
        return (row, col)
    
    return None


# ============================================================================
# 5. 方向一致性检查函数
# ============================================================================

def consistency_check(verbose: bool = True) -> Dict[str, bool]:
    """
    运行一套完整的方向一致性检查。
    
    验证所有方向定义是否符合预期要求：
      - 右动作确实对应 x+
      - 下动作确实对应 z+
      - 相反动作确实是 180° 反向
      - 闪现距离确实比移动更远
      - 等等
    
    Args:
        verbose: 是否打印详细检查信息
    
    Returns:
        检查结果字典，所有值都应为 True
    
    Example:
        >>> results = consistency_check()
        >>> all(results.values())  # 应该返回 True
        True
    """
    results = {}
    
    # 检查 1: 右动作对应 x+
    dx_right, dz_right = action_to_delta(0)
    results["right_action_x_positive"] = dx_right > 0 and dz_right == 0
    
    # 检查 2: 下动作(6)对应 z+
    dx_down, dz_down = action_to_delta(6)
    results["down_action_z_positive"] = dx_down == 0 and dz_down > 0
    
    # 检查 3: 上动作(2)对应 z-
    dx_up, dz_up = action_to_delta(2)
    results["up_action_z_negative"] = dx_up == 0 and dz_up < 0
    
    # 检查 4: 左动作对应 x-
    dx_left, dz_left = action_to_delta(4)
    results["left_action_x_negative"] = dx_left < 0 and dz_left == 0
    
    # 检查 5: 相反动作确实相反
    all_opposite_correct = True
    for action_id, opposite_id in ACTION_ID_TO_OPPOSITE.items():
        dx1, dz1 = action_to_delta(action_id)
        dx2, dz2 = action_to_delta(opposite_id)
        if not (abs(dx1 + dx2) < 1e-3 and abs(dz1 + dz2) < 1e-3):
            all_opposite_correct = False
            break
    results["opposite_actions_correct"] = all_opposite_correct
    
    # 检查 6: 闪现距离 > 移动距离
    all_flash_longer = True
    for move_id in range(8):
        flash_id = move_id + 8
        dx_move, dz_move = action_to_delta(move_id)
        dx_flash, dz_flash = action_to_delta(flash_id)
        dist_move = np.sqrt(dx_move**2 + dz_move**2)
        dist_flash = np.sqrt(dx_flash**2 + dz_flash**2)
        if dist_flash <= dist_move:
            all_flash_longer = False
            break
    results["flash_distance_longer_than_move"] = all_flash_longer
    
    # 检查 7: 所有动作 ID 都有定义
    results["all_action_ids_defined"] = len(ACTION_ID_TO_DELTA) == 16
    
    # 检查 8: 所有相反动作对都有定义
    results["all_opposite_actions_defined"] = (
        len(ACTION_ID_TO_OPPOSITE) == 16
        and all(ACTION_ID_TO_OPPOSITE[ACTION_ID_TO_OPPOSITE[k]] == k for k in range(16))
    )
    
    if verbose:
        print("\n" + "="*70)
        print("SPATIAL COORDINATE SYSTEM CONSISTENCY CHECK")
        print("="*70)
        for check_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {check_name}")
        print("="*70 + "\n")
    
    return results


# ============================================================================
# 6. 工具函数
# ============================================================================

def get_action_name(action_id: int) -> str:
    """获取动作的人类可读名称。"""
    return ACTION_ID_TO_NAME.get(int(action_id), "unknown")


def get_opposite_action(action_id: int) -> int:
    """获取某个动作的相反动作。"""
    action_id = int(action_id)
    if action_id not in ACTION_ID_TO_OPPOSITE:
        raise ValueError(f"Invalid action_id: {action_id}")
    return ACTION_ID_TO_OPPOSITE[action_id]


def get_angular_distance_to_action(
    hero_pos: Dict[str, float],
    target_pos: Dict[str, float],
) -> Tuple[int, float]:
    """
    计算从英雄指向目标的最接近的动作方向。
    
    Args:
        hero_pos: 英雄位置
        target_pos: 目标位置
    
    Returns:
        (best_action_id, angle_difference): 最接近的动作 ID 和真实角度误差
    """
    dx, dz, _ = relative_pos(hero_pos, target_pos)
    target_angle = np.arctan2(dz, dx)  # z 是"下"方向
    
    # 尝试所有 8 个移动方向，找最接近的
    best_action_id = 0
    best_angle_diff = float('inf')
    
    for action_id in range(8):
        dx_action, dz_action = action_to_delta(action_id)
        action_angle = np.arctan2(dz_action, dx_action)
        angle_diff = abs(target_angle - action_angle)
        # 标准化角度差到 [0, pi]
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        
        if angle_diff < best_angle_diff:
            best_angle_diff = angle_diff
            best_action_id = action_id
    
    return best_action_id, float(best_angle_diff)


if __name__ == "__main__":
    # 运行一致性检查
    consistency_check(verbose=True)
    
    # 测试基本功能
    print("\n示例用法：")
    print(f"  normalize_global_pos(64, 64) = {normalize_global_pos(64, 64)}")
    print(f"  action_to_delta(0) = {action_to_delta(0)}")
    print(f"  get_action_name(0) = {get_action_name(0)}")
    print(f"  relative_pos_normalized({{'x': 50, 'z': 50}}, {{'x': 60, 'z': 50}}) = "
          f"{relative_pos_normalized({'x': 50, 'z': 50}, {'x': 60, 'z': 50})}")
