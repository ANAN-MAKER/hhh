#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Action Simulator: 规则一致的动作后果仿真器

与环境规则完全对齐的动作模拟，包括：
- 普通移动（含障碍检测、对角线检验）
- 闪现（内寻最远可达）
- 路径对象收集（宝箱/buff）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# 常数
MAP_SIZE = 128.0
MAP_DIAG = float(np.sqrt(MAP_SIZE**2 + MAP_SIZE**2))

# 动作方向映射
ACTION_DELTA = {
    0: (1.0, 0.0),      # 右
    1: (1.0, -1.0),     # 右上
    2: (0.0, -1.0),     # 上
    3: (-1.0, -1.0),    # 左上
    4: (-1.0, 0.0),     # 左
    5: (-1.0, 1.0),     # 左下
    6: (0.0, 1.0),      # 下
    7: (1.0, 1.0),      # 右下
    8: (10.0, 0.0),     # 右闪
    9: (8.0, -8.0),     # 右上闪
    10: (0.0, -10.0),   # 上闪
    11: (-8.0, -8.0),   # 左上闪
    12: (-10.0, 0.0),   # 左闪
    13: (-8.0, 8.0),    # 左下闪
    14: (0.0, 10.0),    # 下闪
    15: (8.0, 8.0),     # 右下闪
}


def is_in_bounds(pos: Dict[str, float], map_size: float = MAP_SIZE) -> bool:
    """检查位置是否在地图范围内。"""
    x = pos.get("x", 0)
    z = pos.get("z", 0)
    return 0 <= x < map_size and 0 <= z < map_size


def world_to_local_map(world_pos: Dict[str, float], hero_pos: Dict[str, float]) -> Optional[Tuple[int, int]]:
    """
    将世界坐标转换为局部地图坐标。
    
    局部地图是21x21，中心是(10,10)对应hero_pos。
    
    Args:
        world_pos: {"x": float, "z": float}
        hero_pos: {"x": float, "z": float}
        
    Returns:
        (row, col) 在local_map中的索引，or None if out of local map range
    """
    dx = world_pos["x"] - hero_pos["x"]
    dz = world_pos["z"] - hero_pos["z"]
    
    # 局部地图中心在(10, 10)，范围[-10, 11)
    row = int(10 + dz)  # z向下为正，对应row
    col = int(10 + dx)  # x向右为正，对应col
    
    if 0 <= row < 21 and 0 <= col < 21:
        return (row, col)
    return None


def local_map_to_world(local_pos: Tuple[int, int], hero_pos: Dict[str, float]) -> Dict[str, float]:
    """将局部地图坐标转换回世界坐标。"""
    row, col = local_pos
    dz = row - 10  # row 10对应z=hero_pos.z
    dx = col - 10  # col 10对应x=hero_pos.x
    return {"x": hero_pos["x"] + dx, "z": hero_pos["z"] + dz}


def get_local_map_cell(local_map: Any, row: int, col: int) -> float:
    """
    安全地读取local_map的某个格子。
    
    Handles both list and numpy array.
    返回该格子的值，超出范围时返回1.0（认为可通行）。
    """
    if local_map is None:
        return 1.0
    
    try:
        if isinstance(local_map, list):
            if 0 <= row < len(local_map) and 0 <= col < len(local_map[0]):
                return float(local_map[row][col])
        else:  # numpy array
            if 0 <= row < local_map.shape[0] and 0 <= col < local_map.shape[1]:
                return float(local_map[row, col])
    except Exception as e:
        logger.warning(f"Error reading local_map[{row}, {col}]: {e}")
    
    return 1.0  # 超出范围视为可通行


def is_cell_passable(local_map: Any, row: int, col: int, threshold: float = 0.7) -> bool:
    """检查某个格子是否可通行。"""
    cell_val = get_local_map_cell(local_map, row, col)
    return cell_val >= threshold


def is_diag_move_legal(
    from_world: Dict[str, float],
    to_world: Dict[str, float],
    local_map: Any,
    hero_pos: Dict[str, float],
) -> bool:
    """
    检查对角线移动是否合法。
    
    规则：目标格可通行 AND (水平相邻可通行 OR 竖直相邻可通行)
    """
    # 目标格必须可通行
    to_local = world_to_local_map(to_world, hero_pos)
    if to_local is None:
        return False
    to_row, to_col = to_local
    if not is_cell_passable(local_map, to_row, to_col):
        return False
    
    # 检查水平相邻
    horiz_world = {"x": to_world["x"], "z": from_world["z"]}
    horiz_local = world_to_local_map(horiz_world, hero_pos)
    horiz_ok = horiz_local is not None and is_cell_passable(local_map, horiz_local[0], horiz_local[1])
    
    # 检查竖直相邻
    vert_world = {"x": from_world["x"], "z": to_world["z"]}
    vert_local = world_to_local_map(vert_world, hero_pos)
    vert_ok = vert_local is not None and is_cell_passable(local_map, vert_local[0], vert_local[1])
    
    return horiz_ok or vert_ok


def simulate_move(
    hero_pos: Dict[str, float],
    action_id: int,
    local_map: Any,
    buff_active: bool = False,
    obstacles_only: bool = False,
) -> Dict[str, Any]:
    """
    模拟普通移动动作（action 0-7）。
    
    规则：
    - 基础速度：1格/步（buff时2格/步）
    - 对角线移动：目标可通行 AND (水平OR竖直相邻可通行)
    - 碰撞处理：速度为1则原地；速度>1则逐格检测，停在最后可达格
    
    Args:
        hero_pos: {"x": float, "z": float}
        action_id: 0-7 (must be movement)
        local_map: 21x21 local map (list or numpy array)
        buff_active: 是否有buff加速
        obstacles_only: 若True，只检测障碍；若False，还检测对角线合法性
        
    Returns:
        {
            "next_pos": {"x": float, "z": float},
            "movement_distance": float,
            "path_cells": [(row, col), ...] in local coordinates,
            "blocked": bool,
            "stay_same_cell": bool,
        }
    """
    
    if action_id >= 8:
        logger.warning(f"simulate_move called with flash action {action_id}")
        action_id = min(action_id, 7)  # Fallback
    
    dx, dz = ACTION_DELTA.get(action_id % 8, (0, 0))  # 归一化到8个方向
    
    # 速度：buff时2格，否则1格
    base_speed = 2.0 if buff_active else 1.0
    
    # 判断是否对角线移动
    is_diag = abs(dx) > 0 and abs(dz) > 0
    
    # 目标位置（直进）
    target_x = hero_pos["x"] + dx * base_speed
    target_z = hero_pos["z"] + dz * base_speed
    target_pos = {"x": target_x, "z": target_z}
    
    # 边界检查
    if not is_in_bounds(target_pos):
        return {
            "next_pos": hero_pos,
            "movement_distance": 0.0,
            "path_cells": [],
            "blocked": True,
            "stay_same_cell": True,
        }
    
    # 对角线移动的合法性检查
    if is_diag and not obstacles_only:
        if not is_diag_move_legal(hero_pos, target_pos, local_map, hero_pos):
            return {
                "next_pos": hero_pos,
                "movement_distance": 0.0,
                "path_cells": [],
                "blocked": True,
                "stay_same_cell": True,
            }
    
    # 如果速度为1，直接检查目标格
    if base_speed == 1.0:
        target_local = world_to_local_map(target_pos, hero_pos)
        if target_local and is_cell_passable(local_map, target_local[0], target_local[1]):
            return {
                "next_pos": target_pos,
                "movement_distance": 1.0,
                "path_cells": [target_local],
                "blocked": False,
                "stay_same_cell": False,
            }
        else:
            return {
                "next_pos": hero_pos,
                "movement_distance": 0.0,
                "path_cells": [],
                "blocked": True,
                "stay_same_cell": True,
            }
    
    # 速度 > 1时，逐格检测
    last_valid_pos = hero_pos
    path_cells = []
    for step in range(1, int(base_speed) + 1):
        intermediate_pos = {
            "x": hero_pos["x"] + dx * step,
            "z": hero_pos["z"] + dz * step,
        }
        
        if not is_in_bounds(intermediate_pos):
            break
        
        # 对角线检查（每一步都要检查）
        if is_diag and not obstacles_only:
            if not is_diag_move_legal(last_valid_pos, intermediate_pos, local_map, hero_pos):
                break
        
        intermediate_local = world_to_local_map(intermediate_pos, hero_pos)
        if intermediate_local and is_cell_passable(local_map, intermediate_local[0], intermediate_local[1]):
            last_valid_pos = intermediate_pos
            path_cells.append(intermediate_local)
        else:
            break
    
    movement_dist = (
        np.sqrt((last_valid_pos["x"] - hero_pos["x"])**2 + (last_valid_pos["z"] - hero_pos["z"])**2)
        if last_valid_pos != hero_pos else 0.0
    )
    
    return {
        "next_pos": last_valid_pos,
        "movement_distance": float(movement_dist),
        "path_cells": path_cells,
        "blocked": last_valid_pos == hero_pos,
        "stay_same_cell": last_valid_pos == hero_pos,
    }


def raycast_farthest_reachable(
    start_world: Dict[str, float],
    direction: Tuple[float, float],
    max_distance: float,
    local_map: Any,
    hero_pos: Dict[str, float],
) -> Tuple[Dict[str, float], float, List[Tuple[int, int]]]:
    """
    沿方向光线追踪，找最远可达位置。
    
    用于闪现等需要找最远可达格的场景。
    
    Args:
        start_world: {"x": float, "z": float}
        direction: (dx, dz) normalized or not
        max_distance: 最大搜索距离
        local_map: 局部地图
        hero_pos: 英雄位置（用于local_map转换）
        
    Returns:
        (farthest_pos, distance_to_farthest, path_cells)
    """
    
    # 归一化方向向量
    dx, dz = direction
    mag = np.sqrt(dx**2 + dz**2)
    if mag < 1e-6:
        return start_world, 0.0, []
    
    dx_norm = dx / mag
    dz_norm = dz / mag
    
    # 逐步搜索，每次前进0.5格（细粒度搜索）
    farthest_pos = start_world
    farthest_distance = 0.0
    path_cells = []
    
    step_size = 0.5
    for step in np.arange(step_size, max_distance + step_size, step_size):
        test_x = start_world["x"] + dx_norm * step
        test_z = start_world["z"] + dz_norm * step
        test_pos = {"x": test_x, "z": test_z}
        
        if not is_in_bounds(test_pos):
            break
        
        test_local = world_to_local_map(test_pos, hero_pos)
        if test_local and is_cell_passable(local_map, test_local[0], test_local[1]):
            farthest_pos = test_pos
            farthest_distance = step
            if test_local not in path_cells:
                path_cells.append(test_local)
        else:
            break
    
    return farthest_pos, float(farthest_distance), path_cells


def simulate_flash(
    hero_pos: Dict[str, float],
    action_id: int,
    local_map: Any,
) -> Dict[str, Any]:
    """
    模拟闪现动作（action 8-15）。
    
    规则：
    - 直线闪现：10格
    - 斜向闪现：8格
    - 找范围内最远可通行格
    - 若都不可通行则原地闪现但仍消耗冷却
    - 路径上的宝箱/buff会被收集
    
    Args:
        hero_pos: {"x": float, "z": float}
        action_id: 8-15 (must be flash)
        local_map: 局部地图
        
    Returns:
        {
            "flash_pos": {"x": float, "z": float},
            "flash_distance": float,
            "reachable": bool,
            "path_cells": [(row, col), ...],
        }
    """
    
    if action_id < 8:
        logger.warning(f"simulate_flash called with movement action {action_id}")
        action_id = min(action_id + 8, 15)  # Fallback
    
    # 获取闪现方向
    dx, dz = ACTION_DELTA.get(action_id, (0, 0))
    
    # 闪现距离：直线10格，斜线8格
    # 判断是否为直线方向（只有x或只有z非零）
    is_linear = (abs(dx) < 0.1 and abs(dz) > 0.1) or (abs(dx) > 0.1 and abs(dz) < 0.1)
    max_flash_dist = 10.0 if is_linear else 8.0
    
    # 光线追踪找最远可达点
    direction = (dx, dz) if (abs(dx) > 0.1 or abs(dz) > 0.1) else (1.0, 0.0)
    flash_pos, flash_distance, path_cells = raycast_farthest_reachable(
        hero_pos, direction, max_flash_dist, local_map, hero_pos
    )
    
    reachable = flash_distance > 0.1  # 至少前进0.1格算可达
    
    return {
        "flash_pos": flash_pos,
        "flash_distance": float(flash_distance),
        "reachable": reachable,
        "path_cells": path_cells,
    }


def simulate_action(
    hero_pos: Dict[str, float],
    action_id: int,
    local_map: Any,
    buff_active: bool = False,
    treasures: Optional[List[Dict]] = None,
    buffs: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    统一的动作仿真器入口。
    
    Args:
        hero_pos: 英雄位置
        action_id: 0-15
        local_map: 局部地图
        buff_active: 是否有buff加速
        treasures: [{"x": float, "z": float}, ...]
        buffs: [{"x": float, "z": float}, ...]
        
    Returns:
        完整的动作后果信息
    """
    
    is_flash = action_id >= 8
    
    if is_flash:
        result = simulate_flash(hero_pos, action_id, local_map)
    else:
        result = simulate_move(hero_pos, action_id, local_map, buff_active)
    
    # 计算动作后的状态变化
    next_pos = result.get("flash_pos") if is_flash else result.get("next_pos", hero_pos)
    
    # 路径上收集的宝箱/buff
    collected_treasures = 0
    collected_buffs = 0
    if treasures or buffs:
        path_cells = result.get("path_cells", [])
        for row, col in path_cells:
            cell_world = local_map_to_world((row, col), hero_pos)
            
            # 检查宝箱
            if treasures:
                for t in treasures:
                    if abs(t["x"] - cell_world["x"]) < 0.5 and abs(t["z"] - cell_world["z"]) < 0.5:
                        collected_treasures += 1
            
            # 检查buff
            if buffs:
                for b in buffs:
                    if abs(b["x"] - cell_world["x"]) < 0.5 and abs(b["z"] - cell_world["z"]) < 0.5:
                        collected_buffs += 1
    
    result["collected_treasures"] = collected_treasures
    result["collected_buffs"] = collected_buffs
    result["is_flash"] = is_flash
    
    return result
