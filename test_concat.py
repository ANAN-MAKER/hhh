#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
简单的 numpy 拼接测试
"""

import numpy as np

# 模拟 9 个部分
parts = [
    np.ones(24),   # self_dim (24D)
    np.ones(7),    # monster1_dim (7D)
    np.ones(7),    # monster2_dim (7D)
    np.ones(12),   # treasure_dim (12D)
    np.ones(8),    # buff_dim (8D)
    np.ones(20),   # progress_dim (20D)
    np.ones(35),   # map_dim (35D)
    np.ones(30),   # plan_dim (30D)
    np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),  # legal_dim (16D)
]

print("各部分的长度:")
for i, part in enumerate(parts):
    print(f"  Part {i+1}: {len(part)}D")

total = sum(len(part) for part in parts)
print(f"总长度: {total}D")

feature = np.concatenate(parts)
print(f"拼接后特征长度: {len(feature)}D")

# 检查各部分是否正确拼接
offsets = np.cumsum([0] + [len(part) for part in parts])
print()
print("拼接后的各部分验证:")
for i, (part, start, end) in enumerate(zip(parts, offsets[:-1], offsets[1:])):
    feat_slice = feature[start:end]
    print(f"  Part {i+1}: [{start:3d}-{end:3d}] {end-start:2d}D, sum={np.sum(feat_slice):.1f}")
