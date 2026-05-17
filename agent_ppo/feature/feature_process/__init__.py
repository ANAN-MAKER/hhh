#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from __future__ import annotations

from typing import Any, Dict

from agent_ppo.feature.feature_process.battlefield_process import BattlefieldFeatureProcessor


class FeatureProcess:
    def __init__(self, camp: str) -> None:
        self.reset(camp)

    def reset(self, camp: str) -> None:
        self.camp = camp
        # Active runtime feature pipeline.
        self.processor = BattlefieldFeatureProcessor(camp)

    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        return self.processor.build(observation)

    def process_feature(self, observation: Dict[str, Any]) -> list:
        return self.process_observation(observation)["feature_vector"]
