#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import List

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xlerobot_joycon")
@dataclass
class XLerobotJoyconTeleopConfig(TeleoperatorConfig):
    """
    Configuration for Joy-Con teleoperation of XLerobot.
    """
    # Joystick center positions
    right_stick_v_center: int = 1900
    right_stick_h_center: int = 2100
    left_stick_v_center: int = 2300
    left_stick_h_center: int = 2000
    # Proportional gain for degree of freedom control
    kp: float = 0.5
    # Degree of freedom speed
    dof_speed: List[float] = None
    # Base control parameters
    base_acceleration_rate: float = 2.0
    base_deceleration_rate: float = 2.5
    base_max_speed: float = 3.0
    # Gripper control parameters
    gripper_speed: float = 0.3
    gripper_min: float = 0.0
    gripper_max: float = 90.0
    change_down_to_gripper: bool = False
    # Joy-Con control parameters
    joycon_reconnect_attempts: int = 3
    joycon_connection_timeout_s: float = 10.0
    
    # Head control parameters
    head_degree_step: float = 1.0
    head_acceleration: float = 1.0
    head_deceleration: float = 1.0
    head_max_speed: float = 1.0

    # Joystick stick threshold and range
    stick_threshold: int = 300
    stick_range: int = 1000

    def __post_init__(self):
        if self.dof_speed is None:
            self.dof_speed = [2, 2, 2, 1, 1, 1]
