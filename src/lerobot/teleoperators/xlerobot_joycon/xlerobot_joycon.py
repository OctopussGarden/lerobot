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

import logging
import time
import math
from typing import Any, Dict, Optional

import numpy as np

from lerobot.model.SO101Robot import SO101Kinematics
from ..teleoperator import Teleoperator
from .configuration_xlerobot_joycon import XLerobotJoyconTeleopConfig

logger = logging.getLogger(__name__)

JOYCON_AVAILABLE = True
try:
    from joyconrobotics import JoyconRobotics
except Exception as e:
    JOYCON_AVAILABLE = False
    JoyconRobotics = None
    logger.warning(f"joyconrobotics not available: {e}")

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}
HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

class FixedAxesJoyconRobotics(JoyconRobotics):
    """
    Modified JoyconRobotics class to control fixed axes with joystick input.
    The center values for the joystick sticks are set differently for left and right Joy-Cons.
    The gripper control is added to handle open/close actions based on button presses.
    """
    def __init__(self, device: str, config: XLerobotJoyconTeleopConfig, **kwargs):
        """
        Initialize Joy-Con wrapper with teleop configuration for stick centers and gripper settings.
        """
        self._config = config

        if device == "right":
            self.joycon_stick_v_0 = config.right_stick_v_center
            self.joycon_stick_h_0 = config.right_stick_h_center
        else:
            self.joycon_stick_v_0 = config.left_stick_v_center
            self.joycon_stick_h_0 = config.left_stick_h_center

        self.gripper_speed = config.gripper_speed
        self.gripper_direction = 1
        self.gripper_min = config.gripper_min
        self.gripper_max = config.gripper_max
        self.last_gripper_button_state = 0

        super().__init__(device, **kwargs)

        try:
            if self.joycon.is_right():
                self.joycon_stick_v_0 = config.right_stick_v_center
                self.joycon_stick_h_0 = config.right_stick_h_center
            else:
                self.joycon_stick_v_0 = config.left_stick_v_center
                self.joycon_stick_h_0 = config.left_stick_h_center
        except Exception:
            pass
    
    def common_update(self):
        # Modified update logic: joystick only controls fixed axes with deadzone
        speed_scale = 0.001
        deadzone = 300
        stick_range = 1000
        
        # Get current orientation data to compute pitch
        orientation_rad = self.get_orientation()
        roll, pitch, yaw = orientation_rad
 
        # Vertical joystick: controls X and Z axes (forward/backward)
        joycon_stick_v = self.joycon.get_stick_right_vertical() if self.joycon.is_right() else self.joycon.get_stick_left_vertical()
        dv = joycon_stick_v - self.joycon_stick_v_0
        if abs(dv) > deadzone:
            norm = dv / stick_range
            self.position[0] += speed_scale * norm * self.dof_speed[0] * self.direction_reverse[0] * math.cos(pitch)
            self.position[2] += speed_scale * norm * self.dof_speed[2] * self.direction_reverse[2] * math.sin(pitch)
        
        # Horizontal joystick: only controls Y axis (left/right)
        joycon_stick_h = self.joycon.get_stick_right_horizontal() if self.joycon.is_right() else self.joycon.get_stick_left_horizontal()
        dh = joycon_stick_h - self.joycon_stick_h_0
        if abs(dh) > deadzone:
            norm = dh / stick_range
            self.position[1] += speed_scale * norm * self.dof_speed[1] * self.direction_reverse[1]
        
        # Z-axis control restored using buttons as per reference implementation
        # For left joycon: Up/Down buttons
        # For right joycon: X/B buttons (but these are used for base on right joycon, so we only enable on left joycon or use different buttons)
        
        # Reference implementation uses R/L buttons for UP and R-Stick/L-Stick buttons for DOWN
        # We need to be careful not to conflict with other mappings
        
        # Enable Z-axis control for both controllers using shoulder buttons or stick buttons if available
        # Left Joycon: Up/Down buttons are used for Z-axis in reference implementation (L/R shoulder buttons)
        joycon_button_up = self.joycon.get_button_r() if self.joycon.is_right() else self.joycon.get_button_l()
        if joycon_button_up == 1:
            self.position[2] += speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        joycon_button_down = self.joycon.get_button_r_stick() if self.joycon.is_right() else self.joycon.get_button_l_stick()
        if joycon_button_down == 1:
            self.position[2] -= speed_scale * self.dof_speed[2] * self.direction_reverse[2]
        
        # Home button reset logic (simplified version)
        joycon_button_home = self.joycon.get_button_home() if self.joycon.is_right() else self.joycon.get_button_capture()
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()
        
        # Gripper control logic (hold for linear increase/decrease mode)
        for event_type, status in self.button.events():
            if (self.joycon.is_right() and event_type == 'plus' and status == 1) or (self.joycon.is_left() and event_type == 'minus' and status == 1):
                self.reset_button = 1
                self.reset_joycon()
            elif self.joycon.is_right() and event_type == 'a':
                self.next_episode_button = status
            elif self.joycon.is_right() and event_type == 'y':
                self.restart_episode_button = status
            else: 
                self.reset_button = 0
        
        # Gripper button state detection and direction control
        gripper_button_pressed = False
        if self.joycon.is_right():
            # Right Joy-Con uses ZR button
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zr() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_r_btn() == 1
        else:
            # Left Joy-Con uses ZL button
            if not self.change_down_to_gripper:
                gripper_button_pressed = self.joycon.get_button_zl() == 1
            else:
                gripper_button_pressed = self.joycon.get_button_stick_l_btn() == 1
        
        # Detect button press events (from 0 to 1) to change direction
        if gripper_button_pressed and self.last_gripper_button_state == 0:
            # Button just pressed, change direction
            self.gripper_direction *= -1
            logger.debug(f"[GRIPPER] Direction changed to: {'Open' if self.gripper_direction == 1 else 'Close'}")
        
        # Update button state record
        self.last_gripper_button_state = gripper_button_pressed
        
        # Linear control of gripper open/close when holding gripper button
        if gripper_button_pressed:
            # Check if exceeding limits
            new_gripper_state = self.gripper_state + self.gripper_direction * self.gripper_speed
            
            # If exceeding limits, stop moving
            if new_gripper_state >= self.gripper_min and new_gripper_state <= self.gripper_max:
                self.gripper_state = new_gripper_state
            # If exceeding limits, stay at current position, don't change direction

        

        # Button control state
        if self.joycon.is_right():
            if self.next_episode_button == 1:
                self.button_control = 1
            elif self.restart_episode_button == 1:
                self.button_control = -1
            elif self.reset_button == 1:
                self.button_control = 8
            else:
                self.button_control = 0
        
        return self.position, self.gripper_state, self.button_control
class SimpleTeleopArm:
    def __init__(self, joint_map, initial_obs, kinematics, prefix="right", kp=1):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.kinematics = kinematics
        
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Set step size
        self.degree_step = 2
        self.xy_step = 0.005
        
        # P control target positions, set to zero position
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        
        # Reset kinematics variables to initial state
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Explicitly set wrist_flex
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon_pose, gripper_state):
        """Handle Joy-Con input, update arm control - based on 6_so100_joycon_ee_control.py"""
        x, y, z, roll_, pitch_, yaw = joycon_pose
        
        # Calculate pitch control - consistent with 6_so100_joycon_ee_control.py
        pitch = -pitch_ * 60 + 10
        
        # Set coordinates - consistent with 6_so100_joycon_ee_control.py
        current_x = 0.1629 + x
        current_y = 0.1131 + z
        
        # Calculate roll - consistent with 6_so100_joycon_ee_control.py
        roll = roll_ * 45
        
        print(f"[{self.prefix}] pitch: {pitch}")
        
        # Add y value to control shoulder_pan joint - consistent with 6_so100_joycon_ee_control.py
        y_scale = 250.0  # Scaling factor, can be adjusted as needed
        self.target_positions["shoulder_pan"] = y * y_scale
        
        # Use inverse kinematics to calculate joint angles - consistent with 6_so100_joycon_ee_control.py
        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(current_x, current_y)
            self.target_positions["shoulder_lift"] = joint2_target
            self.target_positions["elbow_flex"] = joint3_target
        except Exception as e:
            print(f"[{self.prefix}] IK failed: {e}")
        
        # Set wrist_flex - consistent with 6_so100_joycon_ee_control.py
        self.target_positions["wrist_flex"] = -self.target_positions["shoulder_lift"] - self.target_positions["elbow_flex"] + pitch
        
        # Set wrist_roll - consistent with 6_so100_joycon_ee_control.py
        self.target_positions["wrist_roll"] = roll
        
        # Gripper control - now set directly in main loop, no need to handle here
        pass

    def p_control_action(self, robot):
        obs = robot.get_observation() if hasattr(robot, "get_observation") else robot
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action

class SimpleHeadControl:
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.degree_step = 2  # Move 2 degrees each time
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def move_to_zero_position(self, robot):
        print("[HEAD] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot)
        robot.send_action(action)

    def handle_joycon_input(self, joycon):
        """Handle left Joy-Con directional pad input to control head motors"""
        # Get left Joy-Con directional pad state
        button_up = joycon.joycon.get_button_up()      # Up: head_motor_1+
        button_down = joycon.joycon.get_button_down()  # Down: head_motor_1-
        button_left = joycon.joycon.get_button_left()  # Left: head_motor_2+
        button_right = joycon.joycon.get_button_right() # Right: head_motor_2-
        
        if button_up == 1:
            self.target_positions["head_motor_2"] += self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if button_down == 1:
            self.target_positions["head_motor_2"] -= self.degree_step
            print(f"[HEAD] head_motor_2: {self.target_positions['head_motor_2']}")
        if button_left == 1:
            self.target_positions["head_motor_1"] += self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")
        if button_right == 1:
            self.target_positions["head_motor_1"] -= self.degree_step
            print(f"[HEAD] head_motor_1: {self.target_positions['head_motor_1']}")

    def p_control_action(self, robot):
        obs = robot.get_observation() if hasattr(robot, "get_observation") else robot
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action

def get_joycon_base_action(joycon, robot):
    """
    Get base control commands from Joy-Con
    X: forward, B: backward, Y: left turn, A: right turn
    """
    # Get button states
    button_x = joycon.joycon.get_button_x()  # forward
    button_b = joycon.joycon.get_button_b()  # backward
    button_y = joycon.joycon.get_button_y()  # left turn
    button_a = joycon.joycon.get_button_a()  # right turn
    
    # Build key set (simulate keyboard input)
    pressed_keys = set()
    
    if button_x == 1:
        pressed_keys.add('k')  # forward
        print("[BASE] Forward")
    if button_b == 1:
        pressed_keys.add('i')  # backward
        print("[BASE] Backward")
    if button_y == 1:
        pressed_keys.add('u')  # left turn
        print("[BASE] Left turn")
    if button_a == 1:
        pressed_keys.add('o')  # right turn
        print("[BASE] Right turn")
    
    # Convert to numpy array and get base action
    keyboard_keys = np.array(list(pressed_keys))
    base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}
    
    return base_action

# Base speed control parameters - adjustable slopes
BASE_ACCELERATION_RATE = 2.0  # acceleration slope (speed/second)
BASE_DECELERATION_RATE = 2.5  # deceleration slope (speed/second)
BASE_MAX_SPEED = 3.0          # maximum speed multiplier

def get_joycon_speed_control(joycon):
    """
    Get speed control from Joy-Con - linear acceleration and deceleration
    Linearly accelerate to maximum speed when holding any base control button, linearly decelerate to 0 when released
    """
    global current_base_speed, last_update_time, is_accelerating
    
    # Initialize global variables
    if 'current_base_speed' not in globals():
        current_base_speed = 0.0
        last_update_time = time.time()
        is_accelerating = False
    
    current_time = time.time()
    dt = current_time - last_update_time
    last_update_time = current_time
    
    # Check if any base control buttons are pressed
    button_x = joycon.joycon.get_button_x()  # forward
    button_b = joycon.joycon.get_button_b()  # backward
    button_y = joycon.joycon.get_button_y()  # left turn
    button_a = joycon.joycon.get_button_a()  # right turn
    
    any_base_button_pressed = any([button_x, button_b, button_y, button_a])
    
    if any_base_button_pressed:
        # Button pressed - accelerate
        if not is_accelerating:
            is_accelerating = True
            print("[BASE] Starting acceleration")
        
        # Linear acceleration
        current_base_speed += BASE_ACCELERATION_RATE * dt
        current_base_speed = min(current_base_speed, BASE_MAX_SPEED)
        
    else:
        # No button pressed - decelerate
        if is_accelerating:
            is_accelerating = False
            print("[BASE] Starting deceleration")
        
        # Linear deceleration
        current_base_speed -= BASE_DECELERATION_RATE * dt
        current_base_speed = max(current_base_speed, 0.0)
    
    # Print current speed (optional, for debugging)
    if abs(current_base_speed) > 0.01:  # Only print when speed is not 0
        print(f"[BASE] Current speed: {current_base_speed:.2f}")
    
    return current_base_speed



class JoyconEventHandler:
    """
    Event handler implementing strict Joy-Con mapping:
    - Left Minus (-): exit current episode loop early (LEFT controller)
    - Right Plus (+): re-record current episode
    - Right Home: reset robot to zero position (RIGHT controller)
    Debounced with prev-state tracking.
    """
    def __init__(self, teleop: "XLerobotJoyconTeleop"):
        self.teleop = teleop
        self.events = {
            "exit_early": False, # button minus
            "rerecord_episode": False, # button plus
            "stop_recording": False, # button capture
            "reset_position": False, # button home
            "back_position": False, # putton l && button r, In the bucket
        }
        self.prev = {"left_minus": 0, "right_plus": 0, "right_home": 0, "capture": 0, "left_l": 0, "right_r": 0}

    def update(self):
        left = self.teleop.left_joycon
        right = self.teleop.right_joycon
        try:
            if left:
                minus = left.joycon.get_button_minus()
                if minus == 1 and self.prev["left_minus"] == 0:
                    self.events["exit_early"] = True
                self.prev["left_minus"] = minus
                
                # Screenshot button on Left Joy-Con (capture button)
                capture = left.joycon.get_button_capture()
                if capture == 1 and self.prev["capture"] == 0:
                    self.events["stop_recording"] = True
                    logger.info("Recording triggered!")
                else:
                    self.events["stop_recording"] = False
                self.prev["capture"] = capture

            if right:
                plus = right.joycon.get_button_plus()
                if plus == 1 and self.prev["right_plus"] == 0:
                    self.events["rerecord_episode"] = True
                self.prev["right_plus"] = plus
                
                home = right.joycon.get_button_home()
                if home == 1 and self.prev["right_home"] == 0:
                    self.events["reset_position"] = True
                else:
                    self.events["reset_position"] = False
                self.prev["right_home"] = home
                
                # Check for reset combo: Minus (Left) + Plus (Right) + Home (Right)
                if left and right:
                    left_l = left.joycon.get_button_l()
                    right_r = right.joycon.get_button_r()
                    if (left_l == 1 and self.prev["left_l"] == 0) and \
                       (right_r == 1 and self.prev["right_r"] == 0):
                        self.events["back_position"] = True
                        logger.info("Back to bucket position!")
                    else:
                        self.events["back_position"] = False
                    self.prev["left_l"] = left_l
                    self.prev["right_r"] = right_r

        except Exception as e:
            logger.debug(f"Joy-Con event update failed: {e}")
        return self.events.copy()

    def reset_events(self):
        for k in self.events:
            self.events[k] = False

    def print_control_guide(self):
        guide = """
        Joy-Con Mapping:
        Left Minus (-): Exit current episode loop early
        Right Plus (+): Re-record current episode
        Right Home: Reset robot to zero position
        """
        logger.info(guide)


class XLerobotJoyconTeleop(Teleoperator):
    """
    Joy-Con teleoperator for XLerobot following Teleoperator interface.
    Provides real-time mapping from Joy-Con inputs to robot actions while keeping
    Joy-Con-specific logic isolated from core teleoperation methods.
    """
    config_class = XLerobotJoyconTeleopConfig
    name = "xlerobot_joycon"

    def __init__(self, config: XLerobotJoyconTeleopConfig):
        """
        Initialize the teleoperator with the provided configuration.
        """
        self.config = config
        super().__init__(config)
        # self.config = config
        self.left_joycon: FixedAxesJoyconRobotics | None = None
        self.right_joycon: FixedAxesJoyconRobotics | None = None
        self.kin_left = SO101Kinematics()
        self.kin_right = SO101Kinematics()
        self.current_base_speed = 0.0
        self.last_update_time = time.time()
        self.is_accelerating = False
        self._connected = False
        self._calibrated = False
        self.left_arm: SimpleTeleopArm | None = None
        self.right_arm: SimpleTeleopArm | None = None
        self.head_control: SimpleHeadControl | None = None
        self.event_handler: JoyconEventHandler | None = None
        self.logs: Dict[str, Any] = {}

    @property
    def action_features(self) -> dict:
        """
        Describe the action dictionary produced by this teleoperator.
        """
        features = {}
        for joint_name in LEFT_JOINT_MAP.values():
            features[f"{joint_name}.pos"] = "float32"
        for joint_name in RIGHT_JOINT_MAP.values():
            features[f"{joint_name}.pos"] = "float32"
        for motor_name in HEAD_MOTOR_MAP.values():
            features[f"{motor_name}.pos"] = "float32"
        features["base_action"] = "dict"
        return features

    @property
    def feedback_features(self) -> dict:
        """
        Feedback features are not used for Joy-Con teleop.
        """
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected and JOYCON_AVAILABLE and (self.left_joycon or self.right_joycon)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def connect(self, calibrate: bool = True, robot=None) -> None:
        """
        Connect to Joy-Con controllers and optionally calibrate using the robot observation.
        """
        if self.is_connected:
            raise RuntimeError("Joy-Con teleoperator already connected")
        if not JOYCON_AVAILABLE:
            raise RuntimeError("joyconrobotics is not available")
        try:
            self.right_joycon = FixedAxesJoyconRobotics(
                "right",
                config=self.config,
                dof_speed=self.config.dof_speed,
                lerobot=True,
                without_rest_init=False,
                change_down_to_gripper=self.config.change_down_to_gripper,
            )
        except Exception as e:
            self.right_joycon = None
            raise RuntimeError(f"Failed to connect right Joy-Con: {e}")
        try:
            self.left_joycon = FixedAxesJoyconRobotics(
                "left",
                config=self.config,
                dof_speed=self.config.dof_speed,
                lerobot=True,
                without_rest_init=False,
                change_down_to_gripper=self.config.change_down_to_gripper,
            )
        except Exception as e:
            logger.warning(f"Left Joy-Con not connected: {e}")
            self.left_joycon = None
        self._connected = True
        self.event_handler = JoyconEventHandler(self)
        if calibrate and robot is not None:
            robot_obs = robot.get_observation(use_camera=False)
            self.calibrate(robot_obs)

    def calibrate(self, robot_obs: Optional[Dict] = None) -> None:
        """
        Initialize arm and head control targets from a robot observation snapshot.
        """
        if robot_obs is None:
            return
        self.left_arm = SimpleTeleopArm(LEFT_JOINT_MAP, robot_obs, self.kin_left, prefix="left", kp=self.config.kp)
        self.right_arm = SimpleTeleopArm(RIGHT_JOINT_MAP, robot_obs, self.kin_right, prefix="right", kp=self.config.kp)
        self.head_control = SimpleHeadControl(robot_obs, kp=self.config.kp)
        self._calibrated = True

    def _speed_multiplier(self, joycon: JoyconRobotics | None) -> float:
        if joycon is None:
            return 0.0
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        button_x = joycon.joycon.get_button_x()
        button_b = joycon.joycon.get_button_b()
        button_y = joycon.joycon.get_button_y() if hasattr(joycon.joycon, "get_button_y") else 0
        button_a = joycon.joycon.get_button_a() if hasattr(joycon.joycon, "get_button_a") else 0
        any_base = any([button_x, button_b, button_y, button_a])
        if any_base:
            if not self.is_accelerating:
                self.is_accelerating = True
            self.current_base_speed += self.config.base_acceleration_rate * dt
            self.current_base_speed = min(self.current_base_speed, self.config.base_max_speed)
        else:
            if self.is_accelerating:
                self.is_accelerating = False
            self.current_base_speed -= self.config.base_deceleration_rate * dt
            self.current_base_speed = max(self.current_base_speed, 0.0)
        return self.current_base_speed

    def _map_gripper_target(self, gripper_value: Any) -> float | None:
        if gripper_value is None:
            return None
        try:
            v = float(gripper_value)
        except Exception:
            return None

        if 0.0 <= v <= 1.0 and self.config.gripper_max > 1.0 and self.config.gripper_min == 0.0:
            v = v * float(self.config.gripper_max)

        return float(min(max(v, float(self.config.gripper_min)), float(self.config.gripper_max)))

    def get_action(self, robot_obs: Optional[Dict] = None, robot=None) -> dict[str, Any]:
        """
        Read Joy-Con inputs and produce a flat action dict compatible with `XLerobot.send_action`.
        """
        before = time.perf_counter()
        action = {}
        if not self._connected:
            self.logs["read_pos_dt_s"] = time.perf_counter() - before
            return action
        if (self.left_arm is None or self.right_arm is None or self.head_control is None) and robot_obs is not None:
            try:
                self.calibrate(robot_obs)
            except Exception:
                self.logs["read_pos_dt_s"] = time.perf_counter() - before
                return action
        if self.left_arm is None or self.right_arm is None or self.head_control is None:
            self.logs["read_pos_dt_s"] = time.perf_counter() - before
            return action
        try:
            left_pose, left_gripper, _ = (None, None, None)
            right_pose, right_gripper, _ = (None, None, None)
            if self.left_joycon:
                left_pose, left_gripper, _ = self.left_joycon.get_control()
            if self.right_joycon:
                right_pose, right_gripper, _ = self.right_joycon.get_control()
        except Exception as e:
            logger.warning(f"Joy-Con read failed: {e}")
            self._try_reconnect()
            self.logs["read_pos_dt_s"] = time.perf_counter() - before
            return action
        events = self.event_handler.update() if self.event_handler else {"exit_early": False, "rerecord_episode": False, "reset_position": False}
        if robot_obs is not None:
            if events.get("reset_position") and robot is not None:
                # Build zero-position action without side effects
                left_action = {f"{LEFT_JOINT_MAP[j]}.pos": 0.0 for j in LEFT_JOINT_MAP}
                right_action = {f"{RIGHT_JOINT_MAP[j]}.pos": 0.0 for j in RIGHT_JOINT_MAP}
                head_action = {f"{HEAD_MOTOR_MAP[m]}.pos": 0.0 for m in HEAD_MOTOR_MAP}
                base_action = {}
            else:
                if left_pose is not None:
                    # Ignore head D-pad inputs for arm control if needed, but stick is primary
                    # Here we just pass pose to arm handler
                    self.left_arm.handle_joycon_input(left_pose, left_gripper)
                    target_gripper = self._map_gripper_target(left_gripper)
                    if target_gripper is not None:
                        self.left_arm.target_positions["gripper"] = target_gripper

                if right_pose is not None:
                    self.right_arm.handle_joycon_input(right_pose, right_gripper)
                    target_gripper = self._map_gripper_target(right_gripper)
                    if target_gripper is not None:
                        self.right_arm.target_positions["gripper"] = target_gripper

                left_action = self.left_arm.p_control_action(robot_obs)
                right_action = self.right_arm.p_control_action(robot_obs)
                if self.left_joycon:
                    self.head_control.handle_joycon_input(self.left_joycon)
                head_action = self.head_control.p_control_action(robot_obs)
                base_action = get_joycon_base_action(self.right_joycon, robot)
                speed = self._speed_multiplier(self.right_joycon)
                if base_action:
                    for k in base_action:
                        if "vel" in k or "velocity" in k:
                            base_action[k] *= speed
            action.update(left_action)
            action.update(right_action)
            action.update(head_action)
            action.update(base_action or {})
        self.logs["read_pos_dt_s"] = time.perf_counter() - before
        return action

    def send_feedback(self, feedback: dict[str, Any] | None = None) -> None:
        """
        No-op feedback hook; kept for API consistency.
        """
        if not self._connected:
            return
        attempts = 20
        for _ in range(attempts):
            try:
                if self.right_joycon:
                    pose, _, _ = self.right_joycon.get_control()
                    if pose is not None:
                        return
            except Exception:
                pass
            time.sleep(0.5)

    def configure(self) -> None:
        """
        Apply runtime configuration hooks if needed.
        """
        return

    def disconnect(self) -> None:
        """
        Disconnect Joy-Con controllers and cleanup state.
        """
        if not self.is_connected:
            raise RuntimeError("XLerobot Joy-Con is not connected.")
        try:
            if self.right_joycon:
                self.right_joycon.disconnect()
            if self.left_joycon:
                self.left_joycon.disconnect()
            self._connected = False
            self._calibrated = False
        except Exception as e:
            logger.error(f"Joy-Con disconnect error: {e}")

    def move_to_zero_position(self, robot) -> dict[str, float]:
        """
        Build and return an action dict that moves arms and head to zero positions.
        """
        return {
            **{f"{LEFT_JOINT_MAP[j]}.pos": 0.0 for j in LEFT_JOINT_MAP},
            **{f"{RIGHT_JOINT_MAP[j]}.pos": 0.0 for j in RIGHT_JOINT_MAP},
            **{f"{HEAD_MOTOR_MAP[m]}.pos": 0.0 for m in HEAD_MOTOR_MAP},
        }

    def get_joycon_events(self):
        """
        Return current Joy-Con teleop events with debouncing.
        """
        if self.event_handler:
            ev = self.event_handler.update()
            if ev.get("exit_early") or ev.get("rerecord_episode"):
                self.event_handler.reset_events()
            return ev
        return {"exit_early": False, "rerecord_episode": False, "stop_recording": False, "reset_position": False, "back_position": False}

    def reset_joycon_events(self):
        """
        Reset event flags to defaults.
        """
        if self.event_handler:
            self.event_handler.reset_events()

    def print_joycon_control_guide(self):
        """
        Log Joy-Con controls for reference.
        """
        if not self.event_handler:
            logger.info("Joy-Con event handler not initialized")
            return
        guide = """
        🎮 Joy-Con Recording Controls (STRICT MAPPING):
        
        Left Joy-Con:
        ├── Joystick: Left arm position (XYZ via stick + tilt)
        ├── ZL button: Toggle Left Gripper (Open/Close)
        ├── D-pad: Head control (faster & smooth)
        │   ├── Up/Down: Head motor 2
        │   └── Left/Right: Head motor 1
        └── Minus (-) button: Exit current loop early (next episode)

        Right Joy-Con:
        ├── Joystick: Right arm position (XYZ via stick + tilt)
        ├── ZR button: Toggle Right Gripper (Open/Close)
        ├── X/B/Y/A: Base movement (Forward/Back/Turn)
        ├── Plus (+) button: Re-record current episode
        └── Home button: Reset robot to zero position
        
        ⚠️  SAFETY & LIMITS:
        - Arm movement is RESTRICTED to Joysticks only. 
        - Buttons do NOT control arm position (Z-axis buttons disabled).
        - Gripper toggles on button press; visual feedback in logs.
        - Head movement speed increased 1.5x with smooth acceleration.
        """
        logger.info(guide)

    def _try_reconnect(self) -> None:
        """
        Attempt to reconnect Joy-Con controllers when reads fail.
        """
        attempts = max(1, int(self.config.joycon_reconnect_attempts))
        for _ in range(attempts):
            try:
                if self.right_joycon is None:
                    self.right_joycon = FixedAxesJoyconRobotics(
                        "right",
                        config=self.config,
                        dof_speed=self.config.dof_speed,
                        lerobot=True,
                        without_rest_init=False,
                        change_down_to_gripper=self.config.change_down_to_gripper,
                    )
                if self.left_joycon is None:
                    self.left_joycon = FixedAxesJoyconRobotics(
                        "left",
                        config=self.config,
                        dof_speed=self.config.dof_speed,
                        lerobot=True,
                        without_rest_init=False,
                        change_down_to_gripper=self.config.change_down_to_gripper,
                    )
                return
            except Exception as e:
                logger.debug(f"Joy-Con reconnect attempt failed: {e}")


def init_joycon_listener(teleop_joycon: XLerobotJoyconTeleop):
    if not isinstance(teleop_joycon, XLerobotJoyconTeleop):
        return None, {"exit_early": False, "rerecord_episode": False, "stop_recording": False, "reset_position": False, "back_position": False}  # pyright: ignore[reportUnreachable]
    teleop_joycon.print_joycon_control_guide()
    class JoyconListener:
        def __init__(self, teleop_joycon):
            self.teleop_joycon = teleop_joycon
            self.is_alive = True
        def stop(self):
            self.is_alive = False
    listener = JoyconListener(teleop_joycon)
    events = teleop_joycon.get_joycon_events()
    return listener, events
