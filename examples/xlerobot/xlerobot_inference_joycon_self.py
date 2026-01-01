#!/usr/bin/env python3
"""
OpenPi Inference with Joycon Control for XLeRobot.
Integrates OpenPi inference capabilities with Joycon teleoperation for control and safety.
"""

import logging
import time
import traceback
import sys
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Lerobot imports
from lerobot.robots.xlerobot import XLerobotConfig, XLerobot
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.quadratic_spline_via_ipol import Via, Limits, QuadraticSplineInterpolator
from lerobot.teleoperators.xlerobot_joycon.xlerobot_joycon import (
    XLerobotJoyconTeleop, 
    XLerobotJoyconTeleopConfig, 
    init_joycon_listener
)

# Policy imports
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
FPS = 50
NUM_EPISODES = 50
EPISODE_TIME_SEC = 600
# Path to the local policy model (can be changed)
POLICY_PATH = os.environ.get("POLICY_PATH", "models/pi05_base")
# Ensure absolute path to avoid ambiguity with Hugging Face Hub IDs
POLICY_PATH = os.path.abspath(POLICY_PATH)

# Joint Mappings
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

FULL_START_POS = {
    "left_arm_shoulder_pan": 0.0,
    "left_arm_shoulder_lift": -90.0,
    "left_arm_elbow_flex": 45.0,
    "left_arm_wrist_flex": 85.0,
    "left_arm_wrist_roll": -90.0,
    "left_arm_gripper": 50.0,
    "right_arm_shoulder_pan": 0.0,
    "right_arm_shoulder_lift": -90.0,
    "right_arm_elbow_flex": 45.0,
    "right_arm_wrist_flex": 85.0,
    "right_arm_wrist_roll": -90.0,
    "right_arm_gripper": 50.0,
}

class SimpleArmControl:
    """
    A simplified arm control class for trajectory planning compatibility.
    """
    def __init__(self, joint_map, initial_obs, prefix="right", kp=0.75):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp

        # Set target positions to zero for P control
        self.target_positions = {k: FULL_START_POS.get(v, 0.0) for k, v in self.joint_map.items()}

        # Initial joint positions
        self.home_pos = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }

class JointIpol:
    """
    Quadratic Spline Interpolator wrapper for smooth joint control.
    Adapted from 9_pi05_inference_dualarm.py.
    """
    def __init__(self, kp=0.5, duration=3.0):
        self.ipol_path = None
        self.ipol_step = 0
        self.num_via_points = 0
        self.target_positions = None
        self.joint_names = FULL_START_POS.keys()
        self.kp = kp
        self.ctrl_freq = FPS
        self.duration = duration

    def plan_to_target(
            self, robot, left_arm, right_arm, ctrl_freq=FPS,
            target_positions=None, max_vel_per_joint=None, max_acc_per_joint=None, max_dev_per_joint=None):
            
            # 0) define target order explicitly via target_positions (canonical order)
            if target_positions is None:
                # Default to current targets if not specified, or home
                left_target_pos = {v: left_arm.target_positions.get(k, 0.0) for k, v in LEFT_JOINT_MAP.items()}
                right_target_pos = {v: right_arm.target_positions.get(k, 0.0) for k, v in RIGHT_JOINT_MAP.items()}
                target_positions = {**left_target_pos, **right_target_pos}
            self.target_positions = target_positions
            self.ctrl_freq = ctrl_freq
            
            # 1) Read current joint positions
            left_obs = robot.bus1.sync_read("Present_Position", robot.left_arm_motors)
            right_obs = robot.bus2.sync_read("Present_Position", robot.right_arm_motors)
            obs = {**left_obs, **right_obs}
            print(f"current pos: {obs}")

            # 2) build current/goal vectors in that SAME order (no sorting)
            q_now = []
            q_goal = []
            for n in self.joint_names:
                v = float(obs.get(n, 0.0))
                q_now.append(v)
                q_goal.append(float(target_positions.get(n, 0.0)))
            q_now  = np.array(q_now,  dtype=float)
            q_goal = np.array(q_goal, dtype=float)

            # 3) Limits (defaults if not provided)
            J = q_now.size
            if max_vel_per_joint is None:
                max_vel_per_joint = np.full(J, 15)   # deg/s
            if max_acc_per_joint is None:
                max_acc_per_joint = np.full(J, 30)   # deg/s^2
            if max_dev_per_joint is None:
                max_dev_per_joint = np.full(J, 0.0)

            # 4) Build a 2-via path (current -> goal)
            via = [
                Via(q=q_now,  max_dev=max_dev_per_joint),
                Via(q=q_goal, max_dev=max_dev_per_joint),
            ]
            lim = Limits(max_vel=np.asarray(max_vel_per_joint),
                        max_acc=np.asarray(max_acc_per_joint))

            ipol = QuadraticSplineInterpolator(via, lim)
            ipol.build()
            
            # 5) Slow-down scale so we finish exactly at 'duration'
            ipol.scale_to_duration(self.duration)

            # 6) Generate time samples and joint references at controller rate
            dt = 1.0/ float(self.ctrl_freq)
            t, q, qd, qdd = ipol.resample(dt)

            self.ipol_path = q.copy()
            self.num_via_points = len(t)
            self.ipol_step = 0

            # 7) Stream to the robot
            print(f"Streaming ipol trajectory: {len(t)} steps at {self.ctrl_freq:.1f} Hz; "
                f"planned duration ≈ {t[-1]:.3f}s (requested {self.duration:.3f}s)")
    
    def get_next_action(self, robot):
        if self.ipol_path is None:
            return {}
        
        left_obs = robot.bus1.sync_read("Present_Position", robot.left_arm_motors)
        right_obs = robot.bus2.sync_read("Present_Position", robot.right_arm_motors)
        obs = {**left_obs, **right_obs}

        q_meas = np.array([float(obs.get(n, 0.0)) for n in self.joint_names], dtype=float)
        q_ref = self.ipol_path[self.ipol_step, :]
        q_cmd = q_meas + self.kp*(q_ref - q_meas)
        action = {f"{j}.pos": q_cmd[i] for i, j in enumerate(self.joint_names)}

        self.ipol_step += 1

        obs_pos_suffix = {f"{k}.pos": v for k, v in obs.items()}
        return action, obs_pos_suffix
    
    def execute_plan(self, robot, left_arm, right_arm):
        t0 = time.perf_counter()
        next_tick = t0
        dt = 1.0/ float(self.ctrl_freq)
        while self.ipol_step < self.num_via_points:
            action, _ = self.get_next_action(robot)
            robot.send_action(action)

            # sleep to maintain control_freq
            next_tick += dt
            now = time.perf_counter()
            if now < next_tick:
                precise_sleep(next_tick - now)
        
        self.reset_ipol(left_arm, right_arm)
    
    def reset_ipol(self, left_arm, right_arm):
        # Update arm target positions to match the final goal of interpolation
        if self.target_positions:
            for k, v in LEFT_JOINT_MAP.items():
                if v in self.target_positions:
                    left_arm.target_positions[k] = self.target_positions[v]
            for k, v in RIGHT_JOINT_MAP.items():
                if v in self.target_positions:
                    right_arm.target_positions[k] = self.target_positions[v]
            print(f"ipol resets left arm target positions to: {left_arm.target_positions}")
            print(f"ipol resets right arm target positions to: {right_arm.target_positions}")

        self.ipol_path = None
        self.ipol_step = 0
        self.num_via_points = 0
        self.target_positions = None
        print("Reached target pos of full body with ipol trajectory.")


def main():
    print("XLerobot Local Policy Inference with Joycon Control")
    print("="*50)

    # 1. Initialize Robot
    robot = None
    robot_name = "xlerobot"
    try:
        robot_config = XLerobotConfig(id=robot_name, use_degrees=True)
        robot = XLerobot(robot_config)
        robot.connect()
        print(f"[INIT] Successfully connected to robot")
    except Exception as e:
        print(f"[INIT] Failed to connect to robot: {e}")
        traceback.print_exc()
        return

    # 2. Initialize Joycon Teleop (for control buttons)
    teleop = None
    try:
        joycon_config = XLerobotJoyconTeleopConfig()
        teleop = XLerobotJoyconTeleop(joycon_config)
        teleop.connect(robot=robot)
        print(f"[INIT] Joycon Teleop connected")
        listener, events = init_joycon_listener(teleop)
    except Exception as e:
        print(f"[INIT] Failed to connect Joycon: {e}")
        traceback.print_exc()
        if robot and robot.is_connected: robot.disconnect()
        return

    try:
        # 3. Move to Start Pose using JointIpol
        print("🔧 Moving robot to start pose...")
        obs = robot.get_observation()
        # Create temporary helper objects for JointIpol
        left_arm_ctrl = SimpleArmControl(LEFT_JOINT_MAP, obs, prefix="left")
        right_arm_ctrl = SimpleArmControl(RIGHT_JOINT_MAP, obs, prefix="right")

        joint_ipol = JointIpol()
        joint_ipol.plan_to_target(robot, left_arm_ctrl, right_arm_ctrl, ctrl_freq=200, target_positions=FULL_START_POS)
        joint_ipol.execute_plan(robot, left_arm_ctrl, right_arm_ctrl)
        print("✅ Robot in start pose")

        # 4. Load Local Policy
        policy = None
        preprocess = None
        postprocess = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            print(f"[INIT] Loading policy from {POLICY_PATH}...")
            # Check if path exists or if it's a huggingface ID
            # For local directory, ensure it exists
            if not os.path.exists(POLICY_PATH) and not "/" in POLICY_PATH: 
                 # Heuristic: if no slash, might be HF ID. If slash and not exists, warning.
                 pass
            
            # Load policy
            # Note: We use PI05Policy directly as requested/inferred from context.
            # If the model is different, change the class or use AutoPolicy if available.
            policy = PI05Policy.from_pretrained(POLICY_PATH)
            policy.to(device)
            policy.eval()
            
            # Create pre/post processors
            # This handles normalization and device transfer based on config
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                POLICY_PATH,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            print(f"[INIT] Policy loaded successfully on {device}")
            
        except Exception as e:
            print(f"[INIT] Failed to load policy: {e}")
            traceback.print_exc()
            if robot and robot.is_connected: robot.disconnect()
            return

        # 5. Inference Loop
        inference_episodes = 0
        print("Starting inference loop...")
        print("Controls: Left Joycon (-) to Exit, Right Joycon (+) to Re-record/Reset Episode")
        
        while inference_episodes < NUM_EPISODES:
            start_episode_t = time.perf_counter()
            timestamp = 0
            
            # Check events before episode start
            joycon_events = teleop.get_joycon_events()
            if joycon_events.get("exit_early"):
                print("Joycon requested exit.")
                break
                
            print(f"✅ Start episode: {inference_episodes}")
            
            while timestamp < EPISODE_TIME_SEC:
                # Update Joycon events
                joycon_events = teleop.get_joycon_events()
                
                # Exit condition
                if joycon_events.get("exit_early"):
                    print("Joycon requested exit during episode.")
                    break
                
                # Reset condition (Right Joycon Home or Plus usually)
                if joycon_events.get("reset_position") or joycon_events.get("rerecord_episode"):
                    print("Joycon requested reset/re-record.")
                    # Move to start pose again
                    joint_ipol.plan_to_target(robot, left_arm_ctrl, right_arm_ctrl, ctrl_freq=200, target_positions=FULL_START_POS)
                    joint_ipol.execute_plan(robot, left_arm_ctrl, right_arm_ctrl)
                    teleop.reset_joycon_events()
                    break # Break inner loop to restart episode

                # Get observation
                # Note: We need full observation for Policy
                
                left_obs = robot.bus1.sync_read("Present_Position", robot.left_arm_motors)
                right_obs = robot.bus2.sync_read("Present_Position", robot.right_arm_motors)
                joint_obs = {**left_obs, **right_obs}
                camera_obs = robot.get_camera_observation()
                
                # Construct observation dict for Policy
                # Map robot keys to policy keys.
                # Assuming policy expects: observation.images.head, observation.state
                # Adjust keys below if your policy uses different names.
                
                # 1. State: (B, D)
                state_array = np.array([float(joint_obs.get(n, 0.0)) for n in FULL_START_POS.keys()], dtype=np.float32)
                state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(device) # (1, D)
                
                # 2. Images: (B, C, H, W)
                # camera_obs images are (H, W, C) uint8 or similar.
                # We need to convert to (1, C, H, W) and potentially normalize via preprocess.
                # Note: build_inference_frame typically handles this, but here we do manual batching.
                
                observation_batch = {
                    "observation.state": state_tensor
                }
                
                # Add images
                # Mapping: 'image.head' -> 'observation.images.head'
                # Note: keys must match policy config. 
                for cam_key, img_data in camera_obs.items():
                    # img_data is (H, W, C)
                    # Convert to tensor (C, H, W)
                    img_tensor = torch.from_numpy(img_data).permute(2, 0, 1).float().unsqueeze(0).to(device)
                    # Normalize to [0, 1] if preprocess expects it (usually yes for image keys)
                    # However, preprocess usually handles rescaling if configured.
                    # Standard lerobot convention: images in obs dict are [0, 255] float or int? 
                    # preprocess usually expects float [0,1] OR handles from [0,255].
                    # Let's check if we should normalize to [0,1].
                    # build_inference_frame does: item / 255.0
                    img_tensor = img_tensor / 255.0
                    
                    # Map key
                    if "head" in cam_key:
                        observation_batch["observation.images.head"] = img_tensor
                    elif "left_wrist" in cam_key:
                        observation_batch["observation.images.left_wrist"] = img_tensor
                    elif "right_wrist" in cam_key:
                        observation_batch["observation.images.right_wrist"] = img_tensor
                    else:
                        # Fallback or include raw key
                        observation_batch[f"observation.images.{cam_key}"] = img_tensor

                # Preprocess
                if preprocess:
                    observation_batch = preprocess(observation_batch)

                # Get action from Policy
                with torch.no_grad():
                    action = policy.select_action(observation_batch)
                
                # Postprocess (unnormalize)
                if postprocess:
                    action = postprocess(action)

                # Execute action
                # action is (B, D) or (B, H, D). We take first step of first batch.
                # action tensor -> numpy
                action_np = action.squeeze(0).cpu().numpy()
                
                # If action has horizon (H, D), take first step
                if action_np.ndim > 1:
                    action_np = action_np[0]

                target_action = action_np
                    
                # Map array to dict
                action_dict = {}
                keys = list(FULL_START_POS.keys())
                for i, key in enumerate(keys):
                    if i < len(target_action):
                        action_dict[f"{key}.pos"] = target_action[i]
                
                robot.send_action(action_dict)
                
                # Timing
                start_loop_t = time.perf_counter()
                dt_s = time.perf_counter() - start_loop_t
                precise_sleep(max(1 / FPS - dt_s, 0.0))
                
                timestamp = time.perf_counter() - start_episode_t

            inference_episodes += 1
            
            # End of episode handling
            if joycon_events.get("exit_early"):
                break

    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        
    finally:
        # Cleanup
        if robot and robot.is_connected:
            print("Cleaning up...")
            # Try to move to home if possible
            # joint_ipol.plan_to_target(...)
            robot.disconnect()
        
        if teleop:
            teleop.disconnect()
            if 'listener' in locals() and listener:
                listener.stop()

if __name__ == "__main__":
    main()
