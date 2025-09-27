#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import asyncio
import logging
import os
import sys
import threading
import time
import traceback
from queue import Queue
from typing import Any, Dict, Optional

import numpy as np

from lerobot.model.SO101Robot import SO101Kinematics

from ..teleoperator import Teleoperator
from .configuration_xlerobot_vr import XLerobotVRTeleopConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 检查VR Monitor可用性
VR_AVAILABLE = True
try:
    # 动态导入VR Monitor 
    from .vr_monitor import VRMonitor
except ImportError as e:
    VR_AVAILABLE = False
    VRMonitor = None
    logging.warning(f"VR Monitor not available: {e}")
except Exception as e:
    VR_AVAILABLE = False
    VRMonitor = None
    logging.warning(f"Could not import VR Monitor: {e}")


# Joint mapping configurations (从8_xlerobot_VR_teleop.py复制)
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

# Joint calibration coefficients (从8_xlerobot_VR_teleop.py复制)
JOINT_CALIBRATION = [
    ['shoulder_pan', 6.0, 1.0],      
    ['shoulder_lift', 2.0, 0.97],     
    ['elbow_flex', 0.0, 1.05],        
    ['wrist_flex', 0.0, 0.94],        
    ['wrist_roll', 0.0, 0.5],        
    ['gripper', 0.0, 1.0],           
]

class SimpleTeleopArm:
    """
    A class for controlling a robot arm using VR input with delta action control.
    
    This class provides inverse kinematics-based arm control with proportional control
    for smooth movement and gripper operations based on VR controller input.
    """
    
    def __init__(self, joint_map, initial_obs, kinematics, prefix="right", kp=1):
        self.joint_map = joint_map
        self.prefix = prefix
        self.kp = kp
        self.kinematics = kinematics
        
        # Initial joint positions - adapted for XLerobot observation format
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
        
        # Delta control state variables for VR input
        self.last_vr_time = 0.0
        self.vr_deadzone = 0.001  # Minimum movement threshold
        self.max_delta_per_frame = 0.005  # Maximum position change per frame
        
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
        
        # Reset delta control state
        self.last_vr_time = 0.0
        
        # Explicitly set wrist_flex
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        return action

    def handle_vr_input(self, vr_goal, gripper_state):
        """
        Handle VR input with delta action control - incremental position updates.
        
        Args:
            vr_goal: VR controller goal data containing target position and orientations
            gripper_state: Current gripper state (not used in current implementation)
        """
        if vr_goal is None:
            return
        
        # VR goal contains: target_position [x, y, z], wrist_roll_deg, wrist_flex_deg, gripper_closed
        if not hasattr(vr_goal, 'target_position') or vr_goal.target_position is None:
            return
            
        # Extract VR position data
        # Get current VR position
        current_vr_pos = vr_goal.target_position  # [x, y, z] in meters
        
        # Initialize previous VR position if not set
        if not hasattr(self, 'prev_vr_pos'):
            self.prev_vr_pos = current_vr_pos
            return  # Skip first frame to establish baseline
        
        # print(current_vr_pos)
        
        # Calculate relative change (delta) from previous frame
        vr_x = (current_vr_pos[0] - self.prev_vr_pos[0]) * 220 # Scale for the shoulder
        vr_y = (current_vr_pos[1] - self.prev_vr_pos[1]) * 70 
        vr_z = (current_vr_pos[2] - self.prev_vr_pos[2]) * 70

        # print(f'vr_x: {vr_x}, vr_y: {vr_y}, vr_z: {vr_z}')

        # Update previous position for next frame
        self.prev_vr_pos = current_vr_pos
        
        # Delta control parameters - adjust these for sensitivity
        pos_scale = 0.01  # Position sensitivity scaling
        angle_scale = 3.0  # Angle sensitivity scaling
        delta_limit = 0.01  # Maximum delta per update (meters)
        angle_limit = 6.0  # Maximum angle delta per update (degrees)
        
        delta_x = vr_x * pos_scale
        delta_y = vr_y * pos_scale  
        delta_z = vr_z * pos_scale
        
        # Limit delta values to prevent sudden movements
        delta_x = max(-delta_limit, min(delta_limit, delta_x))
        delta_y = max(-delta_limit, min(delta_limit, delta_y))
        delta_z = max(-delta_limit, min(delta_limit, delta_z))
        
        self.current_x += -delta_z  # yy: VR Z maps to robot x, change the direction
        self.current_y += delta_y  # yy:VR Y maps to robot y

        # Handle wrist angles with delta control - use relative changes
        if hasattr(vr_goal, 'wrist_flex_deg') and vr_goal.wrist_flex_deg is not None:
            # Initialize previous wrist_flex if not set
            if not hasattr(self, 'prev_wrist_flex'):
                self.prev_wrist_flex = vr_goal.wrist_flex_deg
                return
            
            # Calculate relative change from previous frame
            delta_pitch = (vr_goal.wrist_flex_deg - self.prev_wrist_flex) * angle_scale
            delta_pitch = max(-angle_limit, min(angle_limit, delta_pitch))
            self.pitch += delta_pitch
            self.pitch = max(-90, min(90, self.pitch))  # Limit pitch range
            
            # Update previous value for next frame
            self.prev_wrist_flex = vr_goal.wrist_flex_deg
        
        if hasattr(vr_goal, 'wrist_roll_deg') and vr_goal.wrist_roll_deg is not None:
            # Initialize previous wrist_roll if not set
            if not hasattr(self, 'prev_wrist_roll'):
                self.prev_wrist_roll = vr_goal.wrist_roll_deg
                return
            
            delta_roll = (vr_goal.wrist_roll_deg - self.prev_wrist_roll) * angle_scale
            delta_roll = max(-angle_limit, min(angle_limit, delta_roll))
            
            current_roll = self.target_positions.get("wrist_roll", 0.0)
            new_roll = current_roll + delta_roll
            new_roll = max(-90, min(90, new_roll))  # Limit roll range
            self.target_positions["wrist_roll"] = new_roll
            
            # Update previous value for next frame
            self.prev_wrist_roll = vr_goal.wrist_roll_deg
        
        # VR Z axis controls shoulder_pan joint (delta control)
        if abs(delta_x) > 0.001:  # Only update if significant movement
            x_scale = 200.0  # Reduced scaling factor for delta control
            delta_pan = delta_x * x_scale
            delta_pan = max(-angle_limit, min(angle_limit, delta_pan))
            current_pan = self.target_positions.get("shoulder_pan", 0.0)
            new_pan = current_pan + delta_pan
            new_pan = max(-180, min(180, new_pan))  # Limit pan range
            self.target_positions["shoulder_pan"] = new_pan
        
        try:
            joint2_target, joint3_target = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            # Smooth transition to new joint positions,  Smoothing factor 0-1, lower = smoother
            alpha = 0.1
            self.target_positions["shoulder_lift"] = (1-alpha) * self.target_positions.get("shoulder_lift", 0.0) + alpha * joint2_target
            self.target_positions["elbow_flex"] = (1-alpha) * self.target_positions.get("elbow_flex", 0.0) + alpha * joint3_target
        except Exception as e:
            print(f"[{self.prefix}] VR IK failed: {e}")
        
        # Calculate wrist_flex to maintain end-effector orientation
        self.target_positions["wrist_flex"] = (-self.target_positions["shoulder_lift"] - 
                                               self.target_positions["elbow_flex"] + self.pitch)
   
        # Handle gripper state directly
        if vr_goal.metadata.get('trigger', 0) > 0.5:
            self.target_positions["gripper"] = 45
        else:
            self.target_positions["gripper"] = 0.0

    def p_control_action(self, robot_obs):
        """
        Generate proportional control action based on target positions.
        
        Args:
            robot: Robot instance to get current observations
            
        Returns:
            dict: Action dictionary with position commands for each joint
        """
        obs = robot_obs
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action


class SimpleHeadControl:
    """
    A class for controlling robot head motors using VR thumbstick input.
    
    Provides simple head movement control with proportional control for smooth operation.
    """
    
    def __init__(self, initial_obs, kp=1):
        self.kp = kp
        self.degree_step = 2  # Move 2 degrees each time
        # Initialize head motor positions
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}

    def handle_vr_input(self, vr_goal):
        # Map VR input to head motor targets
        thumb = vr_goal.metadata.get('thumbstick', {})
        if thumb:
            thumb_x = thumb.get('x', 0)
            thumb_y = thumb.get('y', 0)
            if abs(thumb_x) > 0.1:
                if thumb_x > 0:
                    self.target_positions["head_motor_1"] += self.degree_step
                else:
                    self.target_positions["head_motor_1"] -= self.degree_step
            if abs(thumb_y) > 0.1:
                if thumb_y > 0:
                    self.target_positions["head_motor_2"] += self.degree_step
                else:
                    self.target_positions["head_motor_2"] -= self.degree_step
                    
    def move_to_zero_position(self, robot_obs):
        print(f"[HEAD] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()
        action = self.p_control_action(robot_obs)
        return action
        

    def p_control_action(self, robot_obs):
        """
        Generate proportional control action for head motors.
        
        Args:
            robot: Robot instance to get current observations
            
        Returns:
            dict: Action dictionary with position commands for head motors
        """
        obs = robot_obs
        action = {}
        for motor in self.target_positions:
            current = obs.get(f"{HEAD_MOTOR_MAP[motor]}.pos", 0.0)
            error = self.target_positions[motor] - current
            control = self.kp * error
            action[f"{HEAD_MOTOR_MAP[motor]}.pos"] = current + control
        return action


def get_vr_base_action(vr_goal, robot):
    """
    Get base control commands from VR input.
    
    Args:
        vr_goal: VR controller goal data containing metadata
        robot: Robot instance for action conversion
        
    Returns:
        dict: Base movement actions based on VR thumbstick input
    """
    pressed_keys = set()
    if vr_goal is not None and hasattr(vr_goal, 'metadata'):
    
    # Build key set based on VR input (you can customize this mapping)
    
    # Example VR to base movement mapping - adjust according to your VR system
    # You may need to customize these mappings based on your VR controller buttons
        thumb = vr_goal.metadata.get('thumbstick', {})
        if thumb:
            thumb_x = thumb.get('x', 0)
            thumb_y = thumb.get('y', 0)
            if abs(thumb_x) > 0.2:
                if thumb_x > 0:
                    pressed_keys.add('o')  # Move backward
                else:
                    pressed_keys.add('u')  # Move forward
            if abs(thumb_y) > 0.2:
                if thumb_y > 0:
                    pressed_keys.add('k')  # Move right
                else:
                    pressed_keys.add('i')  # Move backward
    
    # Convert to numpy array and get base action
    keyboard_keys = np.array(list(pressed_keys))
    base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}
    
    return base_action

class XLerobotVRTeleop(Teleoperator):
    """
    XLerobot VR Teleoperator类
    按照teleop_keyboard的格式，集成8_xlerobot_VR_teleop.py中的VR控制逻辑
    """

    config_class = XLerobotVRTeleopConfig
    name = "xlerobot_vr"

    def __init__(self, config: XLerobotVRTeleopConfig):
        super().__init__(config)
        self.config = config
        
        # VR系统相关
        self.vr_monitor = None
        self.vr_thread = None
        self.vr_data_queue = Queue()
        self.latest_vr_data = None
        
        # 新增：VR事件处理器
        self.vr_event_handler = None
                    
        # 运动学实例
        self.kin_left = SO101Kinematics()
        self.kin_right = SO101Kinematics()
        
        # 基座速度控制
        self.current_base_speed = 0.0
        self.last_update_time = time.time()
        self.is_accelerating = False
        
        # 状态标志
        self._connected = False
        self._calibrated = False
        
        self.logs = {}
        self.last_event_update_time = 0  # Initialize event update time

    @property
    def action_features(self) -> dict:
        """定义动作特征结构"""
        # 根据XLerobot的动作空间定义
        # 包括双臂关节、头部电机、基座移动
        features = {}
        
        # 左臂关节
        for joint_name in LEFT_JOINT_MAP.values():
            features[f"{joint_name}.pos"] = "float32"
        
        # 右臂关节
        for joint_name in RIGHT_JOINT_MAP.values():
            features[f"{joint_name}.pos"] = "float32"
            
        # 头部电机
        for motor_name in HEAD_MOTOR_MAP.values():
            features[f"{motor_name}.pos"] = "float32"
            
        # 基座控制（根据XLerobot的基座控制方式）
        features["base_action"] = "dict"
        
        return features

    @property
    def feedback_features(self) -> dict:
        """定义反馈特征结构"""
        return {}  # VR控制器通常不需要反馈

    @property
    def is_connected(self) -> bool:
        """检查连接状态"""
        return (
            self._connected and 
            VR_AVAILABLE and 
            self.vr_monitor is not None and
            (self.vr_thread is not None and self.vr_thread.is_alive())
        )

    @property
    def is_calibrated(self) -> bool:
        """检查校准状态"""
        return self._calibrated

    def connect(self, calibrate: bool = True, robot=None) -> None:
        """建立VR连接 - 优化版本"""
        if self.is_connected:
            raise RuntimeError(
                "XLerobot VR is already connected. Do not run `connect()` twice."
            )

        if not VR_AVAILABLE:
            raise RuntimeError(
                "VR Monitor is not available. Please check VR system installation."
            )

        try:
            logger.info("🔧 Initializing VR monitor...")
            self.vr_monitor = VRMonitor()
            
            # 使用超时机制避免无限等待
            init_success = False
            start_time = time.time()
            timeout = 10.0  # 10秒超时
            
            while time.time() - start_time < timeout:
                if self.vr_monitor.initialize():
                    init_success = True
                    break
                time.sleep(0.1)
            
            if not init_success:
                raise Exception("VR monitor initialization timeout")
                
            logger.info("🚀 Starting VR monitoring...")
            self.vr_thread = threading.Thread(
                target=lambda: asyncio.run(self.vr_monitor.start_monitoring()), 
                daemon=True
            )
            self.vr_thread.start()
            
            # 等待线程启动
            time.sleep(0.5)
            
            if not self.vr_thread.is_alive():
                raise Exception("VR monitoring thread failed to start")
                
            logger.info("✅ VR system ready")
            self._connected = True
            
            # 初始化VR事件处理器
            self.vr_event_handler = VREventHandler(self.vr_monitor)
            logger.info("🎮 VR事件处理器已初始化")
            
            if calibrate and robot is not None:
                robot_obs = robot.get_observation(use_camera=False)
                self.calibrate(robot_obs)
                
        except Exception as e:
            logger.error(f"[VR] Connection failed: {e}")
            self._connected = False
            raise RuntimeError(f"Failed to connect to VR: {e}")

    def calibrate(self, robot_obs: Optional[Dict] = None) -> None:
        """校准VR控制器 - 优化版本"""
        if robot_obs is None:
            logger.warning("[VR] No robot observation provided for calibration")
            return
            
        try:
            # 初始化机械臂控制器
            self.left_arm = SimpleTeleopArm(
                LEFT_JOINT_MAP, robot_obs, self.kin_left, 
                prefix="left", kp=self.config.kp
            )
            self.right_arm = SimpleTeleopArm(
                RIGHT_JOINT_MAP, robot_obs, self.kin_right, 
                prefix="right", kp=self.config.kp
            )
            
            # 初始化头部控制器
            self.head_control = SimpleHeadControl(robot_obs, kp=self.config.kp)
            
            logger.info("[VR] Controllers initialized successfully")
            self._calibrated = True
            
        except Exception as e:
            logger.error(f"[VR] Calibration failed: {e}")
            self._calibrated = False
            raise


    def get_action(self, robot_obs: Optional[Dict] = None, robot = None) -> dict[str, Any]:
        """获取VR控制动作 - 高性能优化版本，并行处理事件和动作"""
        before_read_t = time.perf_counter()
        
        action = {}
        
        # 快速检查VR监控状态
        if not self.vr_monitor:
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            return action
        
        # 一次性获取VR数据，避免重复调用
        try:
            dual_goals = self.vr_monitor.get_latest_goal_nowait()
            if dual_goals is None:
                self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
                return action
                
            left_goal = dual_goals.get("left")
            right_goal = dual_goals.get("right")
            
        except Exception as e:
            logger.warning(f"VR数据获取失败: {e}")
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            return action
        
        # 并行处理：机器人控制高频，事件处理低频
        if robot_obs is not None:
            try:
                current_time = time.perf_counter()
                
                # 机器人控制 - 高频执行（60Hz）
                if left_goal is not None:
                    self.left_arm.handle_vr_input(left_goal, None)
                    
                if right_goal is not None:
                    self.right_arm.handle_vr_input(right_goal, None)
                
                # 事件处理 - 低频执行（10Hz），只在间隔时间到达时处理
                if (current_time - self.last_event_update_time) >= 2:
                    if left_goal is not None:
                        self._update_events_inline(left_goal)
                    self.last_event_update_time = current_time
                
                # 快速生成动作字典
                left_action = self.left_arm.p_control_action(robot_obs)
                right_action = self.right_arm.p_control_action(robot_obs)
                head_action = self.head_control.p_control_action(robot_obs)
                base_action = get_vr_base_action(right_goal, robot)
                
                # 高效合并动作
                action.update(left_action)
                action.update(right_action)
                action.update(head_action)
                action.update(base_action)
                
            except Exception as e:
                logger.error(f"动作生成失败: {e}")
            
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
        return action
    
    def _update_events_inline(self, left_goal):
        """
        低频事件更新 - 10Hz频率，复用已获取的left_goal数据
        只在事件间隔时间到达时执行，大幅减少处理开销
        """
        if not self.vr_event_handler or not left_goal or not hasattr(left_goal, 'metadata'):
            return
            
        # 直接使用已获取的数据，无需重新调用VR接口
        try:
            self.vr_event_handler._process_left_controller(left_goal.metadata)
        except Exception as e:
            logger.debug(f"事件低频更新失败: {e}")  # 降级为debug避免干扰主流程

    def send_feedback(self) -> None:
        """发送反馈 - 优化版本，减少阻塞等待"""
        if not self.vr_monitor:
            logger.warning("VR monitor not available for feedback")
            return

        max_attempts = 200  # 最多尝试100次
        attempt = 0
        
        while attempt < max_attempts:
            try:
                dual_goals = self.vr_monitor.get_latest_goal_nowait()
                if dual_goals and sum(dual_goals.get('right').metadata['vr_position']):
                    logger.info("VR controller data received")
                    return
                    
            except Exception as e:
                logger.warning(f"Error getting VR data: {e}")
            
            attempt += 1
            logger.info(f'Waiting for VR controller data (attempt {attempt}/{max_attempts})')
            time.sleep(0.5)  # 减少等待时间从8秒到0.5秒
        
        logger.warning("Timeout waiting for VR controller data")

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        """断开VR连接"""
        if not self.is_connected:
            raise RuntimeError(
                "XLerobot VR is not connected."
            )
        
        try:
            if self.vr_monitor:
                # VR Monitor通常在线程中运行，停止线程
                pass
            
            self._connected = False
            self._calibrated = False
            print("[VR] Disconnected")
            
        except Exception as e:
            print(f"[VR] Error during disconnect: {e}")

    def move_to_zero_position(self, robot):
        """移动所有控制器到零位"""
        robot_obs = robot.get_observation(use_camera=False)
        action = {}
        left_action = self.left_arm.move_to_zero_position(robot_obs)
        right_action = self.right_arm.move_to_zero_position(robot_obs)
        head_action = self.head_control.move_to_zero_position(robot_obs)
        base_action = get_vr_base_action(None, robot)
        action.update(left_action)
        action.update(right_action)
        action.update(head_action)
        action.update(base_action)

        return action
    
    def get_vr_events(self):
        """获取VR事件状态（高性能版本 - 使用缓存，避免重复VR数据获取）"""
        if self.vr_event_handler:
            # 获取当前事件状态
            events = self.vr_event_handler.get_events()
            
            # 自动重置一次性事件，防止死循环
            # 只有在事件为True时才重置，避免影响正常状态
            if events.get("exit_early", False) or events.get("rerecord_episode", False):
                self.vr_event_handler.reset_events()
            
            return events
        else:
            # 返回默认事件状态
            return {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
                "reset_position": False,
            }
    
    def reset_vr_events(self):
        """重置VR事件状态"""
        if self.vr_event_handler:
            self.vr_event_handler.reset_events()
    
    def print_vr_control_guide(self):
        """打印VR控制指南"""
        if self.vr_event_handler:
            self.vr_event_handler.print_control_guide()
        else:
            logger.info("VR事件处理器未初始化")


def init_vr_listener(teleop_vr):
    """
    初始化VR监听器，提供与init_keyboard_listener相同的接口
    用于替代键盘事件监听，在record.py中使用
    
    Args:
        teleop_vr: XLerobotVRTeleop实例
        
    Returns:
        tuple: (listener, events) - 与init_keyboard_listener相同的返回格式
    """
    if not isinstance(teleop_vr, XLerobotVRTeleop):
        logger.error("teleop_vr必须是XLerobotVRTeleop实例")
        return None, {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "reset_position": False,
        }
    
    # 打印控制指南
    teleop_vr.print_vr_control_guide()
    
    # 创建虚拟listener对象（与keyboard listener兼容）
    class VRListener:
        def __init__(self, teleop_vr):
            self.teleop_vr = teleop_vr
            self.is_alive = True
            
        def stop(self):
            self.is_alive = False
            logger.info("VR监听器已停止")
    
    vr_listener = VRListener(teleop_vr)
    
    # 获取初始事件状态
    events = teleop_vr.get_vr_events()
    
    return vr_listener, events

class VREventHandler:
    """
    VR事件处理器，专门处理录制控制事件
    使用左侧VR手柄替代键盘控制
    """
    
    def __init__(self, vr_monitor):
        self.vr_monitor = vr_monitor
        self.events = {
            "exit_early": False,      # 左手柄→右: 提前退出循环 (原右箭头键)
            "rerecord_episode": False, # 左手柄→左: 重新录制episode (原左箭头键)
            "stop_recording": False,   # 左手柄→上: 停止录制 (原ESC键)
            "reset_position": False,   # 左手柄→下: 复位机器人 (新增功能)
        }
        self.prev_states = {
            'thumbstick_x': 0,
            'thumbstick_y': 0,
            'trigger': False,
            'button_a': False,
            'button_b': False,
        }
        self.threshold = 0.7  # 摇杆触发阈值
        
    def update_events(self):
        """更新VR事件状态"""
        if not self.vr_monitor:
            return self.events
            
        try:
            dual_goals = self.vr_monitor.get_latest_goal_nowait()
            if not dual_goals:
                return self.events
                
            left_goal = dual_goals.get("left")
            if not left_goal or not hasattr(left_goal, 'metadata'):
                return self.events
                
            self._process_left_controller(left_goal.metadata)
            
        except Exception as e:
            logger.error(f"VR事件更新失败: {e}")
            
        return self.events
    
    def _process_left_controller(self, metadata):
        """处理左手柄输入"""
        # 获取摇杆输入
        thumb = metadata.get('thumbstick', {})
        thumb_x = thumb.get('x', 0)
        thumb_y = thumb.get('y', 0)

        
        # 检测摇杆方向事件（只在跨越阈值时触发）
        if thumb_x > self.threshold and self.prev_states['thumbstick_x'] <= self.threshold:
            logger.info("🎮 VR左手柄向右 -> 提前退出循环")
            self.events["exit_early"] = True
            
        elif thumb_x < -self.threshold and self.prev_states['thumbstick_x'] >= -self.threshold:
            logger.info("🎮 VR左手柄向左 -> 重新录制episode")
            self.events["rerecord_episode"] = True
            self.events["exit_early"] = True
            
        if thumb_y > self.threshold and self.prev_states['thumbstick_y'] <= self.threshold:
            logger.info("🎮 VR左手柄 -> 停止录制")
            self.events["stop_recording"] = True
            self.events["exit_early"] = True
            
        elif thumb_y < -self.threshold and self.prev_states['thumbstick_y'] >= -self.threshold:
            logger.info("🎮 VR左手柄向shang -> 复位机器人")
            self.events["reset_position"] = True
        else:
            self.events["reset_position"] = False  # 复位事件为瞬时事件
        
        # 检测扳机键事件
        trigger = metadata.get('trigger', 0) > 0.5
        
        # 更新状态
        self.prev_states.update({
            'thumbstick_x': thumb_x,
            'thumbstick_y': thumb_y,
            'trigger': trigger,
        })
    
    def reset_events(self):
        """重置所有事件状态"""
        for key in self.events:
            self.events[key] = False
    
    def get_events(self):
        """获取当前事件状态"""
        return self.events.copy()
    
    def print_control_guide(self):
        """打印VR控制指南"""
        guide = """
        🎮 VR左手柄控制指南:
        ├── 👈 向左推摇杆: 重新录制当前episode
        ├── 👉 向右推摇杆: 提前退出当前循环  
        ├── 👆 向上推摇杆: 停止录制
        ├── 👇 向下推摇杆: 复位机器人位置
        """
        logger.info(guide)