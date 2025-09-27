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

from dataclasses import dataclass
from typing import Optional

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xlerobot_vr")
@dataclass
class XLerobotVRTeleopConfig(TeleoperatorConfig):
    """Configuration for XLerobot VR Teleoperator - 优化版本"""
    
    # VR系统设置
    vr_enabled: bool = True
    vr_connection_timeout: float = 10.0  # VR连接超时时间(秒)
    vr_data_timeout: float = 5.0  # VR数据获取超时时间(秒)
    
    # 性能优化设置
    target_fps: float = 60.0  # 目标帧率
    enable_performance_monitoring: bool = True  # 启用性能监控
    cache_size_limit: int = 16  # 缓存大小限制
    
    # 事件处理频率控制（新增）
    event_update_frequency: float = 10.0  # 事件检查频率(Hz)，降低以提升性能
    control_update_frequency: float = 60.0  # 机器人控制频率(Hz)
    
    # 机械臂控制参数
    kp: float = 1.0  # 比例控制增益
    enable_smoothing: bool = True  # 启用平滑控制
    smoothing_alpha: float = 0.1  # 平滑因子(0-1，越小越平滑)
    
    # 运动学参数
    initial_x: float = 0.1629  # 初始x位置
    initial_y: float = 0.1131  # 初始y位置
    initial_pitch: float = 0.0  # 初始pitch角度
    
    # Delta控制参数(优化后)
    vr_deadzone: float = 0.001  # VR死区
    max_delta_per_frame: float = 0.005  # 每帧最大位置变化
    
    # 控制灵敏度参数(可调节以减少卡顿)
    pos_scale: float = 0.01  # 位置灵敏度缩放
    angle_scale: float = 2.0  # 角度灵敏度缩放
    delta_limit: float = 0.01  # 最大位置增量限制(米)
    angle_limit: float = 4.0  # 最大角度增量限制(度)
    
    # 头部控制参数(优化后)
    head_degree_step: float = 2.0  # 头部电机每次移动度数
    head_deadzone: float = 0.15  # 头部控制死区
    head_movement_threshold: float = 0.3  # 头部移动阈值
    head_error_threshold: float = 0.1  # 头部误差阈值(度)
    
    # 基座控制参数(优化后)
    base_deadzone: float = 0.25  # 基座控制死区
    base_enable_caching: bool = True  # 启用基座动作缓存
    
    # IK计算优化
    ik_position_threshold: float = 0.001  # IK计算位置阈值
    enable_ik_caching: bool = True  # 启用IK缓存
    
    # 基座控制参数
    base_acceleration_rate: float = 2.0  # 加速度斜率
    base_deceleration_rate: float = 2.5  # 减速度斜率
    base_max_speed: float = 3.0  # 最大速度倍数
    
    # VR路径配置（可选）
    xlevr_path: Optional[str] = "/home/cics/Desktop/codes/projects/XLeRobot/XLeVR"
    
    # 模拟模式（用于测试）
    mock: bool = False
