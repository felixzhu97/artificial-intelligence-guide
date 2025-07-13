#!/usr/bin/env python3
"""
第26章：机器人学 (Robotics)

本模块实现了机器人学的核心概念：
- 路径规划
- 运动控制
- 传感器融合
- SLAM (同步定位与建图)
- 逆向运动学
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import deque
import math

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class Point:
    """二维点"""
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        """计算到另一点的距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)

@dataclass
class Robot:
    """机器人状态"""
    position: Point
    orientation: float  # 角度（弧度）
    velocity: float = 0.0
    angular_velocity: float = 0.0
    
    def update(self, dt: float):
        """更新机器人状态"""
        # 简单的运动模型
        self.position.x += self.velocity * math.cos(self.orientation) * dt
        self.position.y += self.velocity * math.sin(self.orientation) * dt
        self.orientation += self.angular_velocity * dt
        
        # 角度归一化
        self.orientation = (self.orientation + math.pi) % (2 * math.pi) - math.pi

class GridMap:
    """栅格地图"""
    
    def __init__(self, width: int, height: int, resolution: float = 1.0):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((height, width), dtype=int)  # 0: 空闲, 1: 占用, -1: 未知
    
    def set_obstacle(self, x: int, y: int):
        """设置障碍物"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1
    
    def set_free(self, x: int, y: int):
        """设置空闲空间"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 0
    
    def is_obstacle(self, x: int, y: int) -> bool:
        """检查是否为障碍物"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 1
        return True  # 边界外视为障碍
    
    def is_free(self, x: int, y: int) -> bool:
        """检查是否为空闲空间"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x] == 0
        return False
    
    def world_to_grid(self, point: Point) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        x = int(point.x / self.resolution)
        y = int(point.y / self.resolution)
        return x, y
    
    def grid_to_world(self, x: int, y: int) -> Point:
        """栅格坐标转世界坐标"""
        return Point(x * self.resolution, y * self.resolution)

class AStarPlanner:
    """A*路径规划器"""
    
    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                          (0, 1), (1, -1), (1, 0), (1, 1)]
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """启发式函数（欧几里得距离）"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取邻居节点"""
        neighbors = []
        x, y = node
        
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.grid_map.width and 
                0 <= ny < self.grid_map.height and 
                not self.grid_map.is_obstacle(nx, ny)):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def plan(self, start: Point, goal: Point) -> Optional[List[Point]]:
        """A*路径规划"""
        start_grid = self.grid_map.world_to_grid(start)
        goal_grid = self.grid_map.world_to_grid(goal)
        
        if (self.grid_map.is_obstacle(*start_grid) or 
            self.grid_map.is_obstacle(*goal_grid)):
            return None
        
        # A*算法
        open_set = {start_grid}
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        while open_set:
            # 选择f值最小的节点
            current = min(open_set, key=lambda node: f_score.get(node, float('inf')))
            
            if current == goal_grid:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(self.grid_map.grid_to_world(*current))
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            open_set.remove(current)
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                    open_set.add(neighbor)
        
        return None  # 无路径

class PIDController:
    """PID控制器"""
    
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp  # 比例增益
        self.ki = ki  # 积分增益
        self.kd = kd  # 微分增益
        
        self.previous_error = 0.0
        self.integral = 0.0
    
    def update(self, error: float, dt: float) -> float:
        """更新控制器"""
        # 积分项
        self.integral += error * dt
        
        # 微分项
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        
        # PID输出
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        self.previous_error = error
        return output
    
    def reset(self):
        """重置控制器"""
        self.previous_error = 0.0
        self.integral = 0.0

class MotionController:
    """运动控制器"""
    
    def __init__(self):
        self.position_controller = PIDController(kp=1.0, ki=0.1, kd=0.1)
        self.orientation_controller = PIDController(kp=2.0, ki=0.1, kd=0.2)
        self.max_velocity = 2.0
        self.max_angular_velocity = 1.0
    
    def control(self, robot: Robot, target: Point, dt: float) -> Tuple[float, float]:
        """计算控制命令"""
        # 位置误差
        dx = target.x - robot.position.x
        dy = target.y - robot.position.y
        distance_error = math.sqrt(dx**2 + dy**2)
        
        # 目标角度
        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - robot.orientation
        
        # 角度归一化
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        
        # PID控制
        velocity_command = self.position_controller.update(distance_error, dt)
        angular_velocity_command = self.orientation_controller.update(angle_error, dt)
        
        # 限制速度
        velocity_command = max(-self.max_velocity, min(self.max_velocity, velocity_command))
        angular_velocity_command = max(-self.max_angular_velocity, 
                                     min(self.max_angular_velocity, angular_velocity_command))
        
        return velocity_command, angular_velocity_command

class KalmanFilter:
    """卡尔曼滤波器用于状态估计"""
    
    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x  # 状态维度
        self.dim_z = dim_z  # 观测维度
        
        # 状态向量和协方差矩阵
        self.x = np.zeros((dim_x, 1))  # 状态
        self.P = np.eye(dim_x)         # 状态协方差
        
        # 系统模型
        self.F = np.eye(dim_x)         # 状态转移矩阵
        self.Q = np.eye(dim_x) * 0.1   # 过程噪声协方差
        
        # 观测模型
        self.H = np.eye(dim_z, dim_x)  # 观测矩阵
        self.R = np.eye(dim_z) * 0.1   # 观测噪声协方差
    
    def predict(self):
        """预测步骤"""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
    
    def update(self, z: np.ndarray):
        """更新步骤"""
        # 创新
        y = z - np.dot(self.H, self.x)
        
        # 创新协方差
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # 卡尔曼增益
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # 更新状态和协方差
        self.x = self.x + np.dot(K, y)
        I_KH = np.eye(self.dim_x) - np.dot(K, self.H)
        self.P = np.dot(I_KH, self.P)

class SimpleSLAM:
    """简化的SLAM实现"""
    
    def __init__(self, map_size: int = 100):
        self.map_size = map_size
        self.occupancy_grid = np.zeros((map_size, map_size))  # 占用概率
        self.robot_path = []
        self.landmarks = []
    
    def update_map(self, robot_pos: Point, sensor_data: List[Tuple[float, float]]):
        """更新地图"""
        self.robot_path.append((robot_pos.x, robot_pos.y))
        
        # 简化的激光雷达数据处理
        for distance, angle in sensor_data:
            if distance < 10:  # 有效距离内
                # 计算障碍物位置
                obstacle_x = robot_pos.x + distance * math.cos(angle)
                obstacle_y = robot_pos.y + distance * math.sin(angle)
                
                # 更新占用网格
                grid_x = int(obstacle_x + self.map_size // 2)
                grid_y = int(obstacle_y + self.map_size // 2)
                
                if 0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size:
                    self.occupancy_grid[grid_y, grid_x] += 0.1
                    self.occupancy_grid[grid_y, grid_x] = min(1.0, self.occupancy_grid[grid_y, grid_x])
    
    def get_map(self) -> np.ndarray:
        """获取当前地图"""
        return self.occupancy_grid

class RobotArmKinematics:
    """机械臂运动学"""
    
    def __init__(self, link_lengths: List[float]):
        self.link_lengths = link_lengths
        self.n_joints = len(link_lengths)
    
    def forward_kinematics(self, joint_angles: List[float]) -> Point:
        """正向运动学：由关节角度计算末端位置"""
        if len(joint_angles) != self.n_joints:
            raise ValueError("关节角度数量必须等于关节数量")
        
        x, y = 0, 0
        cumulative_angle = 0
        
        for i, (length, angle) in enumerate(zip(self.link_lengths, joint_angles)):
            cumulative_angle += angle
            x += length * math.cos(cumulative_angle)
            y += length * math.sin(cumulative_angle)
        
        return Point(x, y)
    
    def inverse_kinematics_2dof(self, target: Point) -> Optional[List[float]]:
        """逆向运动学：2自由度机械臂"""
        if self.n_joints != 2:
            return None
        
        l1, l2 = self.link_lengths
        x, y = target.x, target.y
        
        # 距离检查
        distance = math.sqrt(x**2 + y**2)
        if distance > l1 + l2 or distance < abs(l1 - l2):
            return None  # 目标不可达
        
        # 计算关节角度
        cos_q2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_q2 = max(-1, min(1, cos_q2))  # 限制在[-1, 1]
        
        q2 = math.acos(cos_q2)
        q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        
        return [q1, q2]
    
    def jacobian_2dof(self, joint_angles: List[float]) -> np.ndarray:
        """雅可比矩阵：2自由度机械臂"""
        if len(joint_angles) != 2:
            return None
        
        q1, q2 = joint_angles
        l1, l2 = self.link_lengths
        
        # 雅可比矩阵
        J = np.array([
            [-l1 * math.sin(q1) - l2 * math.sin(q1 + q2), -l2 * math.sin(q1 + q2)],
            [l1 * math.cos(q1) + l2 * math.cos(q1 + q2), l2 * math.cos(q1 + q2)]
        ])
        
        return J

def demo_path_planning():
    """演示路径规划"""
    print("\n" + "="*50)
    print("路径规划演示")
    print("="*50)
    
    # 创建栅格地图
    grid_map = GridMap(width=20, height=20, resolution=1.0)
    
    # 添加障碍物
    obstacles = [
        (5, 5), (5, 6), (5, 7), (6, 7), (7, 7),
        (10, 10), (10, 11), (11, 10), (11, 11),
        (15, 5), (15, 6), (15, 7), (15, 8),
        (3, 15), (4, 15), (5, 15), (6, 15)
    ]
    
    for x, y in obstacles:
        grid_map.set_obstacle(x, y)
    
    print(f"地图大小: {grid_map.width} x {grid_map.height}")
    print(f"障碍物数量: {len(obstacles)}")
    
    # A*路径规划
    planner = AStarPlanner(grid_map)
    start = Point(1, 1)
    goal = Point(18, 18)
    
    print(f"起点: ({start.x}, {start.y})")
    print(f"终点: ({goal.x}, {goal.y})")
    
    path = planner.plan(start, goal)
    
    if path:
        print(f"路径找到! 路径长度: {len(path)} 个点")
        
        # 可视化
        plt.figure(figsize=(10, 10))
        
        # 绘制地图
        map_display = np.ones((grid_map.height, grid_map.width)) * 0.5  # 灰色为空闲
        for x, y in obstacles:
            map_display[y, x] = 0  # 黑色为障碍物
        
        plt.imshow(map_display, cmap='gray', origin='lower')
        
        # 绘制路径
        if path:
            path_x = [p.x for p in path]
            path_y = [p.y for p in path]
            plt.plot(path_x, path_y, 'r-', linewidth=2, label='A*路径')
            plt.plot(path_x, path_y, 'ro', markersize=4)
        
        # 标记起点和终点
        plt.plot(start.x, start.y, 'go', markersize=10, label='起点')
        plt.plot(goal.x, goal.y, 'bo', markersize=10, label='终点')
        
        plt.title('A*路径规划')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('path_planning.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("路径规划图已保存为 'path_planning.png'")
    else:
        print("未找到路径!")

def demo_motion_control():
    """演示运动控制"""
    print("\n" + "="*50)
    print("运动控制演示")
    print("="*50)
    
    # 创建机器人和控制器
    robot = Robot(position=Point(0, 0), orientation=0)
    controller = MotionController()
    
    # 目标轨迹（圆形）
    target_points = []
    for i in range(100):
        angle = 2 * math.pi * i / 100
        x = 5 * math.cos(angle)
        y = 5 * math.sin(angle)
        target_points.append(Point(x, y))
    
    print(f"目标轨迹: 圆形，半径=5，点数={len(target_points)}")
    
    # 仿真
    dt = 0.1
    robot_trajectory = []
    
    for target in target_points:
        # 控制命令
        velocity, angular_velocity = controller.control(robot, target, dt)
        
        # 更新机器人状态
        robot.velocity = velocity
        robot.angular_velocity = angular_velocity
        robot.update(dt)
        
        robot_trajectory.append((robot.position.x, robot.position.y))
    
    print(f"仿真完成，机器人轨迹长度: {len(robot_trajectory)}")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    # 目标轨迹
    target_x = [p.x for p in target_points]
    target_y = [p.y for p in target_points]
    plt.plot(target_x, target_y, 'b--', linewidth=2, label='目标轨迹')
    
    # 机器人轨迹
    robot_x = [p[0] for p in robot_trajectory]
    robot_y = [p[1] for p in robot_trajectory]
    plt.plot(robot_x, robot_y, 'r-', linewidth=2, label='机器人轨迹')
    
    # 起点和终点
    plt.plot(robot_x[0], robot_y[0], 'go', markersize=10, label='起点')
    plt.plot(robot_x[-1], robot_y[-1], 'ro', markersize=10, label='终点')
    
    plt.title('机器人运动控制')
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('motion_control.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("运动控制图已保存为 'motion_control.png'")

def demo_kalman_filter():
    """演示卡尔曼滤波"""
    print("\n" + "="*50)
    print("卡尔曼滤波演示")
    print("="*50)
    
    # 创建卡尔曼滤波器 (位置和速度)
    kf = KalmanFilter(dim_x=4, dim_z=2)  # [x, y, vx, vy], [x, y]
    
    # 设置状态转移矩阵（匀速模型）
    dt = 0.1
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # 观测矩阵（只观测位置）
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # 噪声设置
    kf.Q *= 0.01  # 过程噪声
    kf.R *= 0.1   # 观测噪声
    
    # 初始状态
    kf.x = np.array([[0], [0], [1], [1]])  # 初始位置(0,0)，速度(1,1)
    
    print("卡尔曼滤波器参数:")
    print(f"状态维度: {kf.dim_x}, 观测维度: {kf.dim_z}")
    print(f"时间步长: {dt}")
    
    # 仿真真实轨迹和观测
    true_positions = []
    observations = []
    estimates = []
    
    for t in range(50):
        # 真实位置（带噪声的圆形运动）
        true_x = 5 * math.cos(0.1 * t) + np.random.normal(0, 0.05)
        true_y = 5 * math.sin(0.1 * t) + np.random.normal(0, 0.05)
        true_positions.append((true_x, true_y))
        
        # 带噪声的观测
        obs_x = true_x + np.random.normal(0, 0.2)
        obs_y = true_y + np.random.normal(0, 0.2)
        observation = np.array([[obs_x], [obs_y]])
        observations.append((obs_x, obs_y))
        
        # 卡尔曼滤波
        kf.predict()
        kf.update(observation)
        
        estimates.append((kf.x[0, 0], kf.x[1, 0]))
    
    print(f"仿真步数: {len(true_positions)}")
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 真实轨迹
    true_x = [p[0] for p in true_positions]
    true_y = [p[1] for p in true_positions]
    plt.plot(true_x, true_y, 'g-', linewidth=2, label='真实轨迹')
    
    # 观测值
    obs_x = [p[0] for p in observations]
    obs_y = [p[1] for p in observations]
    plt.scatter(obs_x, obs_y, c='red', alpha=0.6, s=20, label='噪声观测')
    
    # 卡尔曼滤波估计
    est_x = [p[0] for p in estimates]
    est_y = [p[1] for p in estimates]
    plt.plot(est_x, est_y, 'b-', linewidth=2, label='卡尔曼滤波估计')
    
    plt.title('卡尔曼滤波状态估计')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('kalman_filter.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("卡尔曼滤波图已保存为 'kalman_filter.png'")

def demo_robot_arm_kinematics():
    """演示机械臂运动学"""
    print("\n" + "="*50)
    print("机械臂运动学演示")
    print("="*50)
    
    # 创建2自由度机械臂
    link_lengths = [3.0, 2.0]
    arm = RobotArmKinematics(link_lengths)
    
    print(f"机械臂配置: {len(link_lengths)}自由度")
    print(f"连杆长度: {link_lengths}")
    
    # 正向运动学演示
    joint_angles = [math.pi/4, math.pi/3]
    end_effector = arm.forward_kinematics(joint_angles)
    
    print(f"\n正向运动学:")
    print(f"关节角度: {[math.degrees(a) for a in joint_angles]} 度")
    print(f"末端位置: ({end_effector.x:.2f}, {end_effector.y:.2f})")
    
    # 逆向运动学演示
    target = Point(3.5, 2.5)
    ik_solution = arm.inverse_kinematics_2dof(target)
    
    print(f"\n逆向运动学:")
    print(f"目标位置: ({target.x}, {target.y})")
    
    if ik_solution:
        print(f"求解成功!")
        print(f"关节角度: {[math.degrees(a) for a in ik_solution]} 度")
        
        # 验证
        verification = arm.forward_kinematics(ik_solution)
        error = target.distance_to(verification)
        print(f"验证误差: {error:.6f}")
    else:
        print(f"目标不可达!")
    
    # 可视化工作空间
    plt.figure(figsize=(12, 10))
    
    # 绘制工作空间边界
    angles = np.linspace(0, 2*math.pi, 100)
    
    # 最大伸展圆
    max_reach = sum(link_lengths)
    max_x = max_reach * np.cos(angles)
    max_y = max_reach * np.sin(angles)
    plt.plot(max_x, max_y, 'r--', label=f'最大工作范围 (r={max_reach})')
    
    # 最小伸展圆
    min_reach = abs(link_lengths[0] - link_lengths[1])
    if min_reach > 0:
        min_x = min_reach * np.cos(angles)
        min_y = min_reach * np.sin(angles)
        plt.plot(min_x, min_y, 'r--', label=f'最小工作范围 (r={min_reach})')
    
    # 绘制几个配置
    configurations = [
        [0, 0],
        [math.pi/4, math.pi/4],
        [math.pi/2, -math.pi/4],
        [math.pi, math.pi/2],
        [-math.pi/4, -math.pi/3]
    ]
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (config, color) in enumerate(zip(configurations, colors)):
        # 计算关节位置
        joint1_pos = Point(link_lengths[0] * math.cos(config[0]), 
                          link_lengths[0] * math.sin(config[0]))
        
        end_pos = arm.forward_kinematics(config)
        
        # 绘制机械臂
        plt.plot([0, joint1_pos.x, end_pos.x], 
                [0, joint1_pos.y, end_pos.y], 
                'o-', color=color, linewidth=2, markersize=6,
                label=f'配置{i+1}')
    
    # 标记特殊点
    if ik_solution:
        plt.plot(target.x, target.y, 'rs', markersize=10, label='IK目标')
    
    plt.plot(0, 0, 'ko', markersize=10, label='基座')
    
    plt.title('2自由度机械臂工作空间')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('robot_arm_kinematics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("机械臂运动学图已保存为 'robot_arm_kinematics.png'")

def demo_simple_slam():
    """演示简化SLAM"""
    print("\n" + "="*50)
    print("简化SLAM演示")
    print("="*50)
    
    # 创建SLAM系统
    slam = SimpleSLAM(map_size=50)
    
    # 模拟机器人轨迹和传感器数据
    robot_positions = []
    sensor_data_sequence = []
    
    print("模拟机器人探索...")
    
    for t in range(30):
        # 机器人圆形运动
        x = 5 * math.cos(0.2 * t)
        y = 5 * math.sin(0.2 * t)
        robot_pos = Point(x, y)
        robot_positions.append(robot_pos)
        
        # 模拟激光雷达数据
        sensor_data = []
        for angle in np.linspace(0, 2*math.pi, 8):  # 8方向激光
            # 模拟环境中的障碍物
            distance = 8 + 2 * math.sin(angle * 3) + np.random.normal(0, 0.1)
            distance = max(0.5, min(10, distance))  # 限制距离范围
            sensor_data.append((distance, angle))
        
        sensor_data_sequence.append(sensor_data)
        
        # 更新SLAM
        slam.update_map(robot_pos, sensor_data)
    
    print(f"探索完成，访问了 {len(robot_positions)} 个位置")
    
    # 获取建图结果
    occupancy_map = slam.get_map()
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 轨迹图
    traj_x = [pos.x for pos in robot_positions]
    traj_y = [pos.y for pos in robot_positions]
    ax1.plot(traj_x, traj_y, 'b-', linewidth=2, label='机器人轨迹')
    ax1.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='起点')
    ax1.plot(traj_x[-1], traj_y[-1], 'ro', markersize=10, label='终点')
    
    # 绘制部分传感器数据
    for i in range(0, len(robot_positions), 5):
        robot_pos = robot_positions[i]
        sensor_data = sensor_data_sequence[i]
        
        for distance, angle in sensor_data:
            if distance < 10:
                end_x = robot_pos.x + distance * math.cos(angle)
                end_y = robot_pos.y + distance * math.sin(angle)
                ax1.plot([robot_pos.x, end_x], [robot_pos.y, end_y], 'r-', alpha=0.3)
    
    ax1.set_title('机器人轨迹和传感器数据')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 占用地图
    im = ax2.imshow(occupancy_map, cmap='gray', origin='lower')
    ax2.set_title('SLAM构建的占用地图')
    ax2.set_xlabel('栅格X')
    ax2.set_ylabel('栅格Y')
    plt.colorbar(im, ax=ax2, label='占用概率')
    
    plt.tight_layout()
    plt.savefig('simple_slam.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("SLAM演示图已保存为 'simple_slam.png'")

def run_comprehensive_demo():
    """运行完整演示"""
    print("🤖 第26章：机器人学 - 完整演示")
    print("="*60)
    
    # 运行各个演示
    demo_path_planning()
    demo_motion_control()
    demo_kalman_filter()
    demo_robot_arm_kinematics()
    demo_simple_slam()
    
    print("\n" + "="*60)
    print("机器人学演示完成！")
    print("="*60)
    print("\n📚 学习要点:")
    print("• 路径规划解决机器人如何到达目标的问题")
    print("• 运动控制确保机器人精确跟踪预定轨迹")
    print("• 卡尔曼滤波用于状态估计和传感器融合")
    print("• 运动学分析机械臂的位置和姿态关系")
    print("• SLAM实现机器人同时定位与建图")
    print("• 机器人学是AI与物理世界交互的重要桥梁")

if __name__ == "__main__":
    run_comprehensive_demo() 