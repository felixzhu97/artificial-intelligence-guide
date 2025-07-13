#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《人工智能：现代方法》第17章：复杂决策
Complex Decisions - Markov Decision Processes

本模块实现了马尔可夫决策过程(MDP)的核心算法，包括价值迭代和策略迭代等求解方法。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum
import random
import copy

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ActionType(Enum):
    """动作类型"""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"

class State:
    """状态类"""
    def __init__(self, x: int, y: int, is_terminal: bool = False):
        self.x = x
        self.y = y
        self.is_terminal = is_terminal
    
    def __eq__(self, other):
        return isinstance(other, State) and self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __repr__(self):
        return f"State({self.x}, {self.y})"

class Action:
    """动作类"""
    def __init__(self, action_type: ActionType):
        self.action_type = action_type
    
    def __eq__(self, other):
        return isinstance(other, Action) and self.action_type == other.action_type
    
    def __hash__(self):
        return hash(self.action_type)
    
    def __repr__(self):
        return f"Action({self.action_type.value})"

class MDP:
    """马尔可夫决策过程"""
    
    def __init__(self, width: int, height: int, discount_factor: float = 0.9):
        self.width = width
        self.height = height
        self.discount_factor = discount_factor
        
        # 创建状态空间
        self.states = []
        for y in range(height):
            for x in range(width):
                self.states.append(State(x, y))
        
        # 动作空间
        self.actions = [Action(action_type) for action_type in ActionType]
        
        # 转移概率: P(s'|s,a)
        self.transition_probabilities = {}
        
        # 奖励函数: R(s,a,s')
        self.rewards = {}
        
        # 终端状态
        self.terminal_states = set()
        
        # 障碍物
        self.obstacles = set()
        
        # 初始化环境
        self._initialize_environment()
    
    def _initialize_environment(self):
        """初始化环境"""
        # 设置默认奖励
        for state in self.states:
            for action in self.actions:
                self.rewards[(state, action)] = -0.04  # 生存惩罚
        
        # 设置默认转移概率
        for state in self.states:
            for action in self.actions:
                self.transition_probabilities[(state, action)] = {}
                
                # 计算期望的下一个状态
                next_state = self._get_next_state(state, action)
                
                # 随机性：80%执行期望动作，20%执行其他动作
                if self._is_valid_state(next_state):
                    self.transition_probabilities[(state, action)][next_state] = 0.8
                else:
                    self.transition_probabilities[(state, action)][state] = 0.8
                
                # 侧向滑动
                left_action = self._get_perpendicular_action(action, True)
                right_action = self._get_perpendicular_action(action, False)
                
                left_state = self._get_next_state(state, left_action)
                right_state = self._get_next_state(state, right_action)
                
                if self._is_valid_state(left_state):
                    self.transition_probabilities[(state, action)][left_state] = 0.1
                else:
                    self.transition_probabilities[(state, action)][state] = \
                        self.transition_probabilities[(state, action)].get(state, 0) + 0.1
                
                if self._is_valid_state(right_state):
                    self.transition_probabilities[(state, action)][right_state] = 0.1
                else:
                    self.transition_probabilities[(state, action)][state] = \
                        self.transition_probabilities[(state, action)].get(state, 0) + 0.1
    
    def _get_next_state(self, state: State, action: Action) -> State:
        """获取执行动作后的下一个状态"""
        if action.action_type == ActionType.UP:
            return State(state.x, state.y + 1)
        elif action.action_type == ActionType.DOWN:
            return State(state.x, state.y - 1)
        elif action.action_type == ActionType.LEFT:
            return State(state.x - 1, state.y)
        elif action.action_type == ActionType.RIGHT:
            return State(state.x + 1, state.y)
        else:
            return state
    
    def _get_perpendicular_action(self, action: Action, left: bool) -> Action:
        """获取垂直方向的动作"""
        perpendicular_map = {
            ActionType.UP: ActionType.LEFT if left else ActionType.RIGHT,
            ActionType.DOWN: ActionType.RIGHT if left else ActionType.LEFT,
            ActionType.LEFT: ActionType.DOWN if left else ActionType.UP,
            ActionType.RIGHT: ActionType.UP if left else ActionType.DOWN,
            ActionType.STAY: ActionType.STAY  # STAY动作保持不变
        }
        return Action(perpendicular_map[action.action_type])
    
    def _is_valid_state(self, state: State) -> bool:
        """检查状态是否有效"""
        return (0 <= state.x < self.width and 
                0 <= state.y < self.height and 
                state not in self.obstacles)
    
    def set_terminal_state(self, x: int, y: int, reward: float):
        """设置终端状态"""
        state = State(x, y, is_terminal=True)
        self.terminal_states.add(state)
        
        # 更新奖励
        for action in self.actions:
            self.rewards[(state, action)] = reward
    
    def set_obstacle(self, x: int, y: int):
        """设置障碍物"""
        obstacle = State(x, y)
        self.obstacles.add(obstacle)
    
    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        """获取奖励"""
        if state in self.terminal_states:
            return self.rewards.get((state, action), 0)
        return self.rewards.get((state, action), -0.04)
    
    def get_transition_probability(self, state: State, action: Action, next_state: State) -> float:
        """获取转移概率"""
        if state in self.terminal_states:
            return 1.0 if state == next_state else 0.0
        return self.transition_probabilities.get((state, action), {}).get(next_state, 0.0)
    
    def get_legal_actions(self, state: State) -> List[Action]:
        """获取状态的合法动作"""
        if state in self.terminal_states:
            return []
        return self.actions

class ValueIteration:
    """价值迭代算法"""
    
    def __init__(self, mdp: MDP, tolerance: float = 1e-6):
        self.mdp = mdp
        self.tolerance = tolerance
        self.values = {state: 0.0 for state in mdp.states}
        self.policy = {}
        self.iteration_count = 0
        self.convergence_history = []
    
    def iterate(self) -> bool:
        """执行一次价值迭代"""
        new_values = {}
        max_change = 0.0
        
        for state in self.mdp.states:
            if state in self.mdp.terminal_states:
                new_values[state] = self.values[state]
                continue
            
            # 计算所有动作的期望价值
            action_values = []
            for action in self.mdp.get_legal_actions(state):
                expected_value = 0.0
                
                for next_state in self.mdp.states:
                    prob = self.mdp.get_transition_probability(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    expected_value += prob * (reward + self.mdp.discount_factor * self.values[next_state])
                
                action_values.append(expected_value)
            
            # 选择最大价值
            new_values[state] = max(action_values) if action_values else 0.0
            max_change = max(max_change, abs(new_values[state] - self.values[state]))
        
        self.values = new_values
        self.iteration_count += 1
        self.convergence_history.append(max_change)
        
        return max_change < self.tolerance
    
    def solve(self, max_iterations: int = 1000) -> Dict[State, float]:
        """求解MDP"""
        for _ in range(max_iterations):
            if self.iterate():
                break
        
        # 计算最优策略
        self._compute_policy()
        return self.values
    
    def _compute_policy(self):
        """计算最优策略"""
        for state in self.mdp.states:
            if state in self.mdp.terminal_states:
                continue
            
            best_action = None
            best_value = float('-inf')
            
            for action in self.mdp.get_legal_actions(state):
                expected_value = 0.0
                
                for next_state in self.mdp.states:
                    prob = self.mdp.get_transition_probability(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    expected_value += prob * (reward + self.mdp.discount_factor * self.values[next_state])
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action
            
            self.policy[state] = best_action

class PolicyIteration:
    """策略迭代算法"""
    
    def __init__(self, mdp: MDP, tolerance: float = 1e-6):
        self.mdp = mdp
        self.tolerance = tolerance
        self.policy = {}
        self.values = {state: 0.0 for state in mdp.states}
        self.iteration_count = 0
        self.policy_changes = []
        
        # 初始化随机策略
        self._initialize_random_policy()
    
    def _initialize_random_policy(self):
        """初始化随机策略"""
        for state in self.mdp.states:
            if state not in self.mdp.terminal_states:
                legal_actions = self.mdp.get_legal_actions(state)
                if legal_actions:
                    self.policy[state] = random.choice(legal_actions)
    
    def policy_evaluation(self, max_iterations: int = 1000) -> Dict[State, float]:
        """策略评估"""
        for _ in range(max_iterations):
            new_values = {}
            max_change = 0.0
            
            for state in self.mdp.states:
                if state in self.mdp.terminal_states:
                    new_values[state] = self.values[state]
                    continue
                
                action = self.policy.get(state)
                if action is None:
                    new_values[state] = 0.0
                    continue
                
                expected_value = 0.0
                for next_state in self.mdp.states:
                    prob = self.mdp.get_transition_probability(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    expected_value += prob * (reward + self.mdp.discount_factor * self.values[next_state])
                
                new_values[state] = expected_value
                max_change = max(max_change, abs(new_values[state] - self.values[state]))
            
            self.values = new_values
            
            if max_change < self.tolerance:
                break
        
        return self.values
    
    def policy_improvement(self) -> bool:
        """策略改进"""
        policy_stable = True
        
        for state in self.mdp.states:
            if state in self.mdp.terminal_states:
                continue
            
            old_action = self.policy.get(state)
            best_action = None
            best_value = float('-inf')
            
            for action in self.mdp.get_legal_actions(state):
                expected_value = 0.0
                
                for next_state in self.mdp.states:
                    prob = self.mdp.get_transition_probability(state, action, next_state)
                    reward = self.mdp.get_reward(state, action, next_state)
                    expected_value += prob * (reward + self.mdp.discount_factor * self.values[next_state])
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action
            
            self.policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def solve(self, max_iterations: int = 100) -> Tuple[Dict[State, float], Dict[State, Action]]:
        """求解MDP"""
        for _ in range(max_iterations):
            self.policy_evaluation()
            if self.policy_improvement():
                break
            self.iteration_count += 1
        
        return self.values, self.policy

class MDPVisualization:
    """MDP可视化"""
    
    def __init__(self, mdp: MDP):
        self.mdp = mdp
    
    def plot_values(self, values: Dict[State, float], title: str = "状态价值函数"):
        """绘制状态价值函数"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建价值矩阵
        value_matrix = np.zeros((self.mdp.height, self.mdp.width))
        for state in self.mdp.states:
            if state not in self.mdp.obstacles:
                value_matrix[self.mdp.height - 1 - state.y, state.x] = values[state]
        
        # 绘制热力图
        im = ax.imshow(value_matrix, cmap='RdYlBu', aspect='equal')
        
        # 添加数值标签
        for y in range(self.mdp.height):
            for x in range(self.mdp.width):
                state = State(x, self.mdp.height - 1 - y)
                if state in self.mdp.obstacles:
                    ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                                 facecolor='black', edgecolor='white'))
                    ax.text(x, y, '障碍', ha='center', va='center', color='white', fontsize=10)
                elif state in self.mdp.terminal_states:
                    ax.text(x, y, f'{values[state]:.2f}\n(终端)', ha='center', va='center', 
                           color='black', fontsize=10, weight='bold')
                else:
                    ax.text(x, y, f'{values[state]:.2f}', ha='center', va='center', 
                           color='black', fontsize=12)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        
        # 设置刻度
        ax.set_xticks(range(self.mdp.width))
        ax.set_yticks(range(self.mdp.height))
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('价值', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.show()
    
    def plot_policy(self, policy: Dict[State, Action], title: str = "最优策略"):
        """绘制策略"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制网格
        ax.set_xlim(-0.5, self.mdp.width - 0.5)
        ax.set_ylim(-0.5, self.mdp.height - 0.5)
        
        # 动作箭头映射
        arrow_map = {
            ActionType.UP: (0, 0.3),
            ActionType.DOWN: (0, -0.3),
            ActionType.LEFT: (-0.3, 0),
            ActionType.RIGHT: (0.3, 0),
            ActionType.STAY: (0, 0)
        }
        
        for y in range(self.mdp.height):
            for x in range(self.mdp.width):
                state = State(x, self.mdp.height - 1 - y)
                
                if state in self.mdp.obstacles:
                    ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                                 facecolor='black', edgecolor='white'))
                    ax.text(x, y, '障碍', ha='center', va='center', color='white', fontsize=10)
                elif state in self.mdp.terminal_states:
                    ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                                 facecolor='gold', edgecolor='black'))
                    ax.text(x, y, '终端', ha='center', va='center', color='black', fontsize=10)
                elif state in policy:
                    action = policy[state]
                    dx, dy = arrow_map[action.action_type]
                    ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, 
                            fc='red', ec='red', linewidth=2)
        
        # 绘制网格线
        for i in range(self.mdp.width + 1):
            ax.axvline(i - 0.5, color='gray', linewidth=0.5)
        for i in range(self.mdp.height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        
        # 设置刻度
        ax.set_xticks(range(self.mdp.width))
        ax.set_yticks(range(self.mdp.height))
        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence(self, convergence_history: List[float], title: str = "收敛历史"):
        """绘制收敛历史"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(convergence_history, 'b-', linewidth=2, marker='o')
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('最大价值变化', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()

def demo_grid_world():
    """演示网格世界MDP"""
    print("=" * 50)
    print("网格世界马尔可夫决策过程演示")
    print("=" * 50)
    
    # 创建4x3的网格世界
    mdp = MDP(width=4, height=3, discount_factor=0.9)
    
    # 设置终端状态
    mdp.set_terminal_state(3, 2, 1.0)   # 目标状态：奖励+1
    mdp.set_terminal_state(3, 1, -1.0)  # 陷阱状态：奖励-1
    
    # 设置障碍物
    mdp.set_obstacle(1, 1)
    
    print(f"网格世界设置:")
    print(f"- 大小: {mdp.width}x{mdp.height}")
    print(f"- 折扣因子: {mdp.discount_factor}")
    print(f"- 终端状态: (3,2)奖励+1, (3,1)奖励-1")
    print(f"- 障碍物: (1,1)")
    print(f"- 生存惩罚: -0.04")
    
    # 价值迭代
    print("\n" + "=" * 30)
    print("价值迭代求解")
    print("=" * 30)
    
    vi = ValueIteration(mdp, tolerance=1e-6)
    vi_values = vi.solve(max_iterations=1000)
    
    print(f"收敛迭代次数: {vi.iteration_count}")
    print(f"最终价值函数:")
    for y in range(mdp.height-1, -1, -1):
        for x in range(mdp.width):
            state = State(x, y)
            if state in mdp.obstacles:
                print(f"{'障碍':>8}", end="")
            elif state in mdp.terminal_states:
                print(f"{vi_values[state]:>8.3f}*", end="")
            else:
                print(f"{vi_values[state]:>8.3f}", end="")
        print()
    
    # 策略迭代
    print("\n" + "=" * 30)
    print("策略迭代求解")
    print("=" * 30)
    
    pi = PolicyIteration(mdp, tolerance=1e-6)
    pi_values, pi_policy = pi.solve(max_iterations=100)
    
    print(f"收敛迭代次数: {pi.iteration_count}")
    print(f"最终策略:")
    for y in range(mdp.height-1, -1, -1):
        for x in range(mdp.width):
            state = State(x, y)
            if state in mdp.obstacles:
                print(f"{'障碍':>8}", end="")
            elif state in mdp.terminal_states:
                print(f"{'终端':>8}", end="")
            elif state in pi_policy:
                action_map = {
                    ActionType.UP: "↑",
                    ActionType.DOWN: "↓",
                    ActionType.LEFT: "←",
                    ActionType.RIGHT: "→",
                    ActionType.STAY: "○"
                }
                symbol = action_map[pi_policy[state].action_type]
                print(f"{symbol:>8}", end="")
            else:
                print(f"{'?':>8}", end="")
        print()
    
    # 比较两种方法
    print("\n" + "=" * 30)
    print("算法比较")
    print("=" * 30)
    
    value_diff = 0
    for state in mdp.states:
        if state not in mdp.obstacles:
            value_diff += abs(vi_values[state] - pi_values[state])
    
    print(f"价值函数差异总和: {value_diff:.6f}")
    print(f"价值迭代收敛次数: {vi.iteration_count}")
    print(f"策略迭代收敛次数: {pi.iteration_count}")
    
    # 可视化
    viz = MDPVisualization(mdp)
    viz.plot_values(vi_values, "价值迭代 - 状态价值函数")
    viz.plot_policy(vi.policy, "价值迭代 - 最优策略")
    viz.plot_convergence(vi.convergence_history, "价值迭代收敛历史")

def demo_policy_comparison():
    """演示不同策略的性能比较"""
    print("\n" + "=" * 50)
    print("不同策略性能比较")
    print("=" * 50)
    
    # 创建MDP
    mdp = MDP(width=4, height=3, discount_factor=0.9)
    mdp.set_terminal_state(3, 2, 1.0)
    mdp.set_terminal_state(3, 1, -1.0)
    mdp.set_obstacle(1, 1)
    
    # 随机策略
    print("\n随机策略性能:")
    random_policy = {}
    for state in mdp.states:
        if state not in mdp.terminal_states and state not in mdp.obstacles:
            random_policy[state] = random.choice(mdp.get_legal_actions(state))
    
    # 评估随机策略
    pi_random = PolicyIteration(mdp)
    pi_random.policy = random_policy
    random_values = pi_random.policy_evaluation()
    
    avg_random_value = np.mean([v for s, v in random_values.items() 
                               if s not in mdp.terminal_states and s not in mdp.obstacles])
    print(f"随机策略平均价值: {avg_random_value:.4f}")
    
    # 最优策略
    vi_optimal = ValueIteration(mdp)
    optimal_values = vi_optimal.solve()
    
    avg_optimal_value = np.mean([v for s, v in optimal_values.items() 
                                if s not in mdp.terminal_states and s not in mdp.obstacles])
    print(f"最优策略平均价值: {avg_optimal_value:.4f}")
    
    improvement = avg_optimal_value - avg_random_value
    print(f"性能提升: {improvement:.4f} ({improvement/abs(avg_random_value)*100:.1f}%)")

def demo_parameter_sensitivity():
    """演示参数敏感性分析"""
    print("\n" + "=" * 50)
    print("参数敏感性分析")
    print("=" * 50)
    
    # 折扣因子敏感性
    print("\n折扣因子敏感性:")
    discount_factors = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    
    for gamma in discount_factors:
        mdp = MDP(width=4, height=3, discount_factor=gamma)
        mdp.set_terminal_state(3, 2, 1.0)
        mdp.set_terminal_state(3, 1, -1.0)
        mdp.set_obstacle(1, 1)
        
        vi = ValueIteration(mdp)
        values = vi.solve()
        
        start_state = State(0, 0)
        start_value = values[start_state]
        
        print(f"γ = {gamma:.2f}: 起始状态价值 = {start_value:.4f}, 收敛次数 = {vi.iteration_count}")

if __name__ == "__main__":
    print("《人工智能：现代方法》第17章：复杂决策")
    print("=" * 60)
    
    try:
        # 主要演示
        demo_grid_world()
        
        # 策略比较
        demo_policy_comparison()
        
        # 参数敏感性
        demo_parameter_sensitivity()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 