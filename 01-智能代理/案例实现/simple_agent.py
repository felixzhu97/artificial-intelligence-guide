"""
智能代理案例实现

本模块演示了《人工智能：现代方法》第2章中介绍的各种智能代理类型：
1. 简单反射代理 (Simple Reflex Agent)
2. 基于模型的反射代理 (Model-based Reflex Agent)
3. 基于目标的代理 (Goal-based Agent)
4. 基于效用的代理 (Utility-based Agent)
5. 学习代理 (Learning Agent)
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    """环境中的动作"""
    MOVE_UP = "up"
    MOVE_DOWN = "down"
    MOVE_LEFT = "left"
    MOVE_RIGHT = "right"
    STAY = "stay"
    CLEAN = "clean"


class EnvironmentState(Enum):
    """环境状态"""
    CLEAN = "clean"
    DIRTY = "dirty"
    WALL = "wall"
    EMPTY = "empty"


@dataclass
class Percept:
    """感知信息"""
    location: Tuple[int, int]
    status: EnvironmentState
    bump: bool = False
    
    def __str__(self):
        return f"位置: {self.location}, 状态: {self.status.value}, 撞墙: {self.bump}"


class Environment:
    """环境类 - 代表代理所在的世界"""
    
    def __init__(self, width: int = 4, height: int = 4, dirt_prob: float = 0.3):
        self.width = width
        self.height = height
        self.grid = [[EnvironmentState.CLEAN for _ in range(width)] for _ in range(height)]
        self.agent_location = (0, 0)
        self.performance_score = 0
        self.time_step = 0
        self.max_steps = 100
        
        # 随机放置一些污垢
        for i in range(height):
            for j in range(width):
                if random.random() < dirt_prob:
                    self.grid[i][j] = EnvironmentState.DIRTY
    
    def get_percept(self) -> Percept:
        """获取当前位置的感知信息"""
        x, y = self.agent_location
        status = self.grid[y][x]
        return Percept(location=(x, y), status=status)
    
    def execute_action(self, action: Action) -> Tuple[Percept, float]:
        """执行动作并返回新的感知和奖励"""
        self.time_step += 1
        reward = 0
        
        if action == Action.CLEAN:
            x, y = self.agent_location
            if self.grid[y][x] == EnvironmentState.DIRTY:
                self.grid[y][x] = EnvironmentState.CLEAN
                reward = 10  # 清理污垢的奖励
            else:
                reward = -1  # 清理已经干净的地方的惩罚
        
        elif action in [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]:
            new_location = self._calculate_new_location(action)
            if self._is_valid_location(new_location):
                self.agent_location = new_location
                reward = -0.1  # 移动的小代价
            else:
                reward = -5  # 撞墙的惩罚
        
        elif action == Action.STAY:
            reward = -0.5  # 停留的代价
        
        self.performance_score += reward
        return self.get_percept(), reward
    
    def _calculate_new_location(self, action: Action) -> Tuple[int, int]:
        """计算新位置"""
        x, y = self.agent_location
        if action == Action.MOVE_UP:
            return (x, y - 1)
        elif action == Action.MOVE_DOWN:
            return (x, y + 1)
        elif action == Action.MOVE_LEFT:
            return (x - 1, y)
        elif action == Action.MOVE_RIGHT:
            return (x + 1, y)
        return (x, y)
    
    def _is_valid_location(self, location: Tuple[int, int]) -> bool:
        """检查位置是否有效"""
        x, y = location
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_done(self) -> bool:
        """检查是否结束"""
        return self.time_step >= self.max_steps or self._all_clean()
    
    def _all_clean(self) -> bool:
        """检查是否所有地方都干净"""
        for row in self.grid:
            for cell in row:
                if cell == EnvironmentState.DIRTY:
                    return False
        return True
    
    def print_environment(self):
        """打印环境状态"""
        print(f"时间步: {self.time_step}, 性能分数: {self.performance_score:.2f}")
        for i in range(self.height):
            row = ""
            for j in range(self.width):
                if (j, i) == self.agent_location:
                    row += "A"  # 代理位置
                elif self.grid[i][j] == EnvironmentState.DIRTY:
                    row += "D"  # 污垢
                else:
                    row += "."  # 干净
                row += " "
            print(row)
        print()


class Agent(ABC):
    """代理基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.performance_history = []
    
    @abstractmethod
    def choose_action(self, percept: Percept) -> Action:
        """选择动作"""
        pass
    
    def update_performance(self, reward: float):
        """更新性能记录"""
        self.performance_history.append(reward)


class SimpleReflexAgent(Agent):
    """简单反射代理
    
    基于当前感知直接选择动作，不考虑历史信息。
    """
    
    def __init__(self):
        super().__init__("简单反射代理")
    
    def choose_action(self, percept: Percept) -> Action:
        """基于简单的条件-动作规则选择动作"""
        if percept.status == EnvironmentState.DIRTY:
            return Action.CLEAN
        else:
            # 随机移动
            return random.choice([Action.MOVE_UP, Action.MOVE_DOWN, 
                                Action.MOVE_LEFT, Action.MOVE_RIGHT])


class ModelBasedReflexAgent(Agent):
    """基于模型的反射代理
    
    维护一个内部状态模型，记录已访问的位置和状态。
    """
    
    def __init__(self, world_width: int, world_height: int):
        super().__init__("基于模型的反射代理")
        self.world_width = world_width
        self.world_height = world_height
        self.world_model = {}  # 记录已知的世界状态
        self.current_location = (0, 0)
        self.visited_clean_locations = set()
    
    def choose_action(self, percept: Percept) -> Action:
        """基于内部模型选择动作"""
        self.current_location = percept.location
        self.world_model[percept.location] = percept.status
        
        if percept.status == EnvironmentState.DIRTY:
            return Action.CLEAN
        else:
            self.visited_clean_locations.add(percept.location)
            # 寻找未访问的位置
            unvisited_locations = self._find_unvisited_locations()
            if unvisited_locations:
                return self._move_towards_location(unvisited_locations[0])
            else:
                # 所有位置都访问过了，寻找可能的脏位置
                return self._explore_random()
    
    def _find_unvisited_locations(self) -> List[Tuple[int, int]]:
        """找到未访问的位置"""
        unvisited = []
        for y in range(self.world_height):
            for x in range(self.world_width):
                if (x, y) not in self.world_model:
                    unvisited.append((x, y))
        return unvisited
    
    def _move_towards_location(self, target: Tuple[int, int]) -> Action:
        """向目标位置移动"""
        current_x, current_y = self.current_location
        target_x, target_y = target
        
        if current_x < target_x:
            return Action.MOVE_RIGHT
        elif current_x > target_x:
            return Action.MOVE_LEFT
        elif current_y < target_y:
            return Action.MOVE_DOWN
        elif current_y > target_y:
            return Action.MOVE_UP
        else:
            return Action.STAY
    
    def _explore_random(self) -> Action:
        """随机探索"""
        return random.choice([Action.MOVE_UP, Action.MOVE_DOWN, 
                            Action.MOVE_LEFT, Action.MOVE_RIGHT])


class GoalBasedAgent(Agent):
    """基于目标的代理
    
    有明确的目标：清理所有污垢，并且会制定计划来达成目标。
    """
    
    def __init__(self, world_width: int, world_height: int):
        super().__init__("基于目标的代理")
        self.world_width = world_width
        self.world_height = world_height
        self.world_model = {}
        self.current_location = (0, 0)
        self.goal = "清理所有污垢"
        self.path_plan = []
        self.dirty_locations = set()
    
    def choose_action(self, percept: Percept) -> Action:
        """基于目标选择动作"""
        self.current_location = percept.location
        self.world_model[percept.location] = percept.status
        
        if percept.status == EnvironmentState.DIRTY:
            self.dirty_locations.add(percept.location)
            return Action.CLEAN
        else:
            # 更新脏位置集合
            if percept.location in self.dirty_locations:
                self.dirty_locations.remove(percept.location)
            
            # 如果有已知的脏位置，制定前往计划
            if self.dirty_locations:
                nearest_dirty = self._find_nearest_dirty_location()
                return self._move_towards_location(nearest_dirty)
            else:
                # 探索未知区域
                return self._explore_systematically()
    
    def _find_nearest_dirty_location(self) -> Tuple[int, int]:
        """找到最近的脏位置"""
        min_distance = float('inf')
        nearest_location = None
        
        for location in self.dirty_locations:
            distance = self._manhattan_distance(self.current_location, location)
            if distance < min_distance:
                min_distance = distance
                nearest_location = location
        
        return nearest_location
    
    def _manhattan_distance(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
        """计算曼哈顿距离"""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    
    def _move_towards_location(self, target: Tuple[int, int]) -> Action:
        """向目标位置移动"""
        current_x, current_y = self.current_location
        target_x, target_y = target
        
        if current_x < target_x:
            return Action.MOVE_RIGHT
        elif current_x > target_x:
            return Action.MOVE_LEFT
        elif current_y < target_y:
            return Action.MOVE_DOWN
        elif current_y > target_y:
            return Action.MOVE_UP
        else:
            return Action.STAY
    
    def _explore_systematically(self) -> Action:
        """系统化探索"""
        # 按行列顺序探索
        for y in range(self.world_height):
            for x in range(self.world_width):
                if (x, y) not in self.world_model:
                    return self._move_towards_location((x, y))
        
        # 如果都探索过了，随机移动
        return random.choice([Action.MOVE_UP, Action.MOVE_DOWN, 
                            Action.MOVE_LEFT, Action.MOVE_RIGHT])


class UtilityBasedAgent(Agent):
    """基于效用的代理
    
    考虑不同动作的效用值，选择效用最大的动作。
    """
    
    def __init__(self, world_width: int, world_height: int):
        super().__init__("基于效用的代理")
        self.world_width = world_width
        self.world_height = world_height
        self.world_model = {}
        self.current_location = (0, 0)
        self.utility_weights = {
            'clean_reward': 10,
            'move_cost': 0.1,
            'time_cost': 0.05,
            'exploration_bonus': 2
        }
    
    def choose_action(self, percept: Percept) -> Action:
        """基于效用选择动作"""
        self.current_location = percept.location
        self.world_model[percept.location] = percept.status
        
        # 计算所有可能动作的效用
        action_utilities = {}
        for action in Action:
            utility = self._calculate_utility(action, percept)
            action_utilities[action] = utility
        
        # 选择效用最大的动作
        best_action = max(action_utilities, key=action_utilities.get)
        return best_action
    
    def _calculate_utility(self, action: Action, percept: Percept) -> float:
        """计算动作的效用值"""
        utility = 0
        
        if action == Action.CLEAN:
            if percept.status == EnvironmentState.DIRTY:
                utility += self.utility_weights['clean_reward']
            else:
                utility -= 5  # 清理干净地方的惩罚
        
        elif action in [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]:
            utility -= self.utility_weights['move_cost']
            
            # 计算移动到新位置的探索奖励
            new_location = self._calculate_new_location(action)
            if self._is_valid_location(new_location):
                if new_location not in self.world_model:
                    utility += self.utility_weights['exploration_bonus']
                elif self.world_model.get(new_location) == EnvironmentState.DIRTY:
                    utility += self.utility_weights['clean_reward'] * 0.5
            else:
                utility -= 10  # 撞墙惩罚
        
        elif action == Action.STAY:
            utility -= self.utility_weights['time_cost']
        
        return utility
    
    def _calculate_new_location(self, action: Action) -> Tuple[int, int]:
        """计算新位置"""
        x, y = self.current_location
        if action == Action.MOVE_UP:
            return (x, y - 1)
        elif action == Action.MOVE_DOWN:
            return (x, y + 1)
        elif action == Action.MOVE_LEFT:
            return (x - 1, y)
        elif action == Action.MOVE_RIGHT:
            return (x + 1, y)
        return (x, y)
    
    def _is_valid_location(self, location: Tuple[int, int]) -> bool:
        """检查位置是否有效"""
        x, y = location
        return 0 <= x < self.world_width and 0 <= y < self.world_height


class LearningAgent(Agent):
    """学习代理
    
    通过经验学习改进性能，使用强化学习的思想。
    """
    
    def __init__(self, world_width: int, world_height: int, learning_rate: float = 0.1):
        super().__init__("学习代理")
        self.world_width = world_width
        self.world_height = world_height
        self.learning_rate = learning_rate
        self.epsilon = 0.1  # 探索率
        self.q_table = {}  # Q值表
        self.current_location = (0, 0)
        self.last_state = None
        self.last_action = None
    
    def choose_action(self, percept: Percept) -> Action:
        """基于Q学习选择动作"""
        self.current_location = percept.location
        state = self._get_state(percept)
        
        # 更新Q值（如果不是第一步）
        if self.last_state is not None and self.last_action is not None:
            self._update_q_value(self.last_state, self.last_action, state, 0)
        
        # 选择动作（ε-贪婪策略）
        if random.random() < self.epsilon:
            # 探索
            action = random.choice(list(Action))
        else:
            # 利用
            action = self._get_best_action(state)
        
        self.last_state = state
        self.last_action = action
        
        return action
    
    def _get_state(self, percept: Percept) -> str:
        """获取状态表示"""
        return f"{percept.location}_{percept.status.value}"
    
    def _get_best_action(self, state: str) -> Action:
        """获取状态下的最佳动作"""
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in Action}
        
        best_action = max(self.q_table[state], key=self.q_table[state].get)
        return best_action
    
    def _update_q_value(self, state: str, action: Action, next_state: str, reward: float):
        """更新Q值"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in Action}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in Action}
        
        # Q学习更新公式
        max_next_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def update_performance(self, reward: float):
        """更新性能记录"""
        super().update_performance(reward)
        # 更新Q值（如果有上一步的信息）
        if self.last_state is not None and self.last_action is not None:
            current_state = self._get_state(Percept(self.current_location, EnvironmentState.CLEAN))
            self._update_q_value(self.last_state, self.last_action, current_state, reward)


def run_simulation(agent: Agent, environment: Environment, max_steps: int = 100, verbose: bool = True):
    """运行仿真"""
    print(f"\n开始运行 {agent.name} 的仿真...")
    
    for step in range(max_steps):
        if environment.is_done():
            break
            
        percept = environment.get_percept()
        action = agent.choose_action(percept)
        new_percept, reward = environment.execute_action(action)
        agent.update_performance(reward)
        
        if verbose and step % 10 == 0:
            print(f"步骤 {step}: {percept}, 动作: {action.value}, 奖励: {reward:.2f}")
            environment.print_environment()
    
    print(f"{agent.name} 完成! 总性能分数: {environment.performance_score:.2f}")
    return environment.performance_score


def compare_agents():
    """比较不同类型代理的性能"""
    print("=== 智能代理性能比较 ===")
    
    # 设置参数
    world_width, world_height = 4, 4
    num_trials = 5
    
    # 创建代理
    agents = [
        SimpleReflexAgent(),
        ModelBasedReflexAgent(world_width, world_height),
        GoalBasedAgent(world_width, world_height),
        UtilityBasedAgent(world_width, world_height),
        LearningAgent(world_width, world_height)
    ]
    
    # 运行试验
    results = {}
    for agent in agents:
        scores = []
        for trial in range(num_trials):
            env = Environment(world_width, world_height, dirt_prob=0.3)
            score = run_simulation(agent, env, max_steps=50, verbose=False)
            scores.append(score)
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        results[agent.name] = (avg_score, std_score)
        print(f"{agent.name}: 平均分数 {avg_score:.2f} ± {std_score:.2f}")
    
    # 找出最佳代理
    best_agent = max(results, key=lambda x: results[x][0])
    print(f"\n最佳代理: {best_agent}")
    
    return results


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 单个代理演示
    print("=== 单个代理演示 ===")
    env = Environment(4, 4, dirt_prob=0.3)
    agent = GoalBasedAgent(4, 4)
    run_simulation(agent, env, max_steps=30, verbose=True)
    
    # 代理比较
    print("\n" + "="*50)
    compare_agents() 