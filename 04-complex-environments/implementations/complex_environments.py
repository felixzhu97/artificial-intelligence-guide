"""
第4章：复杂环境中的搜索
实现了AIMA第4章中的复杂环境概念：部分可观察、随机性、多代理等
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set, Optional, Any, Callable
import random
import numpy as np
import time
from collections import defaultdict, deque
from enum import Enum
import math


class EnvironmentType(Enum):
    """环境类型枚举"""
    FULLY_OBSERVABLE = "fully_observable"
    PARTIALLY_OBSERVABLE = "partially_observable"
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    EPISODIC = "episodic"
    SEQUENTIAL = "sequential"
    STATIC = "static"
    DYNAMIC = "dynamic"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"


class Percept:
    """感知类：代理从环境中接收到的信息"""
    
    def __init__(self, data: Any, timestamp: float = None):
        self.data = data
        self.timestamp = timestamp or time.time()
    
    def __str__(self):
        return f"Percept({self.data})"
    
    def __repr__(self):
        return f"Percept({self.data}, t={self.timestamp})"


class Environment(ABC):
    """抽象环境类"""
    
    def __init__(self, environment_types: List[EnvironmentType]):
        self.types = environment_types
        self.agents = []
        self.time = 0
        self.performance_measures = {}
    
    @abstractmethod
    def percept(self, agent_id: str) -> Percept:
        """为指定代理生成感知"""
        pass
    
    @abstractmethod
    def execute_action(self, agent_id: str, action: Any) -> bool:
        """执行代理的动作，返回是否成功"""
        pass
    
    @abstractmethod
    def is_done(self) -> bool:
        """环境是否结束"""
        pass
    
    def add_agent(self, agent):
        """添加代理到环境"""
        self.agents.append(agent)
        self.performance_measures[agent.agent_id] = 0
    
    def step(self):
        """环境前进一步"""
        self.time += 1


class VacuumEnvironment(Environment):
    """吸尘器环境：经典的两格房间清洁问题"""
    
    def __init__(self, dirt_prob: float = 0.1):
        super().__init__([
            EnvironmentType.FULLY_OBSERVABLE,
            EnvironmentType.STOCHASTIC,
            EnvironmentType.SEQUENTIAL,
            EnvironmentType.DYNAMIC
        ])
        
        self.rooms = ['A', 'B']
        self.dirt_status = {room: random.random() < 0.5 for room in self.rooms}
        self.agent_location = random.choice(self.rooms)
        self.dirt_prob = dirt_prob  # 每步产生新污垢的概率
        self.actions_taken = 0
        self.max_actions = 1000
    
    def percept(self, agent_id: str) -> Percept:
        """返回代理的感知：[位置, 是否有污垢]"""
        location = self.agent_location
        dirt = self.dirt_status[location]
        return Percept([location, dirt])
    
    def execute_action(self, agent_id: str, action: str) -> bool:
        """执行动作：Left, Right, Suck, NoOp"""
        self.actions_taken += 1
        
        if action == 'Right':
            self.agent_location = 'B'
        elif action == 'Left':
            self.agent_location = 'A'
        elif action == 'Suck':
            if self.dirt_status[self.agent_location]:
                self.dirt_status[self.agent_location] = False
                self.performance_measures[agent_id] += 10  # 清洁污垢获得奖励
        elif action == 'NoOp':
            pass  # 无操作
        else:
            return False
        
        # 动态环境：随机产生新污垢
        for room in self.rooms:
            if not self.dirt_status[room] and random.random() < self.dirt_prob:
                self.dirt_status[room] = True
        
        # 移动消耗能量
        if action in ['Left', 'Right']:
            self.performance_measures[agent_id] -= 1
        
        return True
    
    def is_done(self) -> bool:
        """当达到最大动作数或清洁完成时结束"""
        return self.actions_taken >= self.max_actions or not any(self.dirt_status.values())
    
    def get_state(self) -> Dict:
        """获取环境状态"""
        return {
            'agent_location': self.agent_location,
            'dirt_status': self.dirt_status.copy(),
            'actions_taken': self.actions_taken
        }


class WumpusWorld(Environment):
    """Wumpus世界：部分可观察的危险探索环境"""
    
    def __init__(self, size: int = 4):
        super().__init__([
            EnvironmentType.PARTIALLY_OBSERVABLE,
            EnvironmentType.DETERMINISTIC,
            EnvironmentType.SEQUENTIAL,
            EnvironmentType.STATIC
        ])
        
        self.size = size
        self.agent_pos = (0, 0)  # 代理位置
        self.agent_direction = 0  # 0:North, 1:East, 2:South, 3:West
        self.agent_has_arrow = True
        self.agent_alive = True
        self.has_gold = False
        
        # 随机生成世界
        self._generate_world()
        
        # 感知历史
        self.percept_history = []
    
    def _generate_world(self):
        """随机生成Wumpus世界"""
        # 初始化空世界
        self.pits = set()
        self.gold_pos = None
        self.wumpus_pos = None
        self.wumpus_alive = True
        
        # 随机放置坑洞（避开起始位置）
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) != (0, 0) and random.random() < 0.2:
                    self.pits.add((i, j))
        
        # 随机放置金子
        while True:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (x, y) != (0, 0) and (x, y) not in self.pits:
                self.gold_pos = (x, y)
                break
        
        # 随机放置Wumpus
        while True:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (x, y) != (0, 0) and (x, y) not in self.pits and (x, y) != self.gold_pos:
                self.wumpus_pos = (x, y)
                break
    
    def percept(self, agent_id: str) -> Percept:
        """生成部分可观察的感知"""
        x, y = self.agent_pos
        
        # 基本感知
        breeze = any((x+dx, y+dy) in self.pits 
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
                    if 0 <= x+dx < self.size and 0 <= y+dy < self.size)
        
        stench = (self.wumpus_alive and 
                 any((x+dx, y+dy) == self.wumpus_pos
                     for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]))
        
        glitter = (self.agent_pos == self.gold_pos)
        bump = False  # 只有撞墙时才为True
        scream = False  # 只有射中Wumpus时才为True
        
        percept_data = {
            'breeze': breeze,
            'stench': stench,
            'glitter': glitter,
            'bump': bump,
            'scream': scream
        }
        
        percept = Percept(percept_data)
        self.percept_history.append(percept)
        return percept
    
    def execute_action(self, agent_id: str, action: str) -> bool:
        """执行动作"""
        if not self.agent_alive:
            return False
        
        x, y = self.agent_pos
        
        if action == 'Forward':
            # 计算新位置
            dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_direction]
            new_x, new_y = x + dx, y + dy
            
            # 检查边界
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                self.agent_pos = (new_x, new_y)
                
                # 检查危险
                if self.agent_pos in self.pits:
                    self.agent_alive = False
                    self.performance_measures[agent_id] -= 1000
                    return True
                
                if self.agent_pos == self.wumpus_pos and self.wumpus_alive:
                    self.agent_alive = False
                    self.performance_measures[agent_id] -= 1000
                    return True
            else:
                # 撞墙
                return True
        
        elif action == 'TurnLeft':
            self.agent_direction = (self.agent_direction - 1) % 4
        
        elif action == 'TurnRight':
            self.agent_direction = (self.agent_direction + 1) % 4
        
        elif action == 'Grab':
            if self.agent_pos == self.gold_pos:
                self.has_gold = True
                self.performance_measures[agent_id] += 1000
        
        elif action == 'Shoot':
            if self.agent_has_arrow:
                self.agent_has_arrow = False
                self.performance_measures[agent_id] -= 10
                
                # 检查是否射中Wumpus
                dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_direction]
                arrow_x, arrow_y = x, y
                
                while 0 <= arrow_x < self.size and 0 <= arrow_y < self.size:
                    arrow_x += dx
                    arrow_y += dy
                    if (arrow_x, arrow_y) == self.wumpus_pos and self.wumpus_alive:
                        self.wumpus_alive = False
                        return True
        
        elif action == 'Climb':
            if self.agent_pos == (0, 0):
                return True  # 成功爬出
        
        # 每个动作消耗1点
        self.performance_measures[agent_id] -= 1
        return True
    
    def is_done(self) -> bool:
        """检查游戏是否结束"""
        return not self.agent_alive or (self.has_gold and self.agent_pos == (0, 0))


class GridWorld(Environment):
    """网格世界：强化学习的经典环境"""
    
    def __init__(self, width: int = 4, height: int = 4):
        super().__init__([
            EnvironmentType.FULLY_OBSERVABLE,
            EnvironmentType.STOCHASTIC,
            EnvironmentType.SEQUENTIAL,
            EnvironmentType.STATIC
        ])
        
        self.width = width
        self.height = height
        self.agent_pos = (0, 0)
        self.goal_pos = (width-1, height-1)
        self.obstacles = {(1, 1), (2, 2)}  # 障碍物位置
        self.step_count = 0
        self.max_steps = 100
    
    def percept(self, agent_id: str) -> Percept:
        """返回代理的位置和目标信息"""
        return Percept({
            'position': self.agent_pos,
            'goal': self.goal_pos,
            'step': self.step_count
        })
    
    def execute_action(self, agent_id: str, action: str) -> bool:
        """执行移动动作"""
        x, y = self.agent_pos
        
        # 动作映射
        moves = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        
        if action in moves:
            dx, dy = moves[action]
            new_x, new_y = x + dx, y + dy
            
            # 检查边界和障碍物
            if (0 <= new_x < self.width and 
                0 <= new_y < self.height and 
                (new_x, new_y) not in self.obstacles):
                
                # 随机性：有10%概率动作失败
                if random.random() > 0.1:
                    self.agent_pos = (new_x, new_y)
        
        self.step_count += 1
        
        # 奖励计算
        if self.agent_pos == self.goal_pos:
            self.performance_measures[agent_id] += 100
        else:
            self.performance_measures[agent_id] -= 1  # 每步的时间惩罚
        
        return True
    
    def is_done(self) -> bool:
        """检查是否到达目标或超时"""
        return self.agent_pos == self.goal_pos or self.step_count >= self.max_steps


class MultiAgentEnvironment(Environment):
    """多代理环境：竞争性资源收集"""
    
    def __init__(self, width: int = 8, height: int = 8):
        super().__init__([
            EnvironmentType.FULLY_OBSERVABLE,
            EnvironmentType.DETERMINISTIC,
            EnvironmentType.SEQUENTIAL,
            EnvironmentType.MULTI_AGENT
        ])
        
        self.width = width
        self.height = height
        self.agent_positions = {}
        self.resources = set()
        self.step_count = 0
        self.max_steps = 200
        
        # 初始化资源
        self._generate_resources()
    
    def _generate_resources(self):
        """生成随机资源位置"""
        num_resources = self.width * self.height // 8
        while len(self.resources) < num_resources:
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            self.resources.add((x, y))
    
    def add_agent(self, agent):
        """添加代理并分配随机位置"""
        super().add_agent(agent)
        
        # 分配随机起始位置
        while True:
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            if (x, y) not in self.agent_positions.values():
                self.agent_positions[agent.agent_id] = (x, y)
                break
    
    def percept(self, agent_id: str) -> Percept:
        """为代理生成感知"""
        agent_pos = self.agent_positions[agent_id]
        
        # 观察周围环境
        visible_range = 2
        x, y = agent_pos
        
        visible_resources = []
        visible_agents = []
        
        for i in range(max(0, x-visible_range), min(self.width, x+visible_range+1)):
            for j in range(max(0, y-visible_range), min(self.height, y+visible_range+1)):
                if (i, j) in self.resources:
                    visible_resources.append((i, j))
                
                for other_id, other_pos in self.agent_positions.items():
                    if other_id != agent_id and other_pos == (i, j):
                        visible_agents.append((other_id, i, j))
        
        return Percept({
            'position': agent_pos,
            'visible_resources': visible_resources,
            'visible_agents': visible_agents,
            'step': self.step_count
        })
    
    def execute_action(self, agent_id: str, action: str) -> bool:
        """执行代理动作"""
        if agent_id not in self.agent_positions:
            return False
        
        x, y = self.agent_positions[agent_id]
        
        # 移动动作
        moves = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        
        if action in moves:
            dx, dy = moves[action]
            new_x, new_y = x + dx, y + dy
            
            # 检查边界
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                # 检查是否与其他代理冲突
                if (new_x, new_y) not in self.agent_positions.values():
                    self.agent_positions[agent_id] = (new_x, new_y)
        
        elif action == 'collect':
            # 收集资源
            if (x, y) in self.resources:
                self.resources.remove((x, y))
                self.performance_measures[agent_id] += 10
        
        return True
    
    def is_done(self) -> bool:
        """检查环境是否结束"""
        return len(self.resources) == 0 or self.step_count >= self.max_steps
    
    def step(self):
        """多代理环境的步进"""
        super().step()
        
        # 添加新资源（模拟动态环境）
        if random.random() < 0.05:  # 5%概率生成新资源
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            if (x, y) not in self.agent_positions.values():
                self.resources.add((x, y))


class SimpleAgent:
    """简单反应式代理"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.percept_history = []
    
    def program(self, percept: Percept) -> str:
        """代理程序：根据感知选择动作"""
        self.percept_history.append(percept)
        return "NoOp"  # 默认无操作


class SimpleReflexVacuumAgent(SimpleAgent):
    """简单反射吸尘器代理"""
    
    def program(self, percept: Percept) -> str:
        location, dirt = percept.data
        
        if dirt:
            return 'Suck'
        elif location == 'A':
            return 'Right'
        elif location == 'B':
            return 'Left'
        else:
            return 'NoOp'


class RandomAgent(SimpleAgent):
    """随机动作代理"""
    
    def __init__(self, agent_id: str, actions: List[str]):
        super().__init__(agent_id)
        self.actions = actions
    
    def program(self, percept: Percept) -> str:
        return random.choice(self.actions)


def demonstrate_vacuum_environment():
    """演示吸尘器环境"""
    print("=== 吸尘器环境演示 ===")
    
    env = VacuumEnvironment(dirt_prob=0.1)
    agent = SimpleReflexVacuumAgent("vacuum_agent")
    env.add_agent(agent)
    
    print("初始状态:", env.get_state())
    
    steps = 0
    while not env.is_done() and steps < 20:
        percept = env.percept(agent.agent_id)
        action = agent.program(percept)
        env.execute_action(agent.agent_id, action)
        env.step()
        
        print(f"步骤 {steps+1}: 感知={percept.data}, 动作={action}, 状态={env.get_state()}")
        steps += 1
    
    print(f"最终性能: {env.performance_measures[agent.agent_id]}")


def demonstrate_wumpus_world():
    """演示Wumpus世界"""
    print("\n=== Wumpus世界演示 ===")
    
    env = WumpusWorld(size=4)
    agent = RandomAgent("explorer", ['Forward', 'TurnLeft', 'TurnRight', 'Grab'])
    env.add_agent(agent)
    
    print(f"世界设置: 金子位置={env.gold_pos}, Wumpus位置={env.wumpus_pos}")
    print(f"坑洞位置: {env.pits}")
    
    steps = 0
    while not env.is_done() and steps < 30:
        percept = env.percept(agent.agent_id)
        action = agent.program(percept)
        env.execute_action(agent.agent_id, action)
        env.step()
        
        print(f"步骤 {steps+1}: 位置={env.agent_pos}, 方向={env.agent_direction}")
        print(f"  感知={percept.data}")
        print(f"  动作={action}, 存活={env.agent_alive}")
        
        if not env.agent_alive:
            print("  代理死亡！")
            break
            
        steps += 1
    
    print(f"最终性能: {env.performance_measures[agent.agent_id]}")


def demonstrate_multi_agent():
    """演示多代理环境"""
    print("\n=== 多代理环境演示 ===")
    
    env = MultiAgentEnvironment(width=6, height=6)
    
    # 添加多个代理
    agents = []
    for i in range(3):
        agent = RandomAgent(f"agent_{i}", ['up', 'down', 'left', 'right', 'collect'])
        agents.append(agent)
        env.add_agent(agent)
    
    print(f"初始资源数量: {len(env.resources)}")
    print(f"代理位置: {env.agent_positions}")
    
    steps = 0
    while not env.is_done() and steps < 50:
        print(f"\n--- 步骤 {steps+1} ---")
        
        # 所有代理同时行动
        for agent in agents:
            if agent.agent_id in env.agent_positions:
                percept = env.percept(agent.agent_id)
                action = agent.program(percept)
                env.execute_action(agent.agent_id, action)
        
        env.step()
        
        print(f"代理位置: {env.agent_positions}")
        print(f"剩余资源: {len(env.resources)}")
        print(f"性能分数: {env.performance_measures}")
        
        steps += 1
    
    print(f"\n最终性能分数: {env.performance_measures}")
    winner = max(env.performance_measures, key=env.performance_measures.get)
    print(f"获胜者: {winner}")


def main():
    """主演示函数"""
    print("第4章：复杂环境中的搜索")
    print("展示了不同类型的环境和相应的代理设计")
    
    demonstrate_vacuum_environment()
    demonstrate_wumpus_world()
    demonstrate_multi_agent()
    
    print("\n=== 环境类型总结 ===")
    print("1. 完全可观察 vs 部分可观察：代理能否感知完整的环境状态")
    print("2. 确定性 vs 随机性：动作结果是否确定")
    print("3. 静态 vs 动态：环境是否会自主变化")
    print("4. 单代理 vs 多代理：是否存在其他智能代理")
    print("5. 不同环境类型需要不同的代理设计策略")


if __name__ == "__main__":
    main() 