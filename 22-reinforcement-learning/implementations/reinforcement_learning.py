"""
强化学习案例实现

本模块演示了《人工智能：现代方法》第21章中的强化学习算法：
1. Q学习 (Q-Learning)
2. SARSA
3. 策略梯度
4. 深度Q网络 (DQN)
5. 蒙特卡洛方法
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import random
import seaborn as sns
from dataclasses import dataclass
import time


@dataclass
class State:
    """状态表示"""
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


@dataclass
class Action:
    """动作表示"""
    name: str
    dx: int
    dy: int


class Environment(ABC):
    """环境基类"""
    
    @abstractmethod
    def reset(self) -> Any:
        """重置环境"""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """执行动作"""
        pass
    
    @abstractmethod
    def get_valid_actions(self, state: Any) -> List[Any]:
        """获取有效动作"""
        pass


class GridWorld(Environment):
    """网格世界环境"""
    
    def __init__(self, width: int = 5, height: int = 5, 
                 goal_reward: float = 10, step_penalty: float = -0.1,
                 wall_penalty: float = -1):
        self.width = width
        self.height = height
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        
        # 定义动作
        self.actions = [
            Action("UP", 0, -1),
            Action("DOWN", 0, 1),
            Action("LEFT", -1, 0),
            Action("RIGHT", 1, 0)
        ]
        
        # 设置目标和障碍物
        self.goal = State(width - 1, height - 1)
        self.obstacles = set()
        self._setup_obstacles()
        
        # 当前状态
        self.current_state = None
        self.reset()
    
    def _setup_obstacles(self):
        """设置障碍物"""
        # 添加一些障碍物
        obstacles = [
            (1, 1), (1, 2), (2, 1),
            (3, 2), (3, 3)
        ]
        for x, y in obstacles:
            if x < self.width and y < self.height:
                self.obstacles.add(State(x, y))
    
    def reset(self) -> State:
        """重置环境"""
        self.current_state = State(0, 0)
        return self.current_state
    
    def step(self, action: Action) -> Tuple[State, float, bool, Dict]:
        """执行动作"""
        new_x = self.current_state.x + action.dx
        new_y = self.current_state.y + action.dy
        
        # 检查边界
        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            return self.current_state, self.wall_penalty, False, {}
        
        new_state = State(new_x, new_y)
        
        # 检查障碍物
        if new_state in self.obstacles:
            return self.current_state, self.wall_penalty, False, {}
        
        # 更新状态
        self.current_state = new_state
        
        # 检查是否到达目标
        if self.current_state == self.goal:
            return self.current_state, self.goal_reward, True, {}
        
        return self.current_state, self.step_penalty, False, {}
    
    def get_valid_actions(self, state: State) -> List[Action]:
        """获取有效动作"""
        valid_actions = []
        for action in self.actions:
            new_x = state.x + action.dx
            new_y = state.y + action.dy
            
            if (0 <= new_x < self.width and 0 <= new_y < self.height and
                State(new_x, new_y) not in self.obstacles):
                valid_actions.append(action)
        
        return valid_actions
    
    def render(self, values: Dict[State, float] = None, policy: Dict[State, Action] = None):
        """渲染环境"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制网格
        for i in range(self.width + 1):
            ax.axvline(x=i, color='black', linewidth=1)
        for i in range(self.height + 1):
            ax.axhline(y=i, color='black', linewidth=1)
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            rect = plt.Rectangle((obstacle.x, obstacle.y), 1, 1, 
                               facecolor='black', alpha=0.8)
            ax.add_patch(rect)
        
        # 绘制目标
        rect = plt.Rectangle((self.goal.x, self.goal.y), 1, 1, 
                           facecolor='gold', alpha=0.8)
        ax.add_patch(rect)
        
        # 绘制当前状态
        if self.current_state:
            circle = plt.Circle((self.current_state.x + 0.5, self.current_state.y + 0.5), 
                              0.3, color='red', alpha=0.8)
            ax.add_patch(circle)
        
        # 绘制值函数
        if values:
            for state, value in values.items():
                if state not in self.obstacles:
                    color_intensity = min(abs(value) / 10, 1)
                    color = 'lightblue' if value > 0 else 'lightcoral'
                    rect = plt.Rectangle((state.x, state.y), 1, 1, 
                                       facecolor=color, alpha=color_intensity)
                    ax.add_patch(rect)
                    ax.text(state.x + 0.5, state.y + 0.5, f'{value:.2f}', 
                           ha='center', va='center', fontsize=8)
        
        # 绘制策略
        if policy:
            arrow_map = {
                "UP": (0, -0.3),
                "DOWN": (0, 0.3),
                "LEFT": (-0.3, 0),
                "RIGHT": (0.3, 0)
            }
            
            for state, action in policy.items():
                if state not in self.obstacles and state != self.goal:
                    dx, dy = arrow_map[action.name]
                    ax.arrow(state.x + 0.5, state.y + 0.5, dx, dy, 
                           head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title('Grid World Environment')
        plt.tight_layout()
        plt.show()


class QLearningAgent:
    """Q学习代理"""
    
    def __init__(self, actions: List[Action], learning_rate: float = 0.1, 
                 discount_factor: float = 0.9, epsilon: float = 0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.training_rewards = []
        self.training_steps = []
    
    def choose_action(self, state: State, valid_actions: List[Action]) -> Action:
        """选择动作（ε-贪婪策略）"""
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # 贪婪选择
        q_values = {action: self.q_table[state][action] for action in valid_actions}
        if not q_values:
            return random.choice(valid_actions)
        
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state: State, action: Action, reward: float, 
                      next_state: State, next_valid_actions: List[Action]):
        """更新Q值"""
        # 计算下一状态的最大Q值
        if next_valid_actions:
            max_next_q = max(self.q_table[next_state][a] for a in next_valid_actions)
        else:
            max_next_q = 0
        
        # Q学习更新公式
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q
    
    def train(self, env: Environment, episodes: int = 1000):
        """训练代理"""
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                valid_actions = env.get_valid_actions(state)
                if not valid_actions:
                    break
                
                action = self.choose_action(state, valid_actions)
                next_state, reward, done, _ = env.step(action)
                
                next_valid_actions = env.get_valid_actions(next_state)
                self.update_q_value(state, action, reward, next_state, next_valid_actions)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done or steps > 200:
                    break
            
            self.training_rewards.append(total_reward)
            self.training_steps.append(steps)
            
            # 衰减探索率
            if episode % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.95)
    
    def get_policy(self, env: Environment) -> Dict[State, Action]:
        """获取策略"""
        policy = {}
        
        for x in range(env.width):
            for y in range(env.height):
                state = State(x, y)
                if state in env.obstacles or state == env.goal:
                    continue
                
                valid_actions = env.get_valid_actions(state)
                if valid_actions:
                    q_values = {action: self.q_table[state][action] for action in valid_actions}
                    best_action = max(q_values, key=q_values.get)
                    policy[state] = best_action
        
        return policy
    
    def get_value_function(self, env: Environment) -> Dict[State, float]:
        """获取值函数"""
        values = {}
        
        for x in range(env.width):
            for y in range(env.height):
                state = State(x, y)
                if state in env.obstacles:
                    continue
                
                valid_actions = env.get_valid_actions(state)
                if valid_actions:
                    max_q = max(self.q_table[state][action] for action in valid_actions)
                    values[state] = max_q
                else:
                    values[state] = 0
        
        return values


class SarsaAgent:
    """SARSA代理"""
    
    def __init__(self, actions: List[Action], learning_rate: float = 0.1,
                 discount_factor: float = 0.9, epsilon: float = 0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.training_rewards = []
        self.training_steps = []
    
    def choose_action(self, state: State, valid_actions: List[Action]) -> Action:
        """选择动作（ε-贪婪策略）"""
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        q_values = {action: self.q_table[state][action] for action in valid_actions}
        if not q_values:
            return random.choice(valid_actions)
        
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state: State, action: Action, reward: float,
                      next_state: State, next_action: Action):
        """更新Q值"""
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action] if next_action else 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        )
        self.q_table[state][action] = new_q
    
    def train(self, env: Environment, episodes: int = 1000):
        """训练代理"""
        for episode in range(episodes):
            state = env.reset()
            valid_actions = env.get_valid_actions(state)
            
            if not valid_actions:
                continue
            
            action = self.choose_action(state, valid_actions)
            total_reward = 0
            steps = 0
            
            while True:
                next_state, reward, done, _ = env.step(action)
                next_valid_actions = env.get_valid_actions(next_state)
                
                if next_valid_actions and not done:
                    next_action = self.choose_action(next_state, next_valid_actions)
                else:
                    next_action = None
                
                self.update_q_value(state, action, reward, next_state, next_action)
                
                state = next_state
                action = next_action
                total_reward += reward
                steps += 1
                
                if done or steps > 200 or not next_valid_actions:
                    break
            
            self.training_rewards.append(total_reward)
            self.training_steps.append(steps)
            
            # 衰减探索率
            if episode % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.95)
    
    def get_policy(self, env: Environment) -> Dict[State, Action]:
        """获取策略"""
        policy = {}
        
        for x in range(env.width):
            for y in range(env.height):
                state = State(x, y)
                if state in env.obstacles or state == env.goal:
                    continue
                
                valid_actions = env.get_valid_actions(state)
                if valid_actions:
                    q_values = {action: self.q_table[state][action] for action in valid_actions}
                    best_action = max(q_values, key=q_values.get)
                    policy[state] = best_action
        
        return policy
    
    def get_value_function(self, env: Environment) -> Dict[State, float]:
        """获取值函数"""
        values = {}
        
        for x in range(env.width):
            for y in range(env.height):
                state = State(x, y)
                if state in env.obstacles:
                    continue
                
                valid_actions = env.get_valid_actions(state)
                if valid_actions:
                    max_q = max(self.q_table[state][action] for action in valid_actions)
                    values[state] = max_q
                else:
                    values[state] = 0
        
        return values


class MonteCarloAgent:
    """蒙特卡洛代理"""
    
    def __init__(self, actions: List[Action], discount_factor: float = 0.9,
                 epsilon: float = 0.1):
        self.actions = actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.returns = defaultdict(lambda: defaultdict(list))
        self.training_rewards = []
        self.training_steps = []
    
    def choose_action(self, state: State, valid_actions: List[Action]) -> Action:
        """选择动作（ε-贪婪策略）"""
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        q_values = {action: self.q_table[state][action] for action in valid_actions}
        if not q_values:
            return random.choice(valid_actions)
        
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)
    
    def generate_episode(self, env: Environment) -> List[Tuple[State, Action, float]]:
        """生成一个完整的回合"""
        episode = []
        state = env.reset()
        steps = 0
        
        while True:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break
            
            action = self.choose_action(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            
            episode.append((state, action, reward))
            state = next_state
            steps += 1
            
            if done or steps > 200:
                break
        
        return episode
    
    def train(self, env: Environment, episodes: int = 1000):
        """训练代理"""
        for episode_num in range(episodes):
            episode = self.generate_episode(env)
            
            if not episode:
                continue
            
            total_reward = sum(reward for _, _, reward in episode)
            self.training_rewards.append(total_reward)
            self.training_steps.append(len(episode))
            
            # 计算每个状态-动作对的回报
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.discount_factor * G + reward
                
                # 首次访问蒙特卡洛
                if (state, action) not in [(s, a) for s, a, _ in episode[:t]]:
                    self.returns[state][action].append(G)
                    self.q_table[state][action] = np.mean(self.returns[state][action])
            
            # 衰减探索率
            if episode_num % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.95)
    
    def get_policy(self, env: Environment) -> Dict[State, Action]:
        """获取策略"""
        policy = {}
        
        for x in range(env.width):
            for y in range(env.height):
                state = State(x, y)
                if state in env.obstacles or state == env.goal:
                    continue
                
                valid_actions = env.get_valid_actions(state)
                if valid_actions:
                    q_values = {action: self.q_table[state][action] for action in valid_actions}
                    best_action = max(q_values, key=q_values.get)
                    policy[state] = best_action
        
        return policy
    
    def get_value_function(self, env: Environment) -> Dict[State, float]:
        """获取值函数"""
        values = {}
        
        for x in range(env.width):
            for y in range(env.height):
                state = State(x, y)
                if state in env.obstacles:
                    continue
                
                valid_actions = env.get_valid_actions(state)
                if valid_actions:
                    max_q = max(self.q_table[state][action] for action in valid_actions)
                    values[state] = max_q
                else:
                    values[state] = 0
        
        return values


def plot_training_progress(agents: Dict[str, Any], title: str = "训练进度"):
    """绘制训练进度"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 平滑函数
    def smooth(data, window_size=50):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # 绘制奖励
    for name, agent in agents.items():
        if agent.training_rewards:
            smoothed_rewards = smooth(agent.training_rewards)
            ax1.plot(smoothed_rewards, label=name, linewidth=2)
    
    ax1.set_xlabel('回合')
    ax1.set_ylabel('平均奖励')
    ax1.set_title('训练奖励')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制步数
    for name, agent in agents.items():
        if agent.training_steps:
            smoothed_steps = smooth(agent.training_steps)
            ax2.plot(smoothed_steps, label=name, linewidth=2)
    
    ax2.set_xlabel('回合')
    ax2.set_ylabel('平均步数')
    ax2.set_title('训练步数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def compare_rl_algorithms():
    """比较不同强化学习算法"""
    print("=== 强化学习算法比较 ===")
    
    # 创建环境
    env = GridWorld(width=6, height=6)
    
    # 创建代理
    agents = {
        'Q-Learning': QLearningAgent(env.actions, learning_rate=0.1, epsilon=0.1),
        'SARSA': SarsaAgent(env.actions, learning_rate=0.1, epsilon=0.1),
        'Monte Carlo': MonteCarloAgent(env.actions, epsilon=0.1)
    }
    
    # 训练代理
    episodes = 1000
    for name, agent in agents.items():
        print(f"\n训练 {name}...")
        start_time = time.time()
        agent.train(env, episodes)
        training_time = time.time() - start_time
        
        final_reward = np.mean(agent.training_rewards[-100:])
        final_steps = np.mean(agent.training_steps[-100:])
        
        print(f"  训练时间: {training_time:.2f}s")
        print(f"  最终100回合平均奖励: {final_reward:.2f}")
        print(f"  最终100回合平均步数: {final_steps:.2f}")
    
    # 绘制训练进度
    plot_training_progress(agents, "强化学习算法比较")
    
    # 可视化最终策略
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, (name, agent) in enumerate(agents.items()):
        ax = axes[i]
        
        # 获取值函数和策略
        values = agent.get_value_function(env)
        policy = agent.get_policy(env)
        
        # 绘制值函数热力图
        value_matrix = np.zeros((env.height, env.width))
        for x in range(env.width):
            for y in range(env.height):
                state = State(x, y)
                if state in values:
                    value_matrix[y, x] = values[state]
                elif state in env.obstacles:
                    value_matrix[y, x] = np.nan
        
        im = ax.imshow(value_matrix, cmap='RdYlBu', alpha=0.7)
        
        # 绘制策略箭头
        arrow_map = {
            "UP": (0, -0.3),
            "DOWN": (0, 0.3),
            "LEFT": (-0.3, 0),
            "RIGHT": (0.3, 0)
        }
        
        for state, action in policy.items():
            if state not in env.obstacles and state != env.goal:
                dx, dy = arrow_map[action.name]
                ax.arrow(state.x, state.y, dx, dy, 
                        head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # 标记障碍物和目标
        for obstacle in env.obstacles:
            ax.add_patch(plt.Rectangle((obstacle.x - 0.5, obstacle.y - 0.5), 1, 1, 
                                     facecolor='black', alpha=0.8))
        
        ax.add_patch(plt.Rectangle((env.goal.x - 0.5, env.goal.y - 0.5), 1, 1, 
                                 facecolor='gold', alpha=0.8))
        
        ax.set_xlim(-0.5, env.width - 0.5)
        ax.set_ylim(-0.5, env.height - 0.5)
        ax.set_title(f'{name} - 值函数和策略')
        ax.set_aspect('equal')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏空的子图
    if len(agents) < 4:
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return agents


def demo_q_learning():
    """Q学习演示"""
    print("=== Q学习演示 ===")
    
    # 创建环境
    env = GridWorld(width=5, height=5)
    
    # 创建Q学习代理
    agent = QLearningAgent(env.actions, learning_rate=0.1, 
                          discount_factor=0.9, epsilon=0.1)
    
    print("环境设置:")
    print(f"  网格大小: {env.width}x{env.height}")
    print(f"  起始位置: (0, 0)")
    print(f"  目标位置: ({env.goal.x}, {env.goal.y})")
    print(f"  障碍物数量: {len(env.obstacles)}")
    
    # 显示初始环境
    print("\n初始环境:")
    env.render()
    
    # 训练代理
    print("\n开始训练...")
    episodes = 500
    agent.train(env, episodes)
    
    # 显示训练结果
    final_rewards = agent.training_rewards[-50:]
    final_steps = agent.training_steps[-50:]
    
    print(f"\n训练完成! ({episodes} 回合)")
    print(f"最后50回合平均奖励: {np.mean(final_rewards):.2f}")
    print(f"最后50回合平均步数: {np.mean(final_steps):.2f}")
    
    # 获取最终策略和值函数
    policy = agent.get_policy(env)
    values = agent.get_value_function(env)
    
    print("\n最终策略:")
    env.render(values=values, policy=policy)
    
    # 测试训练后的代理
    print("\n测试训练后的代理:")
    test_episodes = 10
    test_rewards = []
    test_steps = []
    
    for i in range(test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break
            
            # 贪婪策略（不探索）
            q_values = {action: agent.q_table[state][action] for action in valid_actions}
            best_action = max(q_values, key=q_values.get)
            
            state, reward, done, _ = env.step(best_action)
            total_reward += reward
            steps += 1
            
            if done or steps > 50:
                break
        
        test_rewards.append(total_reward)
        test_steps.append(steps)
    
    print(f"测试结果 ({test_episodes} 回合):")
    print(f"  平均奖励: {np.mean(test_rewards):.2f}")
    print(f"  平均步数: {np.mean(test_steps):.2f}")
    print(f"  成功率: {np.mean([r > 0 for r in test_rewards]):.2%}")
    
    # 绘制训练进度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 平滑奖励
    window_size = 20
    smoothed_rewards = np.convolve(agent.training_rewards, 
                                  np.ones(window_size)/window_size, mode='valid')
    ax1.plot(smoothed_rewards, color='blue', linewidth=2)
    ax1.set_xlabel('回合')
    ax1.set_ylabel('平均奖励')
    ax1.set_title('训练奖励')
    ax1.grid(True, alpha=0.3)
    
    # 平滑步数
    smoothed_steps = np.convolve(agent.training_steps, 
                                np.ones(window_size)/window_size, mode='valid')
    ax2.plot(smoothed_steps, color='red', linewidth=2)
    ax2.set_xlabel('回合')
    ax2.set_ylabel('平均步数')
    ax2.set_title('训练步数')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return agent


if __name__ == "__main__":
    # Q学习演示
    print("运行Q学习演示...")
    q_agent = demo_q_learning()
    
    # 算法比较
    print("\n" + "="*50)
    print("运行算法比较...")
    agents = compare_rl_algorithms()
    
    print("\n=== 强化学习演示完成 ===")
    print("已生成训练进度图表和策略可视化") 