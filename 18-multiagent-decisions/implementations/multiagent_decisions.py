"""
第18章：多代理决策
Multi-Agent Decisions

本章实现多代理系统中的决策理论，包括：
1. 博弈论基础
2. 纳什均衡
3. 合作博弈
4. 拍卖理论
5. 多代理强化学习
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Optional, Any
import seaborn as sns
import math

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Game:
    """博弈基础类"""
    
    def __init__(self, players: List[str], actions: Dict[str, List[str]], payoffs: Dict[tuple, Dict[str, float]]):
        self.players = players
        self.actions = actions
        self.payoffs = payoffs
        
    def get_payoff(self, action_profile: tuple, player: str) -> float:
        """获取特定动作组合下玩家的收益"""
        return self.payoffs.get(action_profile, {}).get(player, 0)
    
    def display_game(self):
        """显示博弈矩阵"""
        print(f"博弈参与者：{self.players}")
        print(f"动作空间：{self.actions}")
        print("收益矩阵：")
        for action_profile, payoffs in self.payoffs.items():
            print(f"  {action_profile}: {payoffs}")

class NashEquilibrium:
    """纳什均衡求解器"""
    
    def __init__(self, game: Game):
        self.game = game
        
    def find_pure_strategy_equilibria(self) -> List[tuple]:
        """寻找纯策略纳什均衡"""
        equilibria = []
        
        # 生成所有可能的动作组合
        all_actions = [self.game.actions[player] for player in self.game.players]
        
        for action_profile in product(*all_actions):
            is_equilibrium = True
            
            # 检查每个玩家是否有改进的动机
            for i, player in enumerate(self.game.players):
                current_payoff = self.game.get_payoff(action_profile, player)
                
                # 检查该玩家的所有其他动作
                for alternative_action in self.game.actions[player]:
                    if alternative_action != action_profile[i]:
                        # 构造新的动作组合
                        new_profile = list(action_profile)
                        new_profile[i] = alternative_action
                        new_profile = tuple(new_profile)
                        
                        new_payoff = self.game.get_payoff(new_profile, player)
                        
                        # 如果有更好的动作，不是纳什均衡
                        if new_payoff > current_payoff:
                            is_equilibrium = False
                            break
                
                if not is_equilibrium:
                    break
            
            if is_equilibrium:
                equilibria.append(action_profile)
        
        return equilibria
    
    def find_mixed_strategy_equilibrium_2x2(self) -> Optional[Dict[str, Dict[str, float]]]:
        """寻找2x2博弈的混合策略纳什均衡"""
        if len(self.game.players) != 2:
            return None
        
        player1, player2 = self.game.players
        actions1, actions2 = self.game.actions[player1], self.game.actions[player2]
        
        if len(actions1) != 2 or len(actions2) != 2:
            return None
        
        # 构造收益矩阵
        a1, a2 = actions1
        b1, b2 = actions2
        
        # 玩家1的收益矩阵
        u1_11 = self.game.get_payoff((a1, b1), player1)
        u1_12 = self.game.get_payoff((a1, b2), player1)
        u1_21 = self.game.get_payoff((a2, b1), player1)
        u1_22 = self.game.get_payoff((a2, b2), player1)
        
        # 玩家2的收益矩阵
        u2_11 = self.game.get_payoff((a1, b1), player2)
        u2_12 = self.game.get_payoff((a1, b2), player2)
        u2_21 = self.game.get_payoff((a2, b1), player2)
        u2_22 = self.game.get_payoff((a2, b2), player2)
        
        # 计算混合策略均衡
        denom1 = (u2_11 - u2_21) - (u2_12 - u2_22)
        denom2 = (u1_11 - u1_12) - (u1_21 - u1_22)
        
        if denom1 == 0 or denom2 == 0:
            return None
        
        p = (u2_22 - u2_12) / denom1  # 玩家1选择动作a1的概率
        q = (u1_22 - u1_21) / denom2  # 玩家2选择动作b1的概率
        
        if 0 <= p <= 1 and 0 <= q <= 1:
            return {
                player1: {a1: p, a2: 1-p},
                player2: {b1: q, b2: 1-q}
            }
        
        return None

class CooperativeGame:
    """合作博弈"""
    
    def __init__(self, players: List[str], characteristic_function: Dict[frozenset, float]):
        self.players = players
        self.v = characteristic_function  # 特征函数
        
    def shapley_value(self) -> Dict[str, float]:
        """计算Shapley值"""
        n = len(self.players)
        shapley_values = {player: 0.0 for player in self.players}
        
        # 生成所有可能的联盟
        from itertools import chain, combinations
        
        def powerset(iterable):
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        
        # 对每个玩家计算Shapley值
        for player in self.players:
            total_contribution = 0.0
            
            # 考虑所有不包含该玩家的联盟
            other_players = [p for p in self.players if p != player]
            
            for coalition in powerset(other_players):
                coalition_set = frozenset(coalition)
                coalition_with_player = frozenset(list(coalition) + [player])
                
                # 计算边际贡献
                contribution = (self.v.get(coalition_with_player, 0) - 
                              self.v.get(coalition_set, 0))
                
                # 计算权重
                size = len(coalition)
                weight = 1.0 / (n * math.comb(n-1, size))
                
                total_contribution += weight * contribution
            
            shapley_values[player] = total_contribution
        
        return shapley_values
    
    def core(self) -> List[Dict[str, float]]:
        """计算核心解"""
        # 这里提供一个简化的核心解检查
        # 实际实现需要更复杂的线性规划
        pass
    
    def nucleolus(self) -> Dict[str, float]:
        """计算核仁解"""
        # 简化实现
        pass

class Auction:
    """拍卖机制"""
    
    def __init__(self, items: List[str], bidders: List[str]):
        self.items = items
        self.bidders = bidders
        
    def first_price_sealed_bid(self, bids: Dict[str, float]) -> Tuple[str, float]:
        """一价密封拍卖"""
        if not bids:
            return None, 0
        
        winner = max(bids, key=bids.get)
        price = bids[winner]
        return winner, price
    
    def second_price_sealed_bid(self, bids: Dict[str, float]) -> Tuple[str, float]:
        """二价密封拍卖（Vickrey拍卖）"""
        if not bids:
            return None, 0
        
        sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_bids) >= 2:
            winner = sorted_bids[0][0]
            price = sorted_bids[1][1]
        else:
            winner = sorted_bids[0][0]
            price = sorted_bids[0][1]
        
        return winner, price
    
    def english_auction(self, initial_price: float, increment: float, 
                       max_bids: Dict[str, float]) -> Tuple[str, float]:
        """英式拍卖"""
        current_price = initial_price
        active_bidders = set(self.bidders)
        
        while len(active_bidders) > 1:
            # 移除出价低于当前价格的竞拍者
            active_bidders = {b for b in active_bidders if max_bids.get(b, 0) >= current_price}
            
            if len(active_bidders) <= 1:
                break
                
            current_price += increment
        
        if active_bidders:
            winner = list(active_bidders)[0]
            return winner, current_price - increment
        else:
            return None, 0

class MultiAgentQLearning:
    """多代理Q学习"""
    
    def __init__(self, n_agents: int, n_states: int, n_actions: int, 
                 alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 每个代理的Q表
        self.q_tables = [np.zeros((n_states, n_actions)) for _ in range(n_agents)]
        
    def select_action(self, agent_id: int, state: int) -> int:
        """选择动作（epsilon-greedy策略）"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_tables[agent_id][state])
    
    def update_q_value(self, agent_id: int, state: int, action: int, 
                      reward: float, next_state: int):
        """更新Q值"""
        best_next_action = np.argmax(self.q_tables[agent_id][next_state])
        td_target = reward + self.gamma * self.q_tables[agent_id][next_state][best_next_action]
        td_error = td_target - self.q_tables[agent_id][state][action]
        self.q_tables[agent_id][state][action] += self.alpha * td_error
    
    def train(self, environment, episodes: int = 1000):
        """训练多代理系统"""
        for episode in range(episodes):
            state = environment.reset()
            done = False
            
            while not done:
                # 所有代理选择动作
                actions = [self.select_action(i, state) for i in range(self.n_agents)]
                
                # 环境执行动作
                next_state, rewards, done = environment.step(actions)
                
                # 更新所有代理的Q值
                for i in range(self.n_agents):
                    self.update_q_value(i, state, actions[i], rewards[i], next_state)
                
                state = next_state

class CoordinationGame:
    """协调博弈环境"""
    
    def __init__(self, n_agents: int = 2, grid_size: int = 5):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        """重置环境"""
        self.agent_positions = [(random.randint(0, self.grid_size-1), 
                                random.randint(0, self.grid_size-1)) 
                               for _ in range(self.n_agents)]
        self.target_position = (random.randint(0, self.grid_size-1), 
                               random.randint(0, self.grid_size-1))
        return self.get_state()
    
    def get_state(self) -> int:
        """获取当前状态"""
        # 简化状态表示
        return hash(tuple(self.agent_positions + [self.target_position])) % 100
    
    def step(self, actions: List[int]) -> Tuple[int, List[float], bool]:
        """执行动作"""
        # 动作：0=上, 1=下, 2=左, 3=右, 4=停留
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
        
        # 更新代理位置
        for i, action in enumerate(actions):
            if action < len(moves):
                dx, dy = moves[action]
                x, y = self.agent_positions[i]
                new_x = max(0, min(self.grid_size-1, x + dx))
                new_y = max(0, min(self.grid_size-1, y + dy))
                self.agent_positions[i] = (new_x, new_y)
        
        # 计算奖励
        rewards = []
        for pos in self.agent_positions:
            distance = abs(pos[0] - self.target_position[0]) + abs(pos[1] - self.target_position[1])
            reward = -distance / self.grid_size  # 距离越近奖励越大
            rewards.append(reward)
        
        # 检查是否完成
        done = any(pos == self.target_position for pos in self.agent_positions)
        
        return self.get_state(), rewards, done

def demonstrate_game_theory():
    """演示博弈论基础"""
    print("=" * 50)
    print("博弈论基础演示")
    print("=" * 50)
    
    # 创建囚徒困境
    prisoners_dilemma = Game(
        players=['囚徒1', '囚徒2'],
        actions={'囚徒1': ['合作', '背叛'], '囚徒2': ['合作', '背叛']},
        payoffs={
            ('合作', '合作'): {'囚徒1': -1, '囚徒2': -1},
            ('合作', '背叛'): {'囚徒1': -3, '囚徒2': 0},
            ('背叛', '合作'): {'囚徒1': 0, '囚徒2': -3},
            ('背叛', '背叛'): {'囚徒1': -2, '囚徒2': -2}
        }
    )
    
    print("囚徒困境博弈：")
    prisoners_dilemma.display_game()
    
    # 寻找纳什均衡
    nash_solver = NashEquilibrium(prisoners_dilemma)
    pure_equilibria = nash_solver.find_pure_strategy_equilibria()
    print(f"\n纯策略纳什均衡：{pure_equilibria}")
    
    mixed_equilibrium = nash_solver.find_mixed_strategy_equilibrium_2x2()
    if mixed_equilibrium:
        print(f"混合策略纳什均衡：{mixed_equilibrium}")
    
    # 协调博弈
    coordination_game = Game(
        players=['玩家1', '玩家2'],
        actions={'玩家1': ['A', 'B'], '玩家2': ['A', 'B']},
        payoffs={
            ('A', 'A'): {'玩家1': 5, '玩家2': 5},
            ('A', 'B'): {'玩家1': 0, '玩家2': 0},
            ('B', 'A'): {'玩家1': 0, '玩家2': 0},
            ('B', 'B'): {'玩家1': 3, '玩家2': 3}
        }
    )
    
    print("\n\n协调博弈：")
    coordination_game.display_game()
    
    nash_solver2 = NashEquilibrium(coordination_game)
    pure_equilibria2 = nash_solver2.find_pure_strategy_equilibria()
    print(f"\n纯策略纳什均衡：{pure_equilibria2}")

def demonstrate_cooperative_game():
    """演示合作博弈"""
    print("=" * 50)
    print("合作博弈演示")
    print("=" * 50)
    
    # 创建三人合作博弈
    players = ['A', 'B', 'C']
    characteristic_function = {
        frozenset(): 0,
        frozenset(['A']): 10,
        frozenset(['B']): 20,
        frozenset(['C']): 25,
        frozenset(['A', 'B']): 40,
        frozenset(['A', 'C']): 45,
        frozenset(['B', 'C']): 55,
        frozenset(['A', 'B', 'C']): 72
    }
    
    coop_game = CooperativeGame(players, characteristic_function)
    
    print("特征函数：")
    for coalition, value in characteristic_function.items():
        coalition_str = '{' + ', '.join(sorted(coalition)) + '}' if coalition else '∅'
        print(f"  v({coalition_str}) = {value}")
    
    # 计算Shapley值
    shapley_values = coop_game.shapley_value()
    print(f"\nShapley值：{shapley_values}")
    
    # 验证有效性
    total_shapley = sum(shapley_values.values())
    grand_coalition_value = characteristic_function[frozenset(players)]
    print(f"Shapley值总和：{total_shapley}")
    print(f"大联盟价值：{grand_coalition_value}")
    print(f"有效性验证：{abs(total_shapley - grand_coalition_value) < 1e-10}")

def demonstrate_auction():
    """演示拍卖机制"""
    print("=" * 50)
    print("拍卖机制演示")
    print("=" * 50)
    
    auction = Auction(['物品1'], ['买家1', '买家2', '买家3'])
    bids = {'买家1': 100, '买家2': 120, '买家3': 90}
    
    print(f"竞拍者出价：{bids}")
    
    # 一价密封拍卖
    winner1, price1 = auction.first_price_sealed_bid(bids)
    print(f"\n一价密封拍卖结果：")
    print(f"获胜者：{winner1}，支付价格：{price1}")
    
    # 二价密封拍卖
    winner2, price2 = auction.second_price_sealed_bid(bids)
    print(f"\n二价密封拍卖结果：")
    print(f"获胜者：{winner2}，支付价格：{price2}")
    
    # 英式拍卖
    max_bids = {'买家1': 110, '买家2': 130, '买家3': 95}
    winner3, price3 = auction.english_auction(50, 10, max_bids)
    print(f"\n英式拍卖结果：")
    print(f"获胜者：{winner3}，支付价格：{price3}")

def demonstrate_multiagent_learning():
    """演示多代理学习"""
    print("=" * 50)
    print("多代理学习演示")
    print("=" * 50)
    
    # 创建协调博弈环境
    env = CoordinationGame(n_agents=2, grid_size=5)
    
    # 创建多代理Q学习算法
    maql = MultiAgentQLearning(n_agents=2, n_states=100, n_actions=5)
    
    # 训练
    print("开始训练多代理系统...")
    
    # 记录训练过程
    episode_rewards = []
    
    for episode in range(500):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 50:
            actions = [maql.select_action(i, state) for i in range(2)]
            next_state, rewards, done = env.step(actions)
            
            for i in range(2):
                maql.update_q_value(i, state, actions[i], rewards[i], next_state)
            
            state = next_state
            total_reward += sum(rewards)
            steps += 1
        
        episode_rewards.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}: 平均奖励 = {avg_reward:.2f}")
    
    # 可视化训练结果
    plt.figure(figsize=(10, 6))
    
    # 计算移动平均
    window_size = 50
    moving_avg = []
    for i in range(len(episode_rewards)):
        start = max(0, i - window_size + 1)
        moving_avg.append(np.mean(episode_rewards[start:i+1]))
    
    plt.plot(episode_rewards, alpha=0.3, label='实际奖励')
    plt.plot(moving_avg, label=f'{window_size}期移动平均')
    plt.xlabel('训练轮次')
    plt.ylabel('累计奖励')
    plt.title('多代理Q学习训练过程')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_game_theory_concepts():
    """可视化博弈论概念"""
    print("=" * 50)
    print("博弈论概念可视化")
    print("=" * 50)
    
    # 创建不同类型的博弈可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 囚徒困境收益矩阵
    ax1 = axes[0, 0]
    payoff_matrix = np.array([[-1, -3], [0, -2]])
    im1 = ax1.imshow(payoff_matrix, cmap='RdYlBu', aspect='auto')
    ax1.set_title('囚徒困境收益矩阵（囚徒1）')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['合作', '背叛'])
    ax1.set_yticklabels(['合作', '背叛'])
    ax1.set_xlabel('囚徒2的策略')
    ax1.set_ylabel('囚徒1的策略')
    
    # 添加数值标签
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f'{payoff_matrix[i, j]}', ha='center', va='center', 
                    color='white' if payoff_matrix[i, j] < -1.5 else 'black')
    
    # 2. 协调博弈收益矩阵
    ax2 = axes[0, 1]
    coord_matrix = np.array([[5, 0], [0, 3]])
    im2 = ax2.imshow(coord_matrix, cmap='RdYlBu', aspect='auto')
    ax2.set_title('协调博弈收益矩阵')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['A', 'B'])
    ax2.set_yticklabels(['A', 'B'])
    ax2.set_xlabel('玩家2的策略')
    ax2.set_ylabel('玩家1的策略')
    
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{coord_matrix[i, j]}', ha='center', va='center', 
                    color='white' if coord_matrix[i, j] < 2 else 'black')
    
    # 3. 拍卖价格比较
    ax3 = axes[1, 0]
    auction_types = ['一价密封', '二价密封', '英式拍卖']
    prices = [120, 100, 120]  # 示例价格
    bars = ax3.bar(auction_types, prices, color=['red', 'green', 'blue'], alpha=0.7)
    ax3.set_title('不同拍卖机制的价格比较')
    ax3.set_ylabel('成交价格')
    
    for i, (bar, price) in enumerate(zip(bars, prices)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{price}', ha='center', va='bottom')
    
    # 4. Shapley值分布
    ax4 = axes[1, 1]
    players = ['A', 'B', 'C']
    shapley_values = [20.83, 23.33, 27.83]  # 示例Shapley值
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = ax4.bar(players, shapley_values, color=colors, alpha=0.7)
    ax4.set_title('Shapley值分布')
    ax4.set_ylabel('Shapley值')
    
    for i, (bar, value) in enumerate(zip(bars, shapley_values)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("第18章：多代理决策")
    print("Multi-Agent Decisions")
    print("=" * 60)
    
    # 演示各个概念
    demonstrate_game_theory()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_cooperative_game()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_auction()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_multiagent_learning()
    print("\n" + "=" * 60 + "\n")
    
    # 可视化
    visualize_game_theory_concepts()
    
    print("\n多代理决策演示完成！")
    print("涵盖内容：")
    print("1. 博弈论基础与纳什均衡")
    print("2. 合作博弈与Shapley值")
    print("3. 拍卖机制设计")
    print("4. 多代理强化学习")
    print("5. 协调博弈与学习")

if __name__ == "__main__":
    main() 