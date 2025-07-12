"""
高级AI应用：智能游戏AI
整合了搜索算法、对抗性搜索、机器学习、强化学习等多种AI技术
"""

import numpy as np
import random
import math
from typing import List, Tuple, Dict, Optional, Any
from abc import ABC, abstractmethod
from collections import defaultdict
import time


class GameState(ABC):
    """抽象游戏状态类"""
    
    @abstractmethod
    def get_legal_actions(self, player: int) -> List[Any]:
        """获取当前玩家的合法动作"""
        pass
    
    @abstractmethod
    def make_move(self, action: Any, player: int) -> 'GameState':
        """执行动作，返回新状态"""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """检查是否为终止状态"""
        pass
    
    @abstractmethod
    def get_winner(self) -> Optional[int]:
        """获取获胜者（如果游戏结束）"""
        pass
    
    @abstractmethod
    def evaluate(self, player: int) -> float:
        """评估状态对指定玩家的价值"""
        pass
    
    @abstractmethod
    def get_current_player(self) -> int:
        """获取当前轮到的玩家"""
        pass


class TicTacToeState(GameState):
    """井字棋游戏状态"""
    
    def __init__(self, board: List[List[int]] = None, current_player: int = 1):
        self.board = board or [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.current_player = current_player
        self.size = 3
    
    def get_legal_actions(self, player: int) -> List[Tuple[int, int]]:
        """获取空位置"""
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    actions.append((i, j))
        return actions
    
    def make_move(self, action: Tuple[int, int], player: int) -> 'TicTacToeState':
        """执行动作"""
        new_board = [row[:] for row in self.board]
        i, j = action
        new_board[i][j] = player
        next_player = 2 if player == 1 else 1
        return TicTacToeState(new_board, next_player)
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self.get_winner() is not None or not self.get_legal_actions(self.current_player)
    
    def get_winner(self) -> Optional[int]:
        """检查获胜者"""
        # 检查行
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]
        
        # 检查列
        for j in range(self.size):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != 0:
                return self.board[0][j]
        
        # 检查对角线
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]
        
        return None
    
    def evaluate(self, player: int) -> float:
        """简单的评估函数"""
        winner = self.get_winner()
        if winner == player:
            return 1.0
        elif winner is not None:
            return -1.0
        else:
            return 0.0
    
    def get_current_player(self) -> int:
        return self.current_player
    
    def __str__(self):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        lines = []
        for row in self.board:
            line = '|'.join(symbols[cell] for cell in row)
            lines.append(line)
        return '\n-----\n'.join(lines)


class ConnectFourState(GameState):
    """四子棋游戏状态"""
    
    def __init__(self, board: List[List[int]] = None, current_player: int = 1):
        self.rows = 6
        self.cols = 7
        self.board = board or [[0] * self.cols for _ in range(self.rows)]
        self.current_player = current_player
    
    def get_legal_actions(self, player: int) -> List[int]:
        """获取可放置的列"""
        actions = []
        for col in range(self.cols):
            if self.board[0][col] == 0:  # 顶部为空
                actions.append(col)
        return actions
    
    def make_move(self, action: int, player: int) -> 'ConnectFourState':
        """在指定列放置棋子"""
        new_board = [row[:] for row in self.board]
        col = action
        
        # 找到最底部的空位
        for row in range(self.rows - 1, -1, -1):
            if new_board[row][col] == 0:
                new_board[row][col] = player
                break
        
        next_player = 2 if player == 1 else 1
        return ConnectFourState(new_board, next_player)
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self.get_winner() is not None or not self.get_legal_actions(self.current_player)
    
    def get_winner(self) -> Optional[int]:
        """检查四子连线"""
        # 检查水平、垂直、对角线
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] == 0:
                    continue
                
                player = self.board[row][col]
                
                for dr, dc in directions:
                    count = 1
                    r, c = row + dr, col + dc
                    
                    while (0 <= r < self.rows and 0 <= c < self.cols and 
                           self.board[r][c] == player):
                        count += 1
                        r, c = r + dr, c + dc
                    
                    if count >= 4:
                        return player
        
        return None
    
    def evaluate(self, player: int) -> float:
        """四子棋评估函数"""
        winner = self.get_winner()
        if winner == player:
            return 1000.0
        elif winner is not None:
            return -1000.0
        
        # 计算连子数得分
        score = 0
        opponent = 2 if player == 1 else 1
        
        # 评估所有可能的四连线
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(self.rows):
            for col in range(self.cols):
                for dr, dc in directions:
                    # 检查从这个位置开始的四个位置
                    if (row + 3*dr >= 0 and row + 3*dr < self.rows and
                        col + 3*dc >= 0 and col + 3*dc < self.cols):
                        
                        window = []
                        for i in range(4):
                            r, c = row + i*dr, col + i*dc
                            window.append(self.board[r][c])
                        
                        score += self._evaluate_window(window, player, opponent)
        
        return score
    
    def _evaluate_window(self, window: List[int], player: int, opponent: int) -> float:
        """评估四个位置的窗口"""
        score = 0
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)
        
        if player_count == 4:
            score += 100
        elif player_count == 3 and empty_count == 1:
            score += 5
        elif player_count == 2 and empty_count == 2:
            score += 2
        
        if opponent_count == 3 and empty_count == 1:
            score -= 80
        
        return score
    
    def get_current_player(self) -> int:
        return self.current_player


class MinimaxAgent:
    """极小极大算法智能体"""
    
    def __init__(self, depth: int = 4):
        self.depth = depth
    
    def get_action(self, state: GameState, player: int) -> Any:
        """获取最佳动作"""
        _, action = self._minimax(state, self.depth, player, True)
        return action
    
    def _minimax(self, state: GameState, depth: int, player: int, maximizing: bool) -> Tuple[float, Any]:
        """极小极大算法"""
        if depth == 0 or state.is_terminal():
            return state.evaluate(player), None
        
        legal_actions = state.get_legal_actions(state.get_current_player())
        best_action = legal_actions[0] if legal_actions else None
        
        if maximizing:
            max_eval = float('-inf')
            for action in legal_actions:
                new_state = state.make_move(action, state.get_current_player())
                eval_score, _ = self._minimax(new_state, depth - 1, player, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in legal_actions:
                new_state = state.make_move(action, state.get_current_player())
                eval_score, _ = self._minimax(new_state, depth - 1, player, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
            return min_eval, best_action


class AlphaBetaAgent:
    """Alpha-Beta剪枝智能体"""
    
    def __init__(self, depth: int = 6):
        self.depth = depth
    
    def get_action(self, state: GameState, player: int) -> Any:
        """获取最佳动作"""
        _, action = self._alpha_beta(state, self.depth, float('-inf'), float('inf'), player, True)
        return action
    
    def _alpha_beta(self, state: GameState, depth: int, alpha: float, beta: float, 
                   player: int, maximizing: bool) -> Tuple[float, Any]:
        """Alpha-Beta剪枝算法"""
        if depth == 0 or state.is_terminal():
            return state.evaluate(player), None
        
        legal_actions = state.get_legal_actions(state.get_current_player())
        best_action = legal_actions[0] if legal_actions else None
        
        if maximizing:
            max_eval = float('-inf')
            for action in legal_actions:
                new_state = state.make_move(action, state.get_current_player())
                eval_score, _ = self._alpha_beta(new_state, depth - 1, alpha, beta, player, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta剪枝
            return max_eval, best_action
        else:
            min_eval = float('inf')
            for action in legal_actions:
                new_state = state.make_move(action, state.get_current_player())
                eval_score, _ = self._alpha_beta(new_state, depth - 1, alpha, beta, player, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha剪枝
            return min_eval, best_action


class MCTSNode:
    """蒙特卡洛树搜索节点"""
    
    def __init__(self, state: GameState, parent: 'MCTSNode' = None, action: Any = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = state.get_legal_actions(state.get_current_player())
    
    def is_fully_expanded(self) -> bool:
        """是否完全展开"""
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param: float = 1.4) -> 'MCTSNode':
        """选择最佳子节点（UCB1）"""
        choices_weights = []
        for child in self.children:
            weight = (child.wins / child.visits) + c_param * math.sqrt(
                (2 * math.log(self.visits)) / child.visits)
            choices_weights.append(weight)
        
        return self.children[np.argmax(choices_weights)]
    
    def expand(self) -> 'MCTSNode':
        """展开一个新子节点"""
        action = self.untried_actions.pop()
        new_state = self.state.make_move(action, self.state.get_current_player())
        child = MCTSNode(new_state, self, action)
        self.children.append(child)
        return child
    
    def rollout(self) -> int:
        """随机模拟到游戏结束"""
        current_state = self.state
        while not current_state.is_terminal():
            actions = current_state.get_legal_actions(current_state.get_current_player())
            if not actions:
                break
            action = random.choice(actions)
            current_state = current_state.make_move(action, current_state.get_current_player())
        
        winner = current_state.get_winner()
        return winner if winner is not None else 0
    
    def backpropagate(self, result: int, player: int):
        """反向传播结果"""
        self.visits += 1
        if result == player:
            self.wins += 1
        elif result != 0:  # 对手获胜
            self.wins -= 1
        
        if self.parent:
            self.parent.backpropagate(result, player)


class MCTSAgent:
    """蒙特卡洛树搜索智能体"""
    
    def __init__(self, iterations: int = 1000):
        self.iterations = iterations
    
    def get_action(self, state: GameState, player: int) -> Any:
        """获取最佳动作"""
        root = MCTSNode(state)
        
        for _ in range(self.iterations):
            # 选择
            node = self._select(root)
            
            # 展开
            if not node.state.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # 模拟
            result = node.rollout()
            
            # 反向传播
            node.backpropagate(result, player)
        
        # 选择访问次数最多的子节点
        best_child = max(root.children, key=lambda x: x.visits)
        return best_child.action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段"""
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                node = node.best_child()
        return node


class QLearningAgent:
    """Q学习智能体"""
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def get_action(self, state: GameState, player: int) -> Any:
        """获取动作（ε-贪心策略）"""
        state_key = str(state.board)
        legal_actions = state.get_legal_actions(player)
        
        if not legal_actions:
            return None
        
        if random.random() < self.epsilon:
            # 探索：随机选择
            return random.choice(legal_actions)
        else:
            # 利用：选择Q值最高的动作
            best_action = legal_actions[0]
            best_q_value = self.q_table[state_key][str(best_action)]
            
            for action in legal_actions:
                q_value = self.q_table[state_key][str(action)]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            return best_action
    
    def update(self, state: GameState, action: Any, reward: float, next_state: GameState):
        """更新Q值"""
        state_key = str(state.board)
        action_key = str(action)
        next_state_key = str(next_state.board)
        
        current_q = self.q_table[state_key][action_key]
        
        # 计算下一状态的最大Q值
        next_legal_actions = next_state.get_legal_actions(next_state.get_current_player())
        max_next_q = 0
        if next_legal_actions:
            max_next_q = max(self.q_table[next_state_key][str(a)] for a in next_legal_actions)
        
        # Q学习更新规则
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action_key] = new_q


def play_game(state: GameState, agent1, agent2, player1: int = 1, player2: int = 2) -> Optional[int]:
    """进行一局游戏"""
    current_state = state
    
    while not current_state.is_terminal():
        current_player = current_state.get_current_player()
        
        if current_player == player1:
            action = agent1.get_action(current_state, player1)
        else:
            action = agent2.get_action(current_state, player2)
        
        if action is None:
            break
        
        current_state = current_state.make_move(action, current_player)
    
    return current_state.get_winner()


def tournament(agents: List[Tuple[str, Any]], game_class: type, num_games: int = 100):
    """智能体锦标赛"""
    print(f"\n=== {game_class.__name__} 锦标赛 ===")
    
    results = defaultdict(lambda: defaultdict(int))
    
    for i, (name1, agent1) in enumerate(agents):
        for j, (name2, agent2) in enumerate(agents):
            if i >= j:
                continue
            
            wins1 = wins2 = draws = 0
            
            for _ in range(num_games):
                # 交替先手
                if _ % 2 == 0:
                    winner = play_game(game_class(), agent1, agent2, 1, 2)
                    if winner == 1:
                        wins1 += 1
                    elif winner == 2:
                        wins2 += 1
                    else:
                        draws += 1
                else:
                    winner = play_game(game_class(), agent2, agent1, 1, 2)
                    if winner == 1:
                        wins2 += 1
                    elif winner == 2:
                        wins1 += 1
                    else:
                        draws += 1
            
            results[name1][name2] = f"{wins1}-{draws}-{wins2}"
            results[name2][name1] = f"{wins2}-{draws}-{wins1}"
            
            print(f"{name1} vs {name2}: {wins1}-{draws}-{wins2}")


def demonstrate_tic_tac_toe():
    """演示井字棋AI"""
    print("=== 井字棋AI演示 ===")
    
    # 创建不同的智能体
    agents = [
        ("随机", type('RandomAgent', (), {
            'get_action': lambda self, state, player: random.choice(state.get_legal_actions(player)) if state.get_legal_actions(player) else None
        })()),
        ("极小极大", MinimaxAgent(depth=9)),
        ("Alpha-Beta", AlphaBetaAgent(depth=9)),
        ("MCTS", MCTSAgent(iterations=500))
    ]
    
    tournament(agents, TicTacToeState, num_games=50)


def demonstrate_connect_four():
    """演示四子棋AI"""
    print("=== 四子棋AI演示 ===")
    
    agents = [
        ("随机", type('RandomAgent', (), {
            'get_action': lambda self, state, player: random.choice(state.get_legal_actions(player)) if state.get_legal_actions(player) else None
        })()),
        ("极小极大", MinimaxAgent(depth=4)),
        ("Alpha-Beta", AlphaBetaAgent(depth=6)),
        ("MCTS", MCTSAgent(iterations=1000))
    ]
    
    tournament(agents, ConnectFourState, num_games=20)


def interactive_game():
    """与AI对战的交互式游戏"""
    print("\n=== 与AI对战井字棋 ===")
    
    ai_agent = AlphaBetaAgent(depth=9)
    state = TicTacToeState()
    human_player = 1
    ai_player = 2
    
    print("你是X，AI是O")
    print("输入行列坐标（0-2），用空格分隔")
    
    while not state.is_terminal():
        print(f"\n当前棋盘：")
        print(state)
        
        if state.get_current_player() == human_player:
            try:
                coord = input("请输入你的移动 (行 列): ").split()
                row, col = int(coord[0]), int(coord[1])
                
                if (row, col) in state.get_legal_actions(human_player):
                    state = state.make_move((row, col), human_player)
                else:
                    print("无效移动，请重试")
                    continue
            except (ValueError, IndexError):
                print("输入格式错误，请重试")
                continue
        else:
            print("AI思考中...")
            action = ai_agent.get_action(state, ai_player)
            state = state.make_move(action, ai_player)
            print(f"AI选择: {action}")
    
    print(f"\n最终棋盘：")
    print(state)
    
    winner = state.get_winner()
    if winner == human_player:
        print("恭喜你获胜！")
    elif winner == ai_player:
        print("AI获胜！")
    else:
        print("平局！")


def main():
    """主演示函数"""
    print("高级AI应用：智能游戏AI")
    print("整合了多种AI技术：搜索、对抗性搜索、机器学习、强化学习")
    
    demonstrate_tic_tac_toe()
    demonstrate_connect_four()
    
    # 可选：交互式游戏
    # interactive_game()
    
    print("\n=== 游戏AI技术总结 ===")
    print("1. 极小极大算法：完全信息零和博弈的最优策略")
    print("2. Alpha-Beta剪枝：提高搜索效率")
    print("3. 蒙特卡洛树搜索：适用于大状态空间的近似算法")
    print("4. 强化学习：通过自我对弈学习策略")
    print("5. 评估函数：领域知识的数值化表示")


if __name__ == "__main__":
    main() 