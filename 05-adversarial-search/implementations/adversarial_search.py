"""
对抗性搜索和游戏算法实现

包含Minimax算法、Alpha-Beta剪枝、Monte Carlo树搜索等
"""

import math
import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from collections import defaultdict
import copy


class GameState(ABC):
    """游戏状态的抽象基类"""
    
    @abstractmethod
    def get_legal_actions(self) -> List[Any]:
        """获取当前状态的合法动作"""
        pass
    
    @abstractmethod
    def make_move(self, action: Any) -> 'GameState':
        """执行动作，返回新的游戏状态"""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """检查是否为终止状态"""
        pass
    
    @abstractmethod
    def get_utility(self, player: int) -> float:
        """获取当前状态对指定玩家的效用值"""
        pass
    
    @abstractmethod
    def get_current_player(self) -> int:
        """获取当前轮到的玩家"""
        pass


class TicTacToe(GameState):
    """井字棋游戏实现"""
    
    def __init__(self, board: Optional[List[List[int]]] = None, player: int = 1):
        self.board = board if board is not None else [[0] * 3 for _ in range(3)]
        self.current_player = player
        self.size = 3
    
    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """获取合法动作（空位置）"""
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    actions.append((i, j))
        return actions
    
    def make_move(self, action: Tuple[int, int]) -> 'TicTacToe':
        """执行动作"""
        i, j = action
        new_board = copy.deepcopy(self.board)
        new_board[i][j] = self.current_player
        return TicTacToe(new_board, -self.current_player)
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self.get_winner() is not None or len(self.get_legal_actions()) == 0
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
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
    
    def get_utility(self, player: int) -> float:
        """获取效用值"""
        winner = self.get_winner()
        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        else:
            return 0.0
    
    def get_current_player(self) -> int:
        """获取当前玩家"""
        return self.current_player
    
    def __str__(self) -> str:
        """字符串表示"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        result = []
        for row in self.board:
            result.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(result)


class MinimaxAgent:
    """Minimax算法实现"""
    
    def __init__(self, player: int, max_depth: int = 5):
        self.player = player
        self.max_depth = max_depth
        self.nodes_expanded = 0
    
    def get_action(self, state: GameState) -> Any:
        """获取最佳动作"""
        self.nodes_expanded = 0
        _, action = self.minimax(state, 0, True)
        return action
    
    def minimax(self, state: GameState, depth: int, maximizing: bool) -> Tuple[float, Any]:
        """Minimax算法"""
        self.nodes_expanded += 1
        
        if state.is_terminal() or depth >= self.max_depth:
            return state.get_utility(self.player), None
        
        legal_actions = state.get_legal_actions()
        best_action = legal_actions[0] if legal_actions else None
        
        if maximizing:
            best_value = float('-inf')
            for action in legal_actions:
                new_state = state.make_move(action)
                value, _ = self.minimax(new_state, depth + 1, False)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action
        else:
            best_value = float('inf')
            for action in legal_actions:
                new_state = state.make_move(action)
                value, _ = self.minimax(new_state, depth + 1, True)
                if value < best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action


class AlphaBetaAgent:
    """Alpha-Beta剪枝算法实现"""
    
    def __init__(self, player: int, max_depth: int = 5):
        self.player = player
        self.max_depth = max_depth
        self.nodes_expanded = 0
        self.pruning_count = 0
    
    def get_action(self, state: GameState) -> Any:
        """获取最佳动作"""
        self.nodes_expanded = 0
        self.pruning_count = 0
        _, action = self.alpha_beta(state, 0, float('-inf'), float('inf'), True)
        return action
    
    def alpha_beta(self, state: GameState, depth: int, alpha: float, beta: float, 
                   maximizing: bool) -> Tuple[float, Any]:
        """Alpha-Beta剪枝算法"""
        self.nodes_expanded += 1
        
        if state.is_terminal() or depth >= self.max_depth:
            return state.get_utility(self.player), None
        
        legal_actions = state.get_legal_actions()
        best_action = legal_actions[0] if legal_actions else None
        
        if maximizing:
            best_value = float('-inf')
            for action in legal_actions:
                new_state = state.make_move(action)
                value, _ = self.alpha_beta(new_state, depth + 1, alpha, beta, False)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, value)
                if beta <= alpha:
                    self.pruning_count += 1
                    break  # Beta剪枝
            return best_value, best_action
        else:
            best_value = float('inf')
            for action in legal_actions:
                new_state = state.make_move(action)
                value, _ = self.alpha_beta(new_state, depth + 1, alpha, beta, True)
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, value)
                if beta <= alpha:
                    self.pruning_count += 1
                    break  # Alpha剪枝
            return best_value, best_action


class MCTSNode:
    """Monte Carlo树搜索节点"""
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, 
                 action: Any = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = state.get_legal_actions()
    
    def is_fully_expanded(self) -> bool:
        """检查节点是否完全展开"""
        return len(self.untried_actions) == 0
    
    def expand(self) -> 'MCTSNode':
        """展开节点"""
        action = self.untried_actions.pop()
        new_state = self.state.make_move(action)
        child = MCTSNode(new_state, self, action)
        self.children.append(child)
        return child
    
    def select_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """使用UCB1选择子节点"""
        def ucb1(node: MCTSNode) -> float:
            if node.visits == 0:
                return float('inf')
            return (node.wins / node.visits + 
                    exploration_weight * math.sqrt(math.log(self.visits) / node.visits))
        
        return max(self.children, key=ucb1)
    
    def update(self, result: float):
        """更新节点统计信息"""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.update(result)


class MCTSAgent:
    """Monte Carlo树搜索算法实现"""
    
    def __init__(self, player: int, simulations: int = 1000):
        self.player = player
        self.simulations = simulations
    
    def get_action(self, state: GameState) -> Any:
        """获取最佳动作"""
        root = MCTSNode(state)
        
        for _ in range(self.simulations):
            node = root
            
            # 选择
            while not node.state.is_terminal() and node.is_fully_expanded():
                node = node.select_child()
            
            # 展开
            if not node.state.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # 模拟
            result = self.simulate(node.state)
            
            # 反向传播
            node.update(result)
        
        # 选择访问次数最多的子节点
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def simulate(self, state: GameState) -> float:
        """随机模拟游戏"""
        current_state = state
        
        while not current_state.is_terminal():
            actions = current_state.get_legal_actions()
            action = random.choice(actions)
            current_state = current_state.make_move(action)
        
        return current_state.get_utility(self.player)


class ConnectFour(GameState):
    """四子连珠游戏实现"""
    
    def __init__(self, board: Optional[List[List[int]]] = None, player: int = 1):
        self.rows = 6
        self.cols = 7
        self.board = board if board is not None else [[0] * self.cols for _ in range(self.rows)]
        self.current_player = player
    
    def get_legal_actions(self) -> List[int]:
        """获取合法动作（可以放置棋子的列）"""
        actions = []
        for col in range(self.cols):
            if self.board[0][col] == 0:
                actions.append(col)
        return actions
    
    def make_move(self, action: int) -> 'ConnectFour':
        """执行动作"""
        new_board = copy.deepcopy(self.board)
        
        # 找到该列的最底层空位
        for row in range(self.rows - 1, -1, -1):
            if new_board[row][action] == 0:
                new_board[row][action] = self.current_player
                break
        
        return ConnectFour(new_board, -self.current_player)
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self.get_winner() is not None or len(self.get_legal_actions()) == 0
    
    def get_winner(self) -> Optional[int]:
        """获取获胜者"""
        # 检查水平方向
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if (self.board[row][col] == self.board[row][col + 1] == 
                    self.board[row][col + 2] == self.board[row][col + 3] != 0):
                    return self.board[row][col]
        
        # 检查垂直方向
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if (self.board[row][col] == self.board[row + 1][col] == 
                    self.board[row + 2][col] == self.board[row + 3][col] != 0):
                    return self.board[row][col]
        
        # 检查对角线方向
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if (self.board[row][col] == self.board[row + 1][col + 1] == 
                    self.board[row + 2][col + 2] == self.board[row + 3][col + 3] != 0):
                    return self.board[row][col]
        
        # 检查反对角线方向
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if (self.board[row][col] == self.board[row - 1][col + 1] == 
                    self.board[row - 2][col + 2] == self.board[row - 3][col + 3] != 0):
                    return self.board[row][col]
        
        return None
    
    def get_utility(self, player: int) -> float:
        """获取效用值"""
        winner = self.get_winner()
        if winner == player:
            return 1.0
        elif winner == -player:
            return -1.0
        else:
            return 0.0
    
    def get_current_player(self) -> int:
        """获取当前玩家"""
        return self.current_player
    
    def __str__(self) -> str:
        """字符串表示"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        result = []
        for row in self.board:
            result.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(result)


class GameSimulator:
    """游戏模拟器"""
    
    def __init__(self, game_class: type, player1_agent: Any, player2_agent: Any):
        self.game_class = game_class
        self.player1_agent = player1_agent
        self.player2_agent = player2_agent
    
    def play_game(self, verbose: bool = False) -> Tuple[int, Dict[str, Any]]:
        """模拟一局游戏"""
        state = self.game_class()
        move_count = 0
        
        if verbose:
            print(f"初始状态:")
            print(state)
            print("-" * 20)
        
        while not state.is_terminal():
            current_player = state.get_current_player()
            
            if current_player == 1:
                action = self.player1_agent.get_action(state)
            else:
                action = self.player2_agent.get_action(state)
            
            state = state.make_move(action)
            move_count += 1
            
            if verbose:
                print(f"玩家 {current_player} 执行动作: {action}")
                print(state)
                print("-" * 20)
        
        winner = state.get_winner()
        
        # 收集统计信息
        stats = {
            'winner': winner,
            'moves': move_count,
            'player1_nodes': getattr(self.player1_agent, 'nodes_expanded', 0),
            'player2_nodes': getattr(self.player2_agent, 'nodes_expanded', 0),
            'player1_pruning': getattr(self.player1_agent, 'pruning_count', 0),
            'player2_pruning': getattr(self.player2_agent, 'pruning_count', 0)
        }
        
        return winner, stats
    
    def tournament(self, num_games: int = 10) -> Dict[str, Any]:
        """进行锦标赛"""
        results = {
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'total_moves': 0,
            'total_nodes': 0
        }
        
        for i in range(num_games):
            winner, stats = self.play_game()
            
            if winner == 1:
                results['player1_wins'] += 1
            elif winner == -1:
                results['player2_wins'] += 1
            else:
                results['draws'] += 1
            
            results['total_moves'] += stats['moves']
            results['total_nodes'] += stats['player1_nodes'] + stats['player2_nodes']
            
            print(f"游戏 {i+1}: 获胜者 = {winner}, 步数 = {stats['moves']}")
        
        results['avg_moves'] = results['total_moves'] / num_games
        results['avg_nodes'] = results['total_nodes'] / num_games
        
        return results


def compare_algorithms():
    """比较不同算法的性能"""
    print("对抗性搜索算法性能比较")
    print("=" * 50)
    
    # 创建不同的智能体
    minimax_agent = MinimaxAgent(1, max_depth=4)
    alphabeta_agent = AlphaBetaAgent(-1, max_depth=4)
    mcts_agent = MCTSAgent(1, simulations=500)
    
    # 比较 Minimax vs Alpha-Beta
    print("\n1. Minimax vs Alpha-Beta (井字棋)")
    print("-" * 30)
    
    simulator = GameSimulator(TicTacToe, minimax_agent, alphabeta_agent)
    results = simulator.tournament(5)
    
    print(f"Minimax 获胜: {results['player1_wins']}")
    print(f"Alpha-Beta 获胜: {results['player2_wins']}")
    print(f"平局: {results['draws']}")
    print(f"平均步数: {results['avg_moves']:.2f}")
    print(f"平均节点展开数: {results['avg_nodes']:.2f}")
    
    # 比较 Alpha-Beta vs MCTS
    print("\n2. Alpha-Beta vs MCTS (四子连珠)")
    print("-" * 30)
    
    # 为四子连珠创建新的智能体
    alphabeta_c4 = AlphaBetaAgent(1, max_depth=3)
    mcts_c4 = MCTSAgent(-1, simulations=200)
    
    simulator = GameSimulator(ConnectFour, alphabeta_c4, mcts_c4)
    results = simulator.tournament(3)
    
    print(f"Alpha-Beta 获胜: {results['player1_wins']}")
    print(f"MCTS 获胜: {results['player2_wins']}")
    print(f"平局: {results['draws']}")
    print(f"平均步数: {results['avg_moves']:.2f}")
    print(f"平均节点展开数: {results['avg_nodes']:.2f}")


def demonstrate_pruning():
    """演示Alpha-Beta剪枝的效果"""
    print("\nAlpha-Beta剪枝效果演示")
    print("=" * 50)
    
    game = TicTacToe()
    
    # 不使用剪枝的Minimax
    minimax_agent = MinimaxAgent(1, max_depth=5)
    minimax_action = minimax_agent.get_action(game)
    minimax_nodes = minimax_agent.nodes_expanded
    
    # 使用剪枝的Alpha-Beta
    alphabeta_agent = AlphaBetaAgent(1, max_depth=5)
    alphabeta_action = alphabeta_agent.get_action(game)
    alphabeta_nodes = alphabeta_agent.nodes_expanded
    alphabeta_pruning = alphabeta_agent.pruning_count
    
    print(f"Minimax 算法:")
    print(f"  最佳动作: {minimax_action}")
    print(f"  节点展开数: {minimax_nodes}")
    
    print(f"\nAlpha-Beta 算法:")
    print(f"  最佳动作: {alphabeta_action}")
    print(f"  节点展开数: {alphabeta_nodes}")
    print(f"  剪枝次数: {alphabeta_pruning}")
    
    print(f"\n剪枝效果:")
    print(f"  节点减少: {minimax_nodes - alphabeta_nodes}")
    print(f"  效率提升: {((minimax_nodes - alphabeta_nodes) / minimax_nodes * 100):.1f}%")


if __name__ == "__main__":
    # 演示对抗性搜索算法
    compare_algorithms()
    
    # 演示剪枝效果
    demonstrate_pruning()
    
    # 交互式游戏演示
    print("\n" + "="*50)
    print("交互式井字棋游戏演示")
    print("=" * 50)
    
    # 人类 vs AI 的简单演示
    game = TicTacToe()
    ai_agent = AlphaBetaAgent(-1, max_depth=5)
    
    print("初始棋盘:")
    print(game)
    
    # AI 先手
    ai_action = ai_agent.get_action(game)
    game = game.make_move(ai_action)
    print(f"\nAI 选择位置: {ai_action}")
    print(game)
    
    print("\n✅ 对抗性搜索算法演示完成！") 