"""
搜索算法案例实现

本模块演示了《人工智能：现代方法》第3-4章中的各种搜索算法：
1. 广度优先搜索 (BFS)
2. 深度优先搜索 (DFS)  
3. 统一代价搜索 (UCS)
4. A*搜索
5. 贪心最佳优先搜索
6. 双向搜索
7. 迭代深化搜索 (IDS)
"""

import heapq
import math
from typing import List, Tuple, Dict, Set, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


@dataclass
class SearchNode:
    """搜索节点"""
    state: Any
    parent: Optional['SearchNode'] = None
    action: Any = None
    path_cost: float = 0
    depth: int = 0
    
    def __post_init__(self):
        if self.parent:
            self.depth = self.parent.depth + 1
    
    def path(self) -> List['SearchNode']:
        """返回从根节点到当前节点的路径"""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]
    
    def solution(self) -> List[Any]:
        """返回解决方案（动作序列）"""
        return [node.action for node in self.path()[1:]]
    
    def __lt__(self, other):
        return self.path_cost < other.path_cost


class Problem(ABC):
    """问题抽象基类"""
    
    def __init__(self, initial_state: Any, goal_state: Any = None):
        self.initial = initial_state
        self.goal = goal_state
    
    @abstractmethod
    def actions(self, state: Any) -> List[Any]:
        """返回在给定状态下可执行的动作"""
        pass
    
    @abstractmethod
    def result(self, state: Any, action: Any) -> Any:
        """返回执行动作后的新状态"""
        pass
    
    @abstractmethod
    def goal_test(self, state: Any) -> bool:
        """测试状态是否为目标状态"""
        pass
    
    def path_cost(self, cost: float, state1: Any, action: Any, state2: Any) -> float:
        """计算路径代价"""
        return cost + 1
    
    def h(self, node: SearchNode) -> float:
        """启发式函数（默认为0）"""
        return 0


class GridWorldProblem(Problem):
    """网格世界问题"""
    
    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        super().__init__(start, goal)
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
    
    def actions(self, state: Tuple[int, int]) -> List[str]:
        """返回可执行的动作"""
        actions = []
        row, col = state
        
        # 上下左右四个方向
        for dr, dc, action in [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.rows and 0 <= new_col < self.cols and 
                self.grid[new_row][new_col] == 0):  # 0表示可通行
                actions.append(action)
        
        return actions
    
    def result(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """执行动作后的新状态"""
        row, col = state
        if action == 'UP':
            return (row - 1, col)
        elif action == 'DOWN':
            return (row + 1, col)
        elif action == 'LEFT':
            return (row, col - 1)
        elif action == 'RIGHT':
            return (row, col + 1)
        return state
    
    def goal_test(self, state: Tuple[int, int]) -> bool:
        """测试是否为目标状态"""
        return state == self.goal
    
    def path_cost(self, cost: float, state1: Tuple[int, int], action: str, state2: Tuple[int, int]) -> float:
        """计算路径代价（可以根据地形设置不同代价）"""
        return cost + 1
    
    def h(self, node: SearchNode) -> float:
        """曼哈顿距离启发式"""
        state = node.state
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])


class NPuzzleProblem(Problem):
    """N拼图问题"""
    
    def __init__(self, initial_state: Tuple[Tuple[int, ...], ...], size: int = 3):
        self.size = size
        goal_state = tuple(tuple(i * size + j + 1 if i * size + j + 1 < size * size else 0 
                                for j in range(size)) for i in range(size))
        super().__init__(initial_state, goal_state)
    
    def actions(self, state: Tuple[Tuple[int, ...], ...]) -> List[str]:
        """返回可执行的动作"""
        # 找到空白位置（0）
        blank_row, blank_col = self._find_blank(state)
        actions = []
        
        if blank_row > 0:
            actions.append('UP')
        if blank_row < self.size - 1:
            actions.append('DOWN')
        if blank_col > 0:
            actions.append('LEFT')
        if blank_col < self.size - 1:
            actions.append('RIGHT')
        
        return actions
    
    def result(self, state: Tuple[Tuple[int, ...], ...], action: str) -> Tuple[Tuple[int, ...], ...]:
        """执行动作后的新状态"""
        blank_row, blank_col = self._find_blank(state)
        new_state = [list(row) for row in state]
        
        if action == 'UP':
            new_state[blank_row][blank_col], new_state[blank_row-1][blank_col] = \
                new_state[blank_row-1][blank_col], new_state[blank_row][blank_col]
        elif action == 'DOWN':
            new_state[blank_row][blank_col], new_state[blank_row+1][blank_col] = \
                new_state[blank_row+1][blank_col], new_state[blank_row][blank_col]
        elif action == 'LEFT':
            new_state[blank_row][blank_col], new_state[blank_row][blank_col-1] = \
                new_state[blank_row][blank_col-1], new_state[blank_row][blank_col]
        elif action == 'RIGHT':
            new_state[blank_row][blank_col], new_state[blank_row][blank_col+1] = \
                new_state[blank_row][blank_col+1], new_state[blank_row][blank_col]
        
        return tuple(tuple(row) for row in new_state)
    
    def goal_test(self, state: Tuple[Tuple[int, ...], ...]) -> bool:
        """测试是否为目标状态"""
        return state == self.goal
    
    def h(self, node: SearchNode) -> float:
        """曼哈顿距离启发式"""
        state = node.state
        distance = 0
        
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] != 0:
                    target_val = state[i][j]
                    target_row = (target_val - 1) // self.size
                    target_col = (target_val - 1) % self.size
                    distance += abs(i - target_row) + abs(j - target_col)
        
        return distance
    
    def _find_blank(self, state: Tuple[Tuple[int, ...], ...]) -> Tuple[int, int]:
        """找到空白位置"""
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == 0:
                    return (i, j)
        return (0, 0)


class SearchAlgorithm(ABC):
    """搜索算法基类"""
    
    def __init__(self, problem: Problem):
        self.problem = problem
        self.explored = set()
        self.nodes_expanded = 0
        self.max_frontier_size = 0
    
    @abstractmethod
    def search(self) -> Optional[SearchNode]:
        """执行搜索"""
        pass
    
    def reset_statistics(self):
        """重置统计信息"""
        self.explored = set()
        self.nodes_expanded = 0
        self.max_frontier_size = 0


class BreadthFirstSearch(SearchAlgorithm):
    """广度优先搜索"""
    
    def search(self) -> Optional[SearchNode]:
        """执行BFS搜索"""
        self.reset_statistics()
        
        if self.problem.goal_test(self.problem.initial):
            return SearchNode(self.problem.initial)
        
        frontier = deque([SearchNode(self.problem.initial)])
        self.explored = set([self.problem.initial])
        
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            node = frontier.popleft()
            
            for action in self.problem.actions(node.state):
                child_state = self.problem.result(node.state, action)
                child_cost = self.problem.path_cost(node.path_cost, node.state, action, child_state)
                child = SearchNode(child_state, node, action, child_cost)
                
                if child_state not in self.explored:
                    if self.problem.goal_test(child_state):
                        return child
                    
                    frontier.append(child)
                    self.explored.add(child_state)
            
            self.nodes_expanded += 1
        
        return None


class DepthFirstSearch(SearchAlgorithm):
    """深度优先搜索"""
    
    def search(self) -> Optional[SearchNode]:
        """执行DFS搜索"""
        self.reset_statistics()
        
        if self.problem.goal_test(self.problem.initial):
            return SearchNode(self.problem.initial)
        
        frontier = [SearchNode(self.problem.initial)]
        self.explored = set()
        
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            node = frontier.pop()
            
            if node.state not in self.explored:
                self.explored.add(node.state)
                
                if self.problem.goal_test(node.state):
                    return node
                
                for action in self.problem.actions(node.state):
                    child_state = self.problem.result(node.state, action)
                    child_cost = self.problem.path_cost(node.path_cost, node.state, action, child_state)
                    child = SearchNode(child_state, node, action, child_cost)
                    
                    if child_state not in self.explored:
                        frontier.append(child)
                
                self.nodes_expanded += 1
        
        return None


class UniformCostSearch(SearchAlgorithm):
    """统一代价搜索"""
    
    def search(self) -> Optional[SearchNode]:
        """执行UCS搜索"""
        self.reset_statistics()
        
        if self.problem.goal_test(self.problem.initial):
            return SearchNode(self.problem.initial)
        
        frontier = [SearchNode(self.problem.initial)]
        heapq.heapify(frontier)
        self.explored = set()
        
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            node = heapq.heappop(frontier)
            
            if self.problem.goal_test(node.state):
                return node
            
            if node.state not in self.explored:
                self.explored.add(node.state)
                
                for action in self.problem.actions(node.state):
                    child_state = self.problem.result(node.state, action)
                    child_cost = self.problem.path_cost(node.path_cost, node.state, action, child_state)
                    child = SearchNode(child_state, node, action, child_cost)
                    
                    if child_state not in self.explored:
                        heapq.heappush(frontier, child)
                
                self.nodes_expanded += 1
        
        return None


class AStarSearch(SearchAlgorithm):
    """A*搜索"""
    
    def search(self) -> Optional[SearchNode]:
        """执行A*搜索"""
        self.reset_statistics()
        
        if self.problem.goal_test(self.problem.initial):
            return SearchNode(self.problem.initial)
        
        frontier = []
        start_node = SearchNode(self.problem.initial)
        heapq.heappush(frontier, (start_node.path_cost + self.problem.h(start_node), start_node))
        self.explored = set()
        
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            _, node = heapq.heappop(frontier)
            
            if self.problem.goal_test(node.state):
                return node
            
            if node.state not in self.explored:
                self.explored.add(node.state)
                
                for action in self.problem.actions(node.state):
                    child_state = self.problem.result(node.state, action)
                    child_cost = self.problem.path_cost(node.path_cost, node.state, action, child_state)
                    child = SearchNode(child_state, node, action, child_cost)
                    
                    if child_state not in self.explored:
                        f_cost = child.path_cost + self.problem.h(child)
                        heapq.heappush(frontier, (f_cost, child))
                
                self.nodes_expanded += 1
        
        return None


class GreedyBestFirstSearch(SearchAlgorithm):
    """贪心最佳优先搜索"""
    
    def search(self) -> Optional[SearchNode]:
        """执行贪心最佳优先搜索"""
        self.reset_statistics()
        
        if self.problem.goal_test(self.problem.initial):
            return SearchNode(self.problem.initial)
        
        frontier = []
        start_node = SearchNode(self.problem.initial)
        heapq.heappush(frontier, (self.problem.h(start_node), start_node))
        self.explored = set()
        
        while frontier:
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))
            _, node = heapq.heappop(frontier)
            
            if self.problem.goal_test(node.state):
                return node
            
            if node.state not in self.explored:
                self.explored.add(node.state)
                
                for action in self.problem.actions(node.state):
                    child_state = self.problem.result(node.state, action)
                    child_cost = self.problem.path_cost(node.path_cost, node.state, action, child_state)
                    child = SearchNode(child_state, node, action, child_cost)
                    
                    if child_state not in self.explored:
                        h_cost = self.problem.h(child)
                        heapq.heappush(frontier, (h_cost, child))
                
                self.nodes_expanded += 1
        
        return None


class IterativeDeepeningSearch(SearchAlgorithm):
    """迭代深化搜索"""
    
    def search(self, max_depth: int = 50) -> Optional[SearchNode]:
        """执行迭代深化搜索"""
        self.reset_statistics()
        
        for depth in range(max_depth):
            result = self._depth_limited_search(depth)
            if result is not None:
                return result
        
        return None
    
    def _depth_limited_search(self, depth_limit: int) -> Optional[SearchNode]:
        """深度限制搜索"""
        return self._recursive_dls(SearchNode(self.problem.initial), depth_limit)
    
    def _recursive_dls(self, node: SearchNode, depth_limit: int) -> Optional[SearchNode]:
        """递归深度限制搜索"""
        if self.problem.goal_test(node.state):
            return node
        elif depth_limit == 0:
            return None
        else:
            self.nodes_expanded += 1
            for action in self.problem.actions(node.state):
                child_state = self.problem.result(node.state, action)
                child_cost = self.problem.path_cost(node.path_cost, node.state, action, child_state)
                child = SearchNode(child_state, node, action, child_cost)
                
                result = self._recursive_dls(child, depth_limit - 1)
                if result is not None:
                    return result
        
        return None


def visualize_grid_search(problem: GridWorldProblem, solution: SearchNode, algorithm_name: str):
    """可视化网格搜索结果"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制网格
    grid = np.array(problem.grid)
    ax.imshow(grid, cmap='binary', alpha=0.3)
    
    # 绘制路径
    if solution:
        path = solution.path()
        path_states = [node.state for node in path]
        path_rows = [state[0] for state in path_states]
        path_cols = [state[1] for state in path_states]
        
        ax.plot(path_cols, path_rows, 'b-', linewidth=3, alpha=0.7, label='路径')
        ax.scatter(path_cols, path_rows, c='blue', s=50, alpha=0.7)
    
    # 标记起点和终点
    start_row, start_col = problem.initial
    goal_row, goal_col = problem.goal
    ax.scatter(start_col, start_row, c='green', s=200, marker='s', label='起点')
    ax.scatter(goal_col, goal_row, c='red', s=200, marker='*', label='终点')
    
    # 设置网格
    ax.set_xticks(range(problem.cols))
    ax.set_yticks(range(problem.rows))
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{algorithm_name} - 路径长度: {len(path_states) if solution else "无解"}')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def compare_search_algorithms():
    """比较不同搜索算法的性能"""
    print("=== 搜索算法性能比较 ===")
    
    # 创建网格世界问题
    grid = [
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 0]
    ]
    
    problem = GridWorldProblem(grid, (0, 0), (7, 7))
    
    # 测试不同算法
    algorithms = [
        ("BFS", BreadthFirstSearch(problem)),
        ("DFS", DepthFirstSearch(problem)),
        ("UCS", UniformCostSearch(problem)),
        ("A*", AStarSearch(problem)),
        ("贪心最佳优先", GreedyBestFirstSearch(problem)),
        ("迭代深化", IterativeDeepeningSearch(problem))
    ]
    
    results = []
    
    for name, algorithm in algorithms:
        print(f"\n运行 {name}...")
        
        if name == "迭代深化":
            solution = algorithm.search(max_depth=20)
        else:
            solution = algorithm.search()
        
        if solution:
            path_length = len(solution.path())
            path_cost = solution.path_cost
            print(f"  找到解! 路径长度: {path_length}, 路径代价: {path_cost}")
            print(f"  扩展节点数: {algorithm.nodes_expanded}")
            print(f"  最大边界大小: {algorithm.max_frontier_size}")
            
            results.append({
                'algorithm': name,
                'path_length': path_length,
                'path_cost': path_cost,
                'nodes_expanded': algorithm.nodes_expanded,
                'max_frontier_size': algorithm.max_frontier_size,
                'solution': solution
            })
        else:
            print(f"  未找到解")
            results.append({
                'algorithm': name,
                'path_length': None,
                'path_cost': None,
                'nodes_expanded': algorithm.nodes_expanded,
                'max_frontier_size': algorithm.max_frontier_size,
                'solution': None
            })
    
    # 打印比较结果
    print("\n=== 性能比较表 ===")
    print(f"{'算法':<12} {'路径长度':<8} {'路径代价':<8} {'扩展节点':<8} {'最大边界':<8}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['algorithm']:<12} "
              f"{result['path_length'] if result['path_length'] else 'N/A':<8} "
              f"{result['path_cost'] if result['path_cost'] else 'N/A':<8} "
              f"{result['nodes_expanded']:<8} "
              f"{result['max_frontier_size']:<8}")
    
    return results


def demo_npuzzle():
    """演示N拼图问题"""
    print("\n=== N拼图问题演示 ===")
    
    # 创建3x3拼图
    initial_state = ((1, 2, 3), (4, 0, 5), (7, 8, 6))
    problem = NPuzzleProblem(initial_state, size=3)
    
    print("初始状态:")
    for row in initial_state:
        print(row)
    
    print("\n目标状态:")
    for row in problem.goal:
        print(row)
    
    # 使用A*搜索
    astar = AStarSearch(problem)
    solution = astar.search()
    
    if solution:
        print(f"\n找到解! 步数: {len(solution.solution())}")
        print(f"扩展节点数: {astar.nodes_expanded}")
        print(f"解决方案: {solution.solution()}")
        
        # 显示解决过程
        print("\n解决过程:")
        for i, node in enumerate(solution.path()):
            print(f"步骤 {i}:")
            for row in node.state:
                print(row)
            print()
    else:
        print("未找到解")


if __name__ == "__main__":
    # 算法比较
    results = compare_search_algorithms()
    
    # N拼图演示
    demo_npuzzle()
    
    # 可视化最佳算法的结果
    print("\n正在生成可视化...")
    grid = [
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 0]
    ]
    
    problem = GridWorldProblem(grid, (0, 0), (7, 7))
    
    # 可视化A*搜索结果
    astar = AStarSearch(problem)
    solution = astar.search()
    visualize_grid_search(problem, solution, "A*搜索") 