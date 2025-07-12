"""
第2章：问题求解的搜索
这个模块实现了AIMA第2章中的核心概念：问题定义、状态空间、搜索节点等
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Optional, Any, Iterator
import heapq
import time
from collections import deque
import math


class State:
    """状态类：表示问题的一个状态"""
    
    def __init__(self, data: Any):
        self.data = data
    
    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return f"State({self.data})"
    
    def __eq__(self, other):
        return isinstance(other, State) and self.data == other.data
    
    def __hash__(self):
        if isinstance(self.data, (list, dict)):
            # 对于可变类型，转换为不可变类型
            if isinstance(self.data, list):
                # 处理嵌套列表
                def to_hashable(obj):
                    if isinstance(obj, list):
                        return tuple(to_hashable(item) for item in obj)
                    elif isinstance(obj, dict):
                        return tuple(sorted(obj.items()))
                    return obj
                return hash(to_hashable(self.data))
            elif isinstance(self.data, dict):
                return hash(tuple(sorted(self.data.items())))
        return hash(self.data)


class Action:
    """动作类：表示从一个状态到另一个状态的操作"""
    
    def __init__(self, name: str, cost: float = 1.0):
        self.name = name
        self.cost = cost
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"Action({self.name}, cost={self.cost})"
    
    def __eq__(self, other):
        return isinstance(other, Action) and self.name == other.name


class Problem(ABC):
    """抽象问题类：定义搜索问题的接口"""
    
    def __init__(self, initial_state: State, goal_state: State = None):
        self.initial_state = initial_state
        self.goal_state = goal_state
    
    @abstractmethod
    def actions(self, state: State) -> List[Action]:
        """返回在给定状态下可执行的动作列表"""
        pass
    
    @abstractmethod
    def result(self, state: State, action: Action) -> State:
        """返回执行动作后的结果状态"""
        pass
    
    @abstractmethod
    def goal_test(self, state: State) -> bool:
        """测试给定状态是否为目标状态"""
        pass
    
    def path_cost(self, cost: float, state1: State, action: Action, state2: State) -> float:
        """计算路径成本（默认为累积动作成本）"""
        return cost + action.cost
    
    def heuristic(self, state: State) -> float:
        """启发式函数（默认返回0，子类可以重写）"""
        return 0


class Node:
    """搜索节点：表示搜索树中的一个节点"""
    
    def __init__(self, state: State, parent: 'Node' = None, 
                 action: Action = None, path_cost: float = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1
    
    def expand(self, problem: Problem) -> List['Node']:
        """扩展节点，返回所有子节点"""
        children = []
        for action in problem.actions(self.state):
            child_state = problem.result(self.state, action)
            child_cost = problem.path_cost(self.path_cost, self.state, action, child_state)
            child_node = Node(child_state, self, action, child_cost)
            children.append(child_node)
        return children
    
    def solution(self) -> List[Action]:
        """返回从根节点到当前节点的动作序列"""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.action)
            node = node.parent
        return list(reversed(path))
    
    def path(self) -> List['Node']:
        """返回从根节点到当前节点的节点序列"""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def __str__(self):
        return f"<Node {self.state}>"
    
    def __repr__(self):
        return f"Node(state={self.state}, cost={self.path_cost})"
    
    def __lt__(self, other):
        return self.path_cost < other.path_cost


class EightPuzzle(Problem):
    """8数码问题：经典的滑动拼图问题"""
    
    def __init__(self, initial: List[List[int]], goal: List[List[int]] = None):
        if goal is None:
            goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        
        initial_state = State(initial)
        goal_state = State(goal)
        super().__init__(initial_state, goal_state)
    
    def find_blank(self, state_data: List[List[int]]) -> Tuple[int, int]:
        """找到空白位置（0的位置）"""
        for i in range(3):
            for j in range(3):
                if state_data[i][j] == 0:
                    return i, j
        return -1, -1
    
    def actions(self, state: State) -> List[Action]:
        """返回可执行的动作（上下左右移动空白）"""
        actions = []
        i, j = self.find_blank(state.data)
        
        if i > 0:  # 可以向上移动
            actions.append(Action("UP"))
        if i < 2:  # 可以向下移动
            actions.append(Action("DOWN"))
        if j > 0:  # 可以向左移动
            actions.append(Action("LEFT"))
        if j < 2:  # 可以向右移动
            actions.append(Action("RIGHT"))
        
        return actions
    
    def result(self, state: State, action: Action) -> State:
        """执行动作后的结果状态"""
        new_state = [row[:] for row in state.data]  # 深拷贝
        i, j = self.find_blank(new_state)
        
        if action.name == "UP":
            new_state[i][j], new_state[i-1][j] = new_state[i-1][j], new_state[i][j]
        elif action.name == "DOWN":
            new_state[i][j], new_state[i+1][j] = new_state[i+1][j], new_state[i][j]
        elif action.name == "LEFT":
            new_state[i][j], new_state[i][j-1] = new_state[i][j-1], new_state[i][j]
        elif action.name == "RIGHT":
            new_state[i][j], new_state[i][j+1] = new_state[i][j+1], new_state[i][j]
        
        return State(new_state)
    
    def goal_test(self, state: State) -> bool:
        """测试是否达到目标状态"""
        return state.data == self.goal_state.data
    
    def heuristic(self, state: State) -> float:
        """曼哈顿距离启发式函数"""
        distance = 0
        for i in range(3):
            for j in range(3):
                if state.data[i][j] != 0:
                    value = state.data[i][j]
                    # 找到目标位置
                    target_i = (value - 1) // 3
                    target_j = (value - 1) % 3
                    distance += abs(i - target_i) + abs(j - target_j)
        return distance


class NQueensProblem(Problem):
    """N皇后问题：在N×N棋盘上放置N个皇后，使其互不攻击"""
    
    def __init__(self, n: int):
        self.n = n
        # 初始状态：空棋盘
        initial_state = State([])
        super().__init__(initial_state)
    
    def actions(self, state: State) -> List[Action]:
        """返回可执行的动作（在下一行放置皇后）"""
        if len(state.data) == self.n:
            return []  # 已经放置了所有皇后
        
        actions = []
        row = len(state.data)
        for col in range(self.n):
            if self.is_safe(state.data, row, col):
                actions.append(Action(f"Place_Q_{row}_{col}"))
        return actions
    
    def is_safe(self, positions: List[int], row: int, col: int) -> bool:
        """检查在(row, col)位置放置皇后是否安全"""
        for i, c in enumerate(positions):
            # 检查列冲突
            if c == col:
                return False
            # 检查对角线冲突
            if abs(i - row) == abs(c - col):
                return False
        return True
    
    def result(self, state: State, action: Action) -> State:
        """执行动作后的结果状态"""
        # 从动作名称中提取列号
        col = int(action.name.split('_')[-1])
        new_positions = state.data + [col]
        return State(new_positions)
    
    def goal_test(self, state: State) -> bool:
        """测试是否放置了所有皇后"""
        return len(state.data) == self.n
    
    def heuristic(self, state: State) -> float:
        """启发式函数：剩余需要放置的皇后数量"""
        return self.n - len(state.data)


class Romania(Problem):
    """罗马尼亚地图问题：从一个城市到另一个城市的路径规划"""
    
    def __init__(self, initial_city: str, goal_city: str):
        # 罗马尼亚城市间的连接
        self.map = {
            'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
            'Zerind': {'Arad': 75, 'Oradea': 71},
            'Oradea': {'Zerind': 71, 'Sibiu': 151},
            'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu': 80},
            'Timisoara': {'Arad': 118, 'Lugoj': 111},
            'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
            'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
            'Drobeta': {'Mehadia': 75, 'Craiova': 120},
            'Craiova': {'Drobeta': 120, 'Rimnicu': 146, 'Pitesti': 138},
            'Rimnicu': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
            'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
            'Pitesti': {'Rimnicu': 97, 'Craiova': 138, 'Bucharest': 101},
            'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
            'Giurgiu': {'Bucharest': 90},
            'Urziceni': {'Bucharest': 85, 'Vaslui': 142, 'Hirsova': 98},
            'Hirsova': {'Urziceni': 98, 'Eforie': 86},
            'Eforie': {'Hirsova': 86},
            'Vaslui': {'Urziceni': 142, 'Iasi': 92},
            'Iasi': {'Vaslui': 92, 'Neamt': 87},
            'Neamt': {'Iasi': 87}
        }
        
        # 直线距离启发式（到布加勒斯特的距离）
        self.straight_distances = {
            'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242,
            'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151,
            'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234,
            'Oradea': 380, 'Pitesti': 100, 'Rimnicu': 193, 'Sibiu': 253,
            'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
        }
        
        initial_state = State(initial_city)
        goal_state = State(goal_city)
        super().__init__(initial_state, goal_state)
    
    def actions(self, state: State) -> List[Action]:
        """返回从当前城市可以到达的城市"""
        current_city = state.data
        actions = []
        if current_city in self.map:
            for neighbor, distance in self.map[current_city].items():
                actions.append(Action(f"Go_to_{neighbor}", distance))
        return actions
    
    def result(self, state: State, action: Action) -> State:
        """执行动作后到达的城市"""
        city = action.name.split('_')[-1]
        return State(city)
    
    def goal_test(self, state: State) -> bool:
        """测试是否到达目标城市"""
        return state.data == self.goal_state.data
    
    def heuristic(self, state: State) -> float:
        """启发式函数：直线距离"""
        return self.straight_distances.get(state.data, 0)


class SearchAlgorithms:
    """搜索算法集合"""
    
    @staticmethod
    def breadth_first_search(problem: Problem) -> Optional[Node]:
        """广度优先搜索"""
        if problem.goal_test(problem.initial_state):
            return Node(problem.initial_state)
        
        frontier = deque([Node(problem.initial_state)])
        explored = set()
        
        while frontier:
            node = frontier.popleft()
            explored.add(node.state)
            
            for child in node.expand(problem):
                if child.state not in explored and child not in frontier:
                    if problem.goal_test(child.state):
                        return child
                    frontier.append(child)
        
        return None
    
    @staticmethod
    def depth_first_search(problem: Problem, limit: int = None) -> Optional[Node]:
        """深度优先搜索（可选深度限制）"""
        def recursive_dfs(node: Node, problem: Problem, limit: int) -> Optional[Node]:
            if problem.goal_test(node.state):
                return node
            elif limit == 0:
                return 'cutoff'
            else:
                cutoff_occurred = False
                for child in node.expand(problem):
                    result = recursive_dfs(child, problem, limit - 1 if limit else None)
                    if result == 'cutoff':
                        cutoff_occurred = True
                    elif result is not None:
                        return result
                return 'cutoff' if cutoff_occurred else None
        
        result = recursive_dfs(Node(problem.initial_state), problem, limit or float('inf'))
        return result if result != 'cutoff' else None
    
    @staticmethod
    def uniform_cost_search(problem: Problem) -> Optional[Node]:
        """一致代价搜索"""
        frontier = [Node(problem.initial_state)]
        heapq.heapify(frontier)
        explored = set()
        
        while frontier:
            node = heapq.heappop(frontier)
            
            if problem.goal_test(node.state):
                return node
            
            explored.add(node.state)
            
            for child in node.expand(problem):
                if child.state not in explored and not any(f.state == child.state for f in frontier):
                    heapq.heappush(frontier, child)
                elif any(f.state == child.state and f.path_cost > child.path_cost for f in frontier):
                    # 更新frontier中成本更高的节点
                    frontier = [f for f in frontier if f.state != child.state]
                    heapq.heappush(frontier, child)
                    heapq.heapify(frontier)
        
        return None


def demonstrate_eight_puzzle():
    """演示8数码问题求解"""
    print("=== 8数码问题演示 ===")
    
    # 创建一个可解的8数码问题
    initial = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    
    problem = EightPuzzle(initial, goal)
    
    print("初始状态:")
    for row in initial:
        print(row)
    
    print("\n目标状态:")
    for row in goal:
        print(row)
    
    # 使用广度优先搜索
    start_time = time.time()
    solution = SearchAlgorithms.breadth_first_search(problem)
    end_time = time.time()
    
    if solution:
        path = solution.solution()
        print(f"\n找到解决方案！步数: {len(path)}")
        print(f"搜索时间: {end_time - start_time:.4f}秒")
        print("解决步骤:", [action.name for action in path])
        print(f"启发式值: {problem.heuristic(problem.initial_state)}")
    else:
        print("\n未找到解决方案")


def demonstrate_n_queens():
    """演示N皇后问题求解"""
    print("\n=== N皇后问题演示 ===")
    
    n = 4
    problem = NQueensProblem(n)
    
    print(f"求解 {n} 皇后问题...")
    
    start_time = time.time()
    solution = SearchAlgorithms.depth_first_search(problem)
    end_time = time.time()
    
    if solution:
        positions = solution.state.data
        print(f"找到解决方案！搜索时间: {end_time - start_time:.4f}秒")
        print(f"皇后位置: {positions}")
        
        # 打印棋盘
        print("棋盘布局:")
        for i in range(n):
            row = []
            for j in range(n):
                if j == positions[i]:
                    row.append('Q')
                else:
                    row.append('.')
            print(' '.join(row))
    else:
        print("未找到解决方案")


def demonstrate_romania():
    """演示罗马尼亚地图问题求解"""
    print("\n=== 罗马尼亚地图问题演示 ===")
    
    problem = Romania('Arad', 'Bucharest')
    
    print("从 Arad 到 Bucharest 的路径规划")
    
    start_time = time.time()
    solution = SearchAlgorithms.uniform_cost_search(problem)
    end_time = time.time()
    
    if solution:
        path = solution.solution()
        print(f"找到路径！总成本: {solution.path_cost}")
        print(f"搜索时间: {end_time - start_time:.4f}秒")
        
        # 打印路径
        current_city = problem.initial_state.data
        print(f"路径: {current_city}", end="")
        for action in path:
            next_city = action.name.split('_')[-1]
            print(f" -> {next_city} (成本: {action.cost})", end="")
            current_city = next_city
        print(f"\n总距离: {solution.path_cost}km")
        print(f"启发式估计: {problem.heuristic(problem.initial_state)}")
    else:
        print("未找到路径")


def main():
    """主演示函数"""
    print("第2章：问题求解的搜索")
    print("实现了状态空间搜索的基本框架和经典问题")
    
    demonstrate_eight_puzzle()
    demonstrate_n_queens()
    demonstrate_romania()
    
    print("\n=== 搜索算法特点总结 ===")
    print("1. 广度优先搜索(BFS): 最优解，空间复杂度高")
    print("2. 深度优先搜索(DFS): 空间效率高，可能不是最优解")
    print("3. 一致代价搜索(UCS): 保证最优解，考虑路径成本")
    print("4. 启发式搜索将在第3章中详细介绍")


if __name__ == "__main__":
    main() 