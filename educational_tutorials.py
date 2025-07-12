"""
交互式教育教程 - AI算法学习平台
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import random
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from pathlib import Path

@dataclass
class Question:
    """问题数据结构"""
    question: str
    options: List[str]
    correct_answer: int
    explanation: str
    difficulty: str  # 'easy', 'medium', 'hard'
    category: str

@dataclass
class LearningProgress:
    """学习进度数据结构"""
    user_name: str
    completed_tutorials: List[str]
    quiz_scores: Dict[str, float]
    total_time_spent: float
    last_activity: str

class Tutorial(ABC):
    """教程基类"""
    
    def __init__(self, title: str, description: str, difficulty: str):
        self.title = title
        self.description = description
        self.difficulty = difficulty
        self.sections = []
        self.exercises = []
        self.quiz_questions = []
    
    @abstractmethod
    def get_content(self) -> List[Dict[str, Any]]:
        """获取教程内容"""
        pass
    
    @abstractmethod
    def get_interactive_demo(self) -> Any:
        """获取交互式演示"""
        pass

class SearchAlgorithmTutorial(Tutorial):
    """搜索算法教程"""
    
    def __init__(self):
        super().__init__(
            title="搜索算法详解",
            description="学习各种搜索算法的原理和实现",
            difficulty="medium"
        )
        self._create_content()
    
    def _create_content(self):
        """创建教程内容"""
        self.sections = [
            {
                "title": "1. 搜索算法概述",
                "content": """
搜索算法是人工智能中的基础技术，用于在问题空间中寻找解决方案。

主要概念：
- 状态空间：所有可能状态的集合
- 搜索树：表示搜索过程的树结构
- 搜索策略：决定如何扩展节点的方法

常见搜索算法分类：
1. 无信息搜索（盲目搜索）
   - 广度优先搜索 (BFS)
   - 深度优先搜索 (DFS)
   - 一致代价搜索 (UCS)

2. 有信息搜索（启发式搜索）
   - 贪婪最佳优先搜索
   - A*搜索
   - IDA*搜索
                """,
                "code_example": """
# 搜索问题的基本框架
class SearchProblem:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
    
    def get_successors(self, state):
        # 返回后继状态列表
        pass
    
    def is_goal(self, state):
        # 判断是否为目标状态
        return state == self.goal_state
    
    def get_cost(self, state1, action, state2):
        # 返回从state1到state2的代价
        return 1
                """
            },
            {
                "title": "2. 广度优先搜索 (BFS)",
                "content": """
广度优先搜索是一种系统性的搜索策略，它按层次顺序探索搜索树。

特点：
- 使用队列(FIFO)存储待探索的节点
- 保证找到最浅的解（最少步数）
- 时间复杂度：O(b^d)，空间复杂度：O(b^d)
- 其中b是分支因子，d是解的深度

算法步骤：
1. 将起始状态加入队列
2. 从队列中取出一个状态
3. 如果是目标状态，返回解
4. 否则将其所有后继状态加入队列
5. 重复步骤2-4直到找到解或队列为空
                """,
                "code_example": """
from collections import deque

def bfs(problem):
    queue = deque([(problem.initial_state, [])])
    visited = set([problem.initial_state])
    
    while queue:
        state, path = queue.popleft()
        
        if problem.is_goal(state):
            return path + [state]
        
        for successor in problem.get_successors(state):
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, path + [state]))
    
    return None  # 无解
                """
            },
            {
                "title": "3. 深度优先搜索 (DFS)",
                "content": """
深度优先搜索优先探索搜索树的深度方向。

特点：
- 使用栈(LIFO)存储待探索的节点
- 可能找到较深的解而非最优解
- 时间复杂度：O(b^m)，空间复杂度：O(bm)
- 其中m是搜索树的最大深度

算法步骤：
1. 将起始状态加入栈
2. 从栈中取出一个状态
3. 如果是目标状态，返回解
4. 否则将其所有后继状态加入栈
5. 重复步骤2-4直到找到解或栈为空
                """,
                "code_example": """
def dfs(problem, max_depth=1000):
    stack = [(problem.initial_state, [], 0)]
    visited = set()
    
    while stack:
        state, path, depth = stack.pop()
        
        if state in visited or depth > max_depth:
            continue
        
        visited.add(state)
        
        if problem.is_goal(state):
            return path + [state]
        
        for successor in problem.get_successors(state):
            if successor not in visited:
                stack.append((successor, path + [state], depth + 1))
    
    return None  # 无解
                """
            },
            {
                "title": "4. A*搜索算法",
                "content": """
A*搜索是最重要的有信息搜索算法之一，结合了实际代价和启发式估计。

关键概念：
- g(n)：从起始状态到状态n的实际代价
- h(n)：从状态n到目标状态的启发式估计
- f(n) = g(n) + h(n)：评估函数

启发式函数的要求：
- 可接受性：h(n) ≤ h*(n)，其中h*(n)是真实代价
- 一致性：h(n) ≤ c(n,n') + h(n')

算法特点：
- 在可接受启发式下保证找到最优解
- 使用优先队列按f值排序
- 效率优于无信息搜索
                """,
                "code_example": """
import heapq

def a_star(problem, heuristic):
    open_set = [(heuristic(problem.initial_state), 0, problem.initial_state, [])]
    closed_set = set()
    
    while open_set:
        f_score, g_score, state, path = heapq.heappop(open_set)
        
        if state in closed_set:
            continue
        
        closed_set.add(state)
        
        if problem.is_goal(state):
            return path + [state]
        
        for successor in problem.get_successors(state):
            if successor not in closed_set:
                new_g = g_score + problem.get_cost(state, None, successor)
                new_f = new_g + heuristic(successor)
                heapq.heappush(open_set, (new_f, new_g, successor, path + [state]))
    
    return None  # 无解
                """
            }
        ]
        
        # 创建练习题
        self.exercises = [
            {
                "title": "8数码问题实现",
                "description": "实现8数码问题的状态表示和后继函数",
                "starter_code": """
class EightPuzzle:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    
    def get_successors(self, state):
        # TODO: 实现获取后继状态的函数
        pass
    
    def is_goal(self, state):
        # TODO: 实现目标检测函数
        pass
                """,
                "solution": """
class EightPuzzle:
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    
    def get_successors(self, state):
        successors = []
        zero_pos = state.index(0)
        row, col = zero_pos // 3, zero_pos % 3
        
        # 上下左右移动
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_pos = new_row * 3 + new_col
                new_state = state[:]
                new_state[zero_pos], new_state[new_pos] = new_state[new_pos], new_state[zero_pos]
                successors.append(new_state)
        
        return successors
    
    def is_goal(self, state):
        return state == self.goal_state
                """
            }
        ]
        
        # 创建测验题
        self.quiz_questions = [
            Question(
                question="广度优先搜索使用什么数据结构来存储待探索的节点？",
                options=["栈", "队列", "堆", "链表"],
                correct_answer=1,
                explanation="广度优先搜索使用队列(FIFO)来确保按层次顺序探索节点。",
                difficulty="easy",
                category="搜索算法"
            ),
            Question(
                question="A*搜索算法的评估函数f(n)等于什么？",
                options=["g(n)", "h(n)", "g(n) + h(n)", "g(n) * h(n)"],
                correct_answer=2,
                explanation="A*算法的评估函数f(n) = g(n) + h(n)，其中g(n)是实际代价，h(n)是启发式估计。",
                difficulty="medium",
                category="搜索算法"
            ),
            Question(
                question="以下哪种搜索算法保证找到最优解？",
                options=["深度优先搜索", "广度优先搜索", "A*搜索（可接受启发式）", "贪婪最佳优先搜索"],
                correct_answer=2,
                explanation="A*搜索在使用可接受启发式函数时保证找到最优解。",
                difficulty="hard",
                category="搜索算法"
            )
        ]
    
    def get_content(self) -> List[Dict[str, Any]]:
        """获取教程内容"""
        return self.sections
    
    def get_interactive_demo(self) -> Any:
        """获取交互式演示"""
        return SearchAlgorithmDemo()

class MachineLearningTutorial(Tutorial):
    """机器学习教程"""
    
    def __init__(self):
        super().__init__(
            title="机器学习基础",
            description="学习机器学习的基本概念和常用算法",
            difficulty="medium"
        )
        self._create_content()
    
    def _create_content(self):
        """创建教程内容"""
        self.sections = [
            {
                "title": "1. 机器学习概述",
                "content": """
机器学习是人工智能的一个分支，使计算机能够从数据中学习而无需明确编程。

机器学习的主要类型：
1. 监督学习：使用标记数据训练模型
   - 分类：预测离散标签
   - 回归：预测连续数值

2. 无监督学习：从无标记数据中发现模式
   - 聚类：将数据分成不同组
   - 降维：减少特征数量

3. 强化学习：通过与环境交互学习最优策略

学习过程：
1. 数据收集和预处理
2. 特征工程
3. 模型选择和训练
4. 模型评估和优化
                """,
                "code_example": """
# 机器学习的基本流程
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. 数据准备
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 模型训练
model = SomeMLModel()
model.fit(X_train_scaled, y_train)

# 4. 模型评估
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
                """
            },
            {
                "title": "2. 决策树算法",
                "content": """
决策树是一种基于树结构的分类和回归算法。

优点：
- 易于理解和解释
- 不需要数据预处理
- 能够处理数值和类别特征
- 可以识别重要特征

缺点：
- 容易过拟合
- 对数据变化敏感
- 偏向于选择有更多分割点的特征

关键概念：
- 信息增益：衡量特征分割的质量
- 熵：衡量数据集的不纯度
- 剪枝：防止过拟合的技术
                """,
                "code_example": """
import math
from collections import Counter

def calculate_entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    entropy = 0
    
    for count in counts.values():
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)
    
    return entropy

def information_gain(data, labels, feature_idx):
    total_entropy = calculate_entropy(labels)
    
    # 按特征值分割数据
    unique_values = set(row[feature_idx] for row in data)
    weighted_entropy = 0
    
    for value in unique_values:
        subset_labels = [labels[i] for i, row in enumerate(data) 
                        if row[feature_idx] == value]
        weight = len(subset_labels) / len(labels)
        weighted_entropy += weight * calculate_entropy(subset_labels)
    
    return total_entropy - weighted_entropy
                """
            }
        ]
        
        self.quiz_questions = [
            Question(
                question="以下哪个不是监督学习的任务？",
                options=["分类", "回归", "聚类", "预测"],
                correct_answer=2,
                explanation="聚类是无监督学习任务，不需要标记数据。",
                difficulty="easy",
                category="机器学习"
            ),
            Question(
                question="决策树算法中，信息增益用于什么？",
                options=["计算准确率", "选择最佳分割特征", "剪枝", "预测"],
                correct_answer=1,
                explanation="信息增益用于选择能最大化信息量的特征作为分割点。",
                difficulty="medium",
                category="机器学习"
            )
        ]
    
    def get_content(self) -> List[Dict[str, Any]]:
        """获取教程内容"""
        return self.sections
    
    def get_interactive_demo(self) -> Any:
        """获取交互式演示"""
        return MachineLearningDemo()

class SearchAlgorithmDemo:
    """搜索算法交互式演示"""
    
    def __init__(self):
        self.grid_size = 8
        self.grid = None
        self.start = (0, 0)
        self.goal = (7, 7)
    
    def create_maze(self, obstacle_ratio=0.3):
        """创建迷宫"""
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # 随机添加障碍物
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) != self.start and (i, j) != self.goal:
                    if random.random() < obstacle_ratio:
                        self.grid[i][j] = 1
    
    def visualize_search(self, algorithm='BFS'):
        """可视化搜索过程"""
        if self.grid is None:
            self.create_maze()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制迷宫
        ax.imshow(self.grid, cmap='binary', alpha=0.8)
        
        # 标记起点和终点
        ax.plot(self.start[1], self.start[0], 'go', markersize=15, label='起点')
        ax.plot(self.goal[1], self.goal[0], 'ro', markersize=15, label='终点')
        
        # 运行搜索算法
        if algorithm == 'BFS':
            path, explored = self._run_bfs()
        elif algorithm == 'DFS':
            path, explored = self._run_dfs()
        elif algorithm == 'A*':
            path, explored = self._run_astar()
        
        # 绘制探索过程
        if explored:
            explored_y = [pos[0] for pos in explored]
            explored_x = [pos[1] for pos in explored]
            ax.scatter(explored_x, explored_y, c='yellow', s=30, alpha=0.6, label='已探索')
        
        # 绘制路径
        if path:
            path_y = [pos[0] for pos in path]
            path_x = [pos[1] for pos in path]
            ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, label='路径')
        
        ax.set_title(f'{algorithm} 搜索演示')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _run_bfs(self):
        """运行BFS算法"""
        from collections import deque
        
        queue = deque([(self.start, [self.start])])
        visited = set([self.start])
        explored = []
        
        while queue:
            current, path = queue.popleft()
            explored.append(current)
            
            if current == self.goal:
                return path, explored
            
            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None, explored
    
    def _run_dfs(self):
        """运行DFS算法"""
        stack = [(self.start, [self.start])]
        visited = set()
        explored = []
        
        while stack:
            current, path = stack.pop()
            
            if current in visited:
                continue
            
            visited.add(current)
            explored.append(current)
            
            if current == self.goal:
                return path, explored
            
            for neighbor in self._get_neighbors(current):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
        
        return None, explored
    
    def _run_astar(self):
        """运行A*算法"""
        import heapq
        
        def heuristic(pos):
            return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
        
        open_set = [(heuristic(self.start), 0, self.start, [self.start])]
        closed_set = set()
        explored = []
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            explored.append(current)
            
            if current == self.goal:
                return path, explored
            
            for neighbor in self._get_neighbors(current):
                if neighbor not in closed_set:
                    new_g = g_score + 1
                    new_f = new_g + heuristic(neighbor)
                    heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))
        
        return None, explored
    
    def _get_neighbors(self, pos):
        """获取邻居节点"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dr, dc in directions:
            new_r, new_c = pos[0] + dr, pos[1] + dc
            
            if (0 <= new_r < self.grid_size and 
                0 <= new_c < self.grid_size and 
                self.grid[new_r][new_c] == 0):
                neighbors.append((new_r, new_c))
        
        return neighbors

class MachineLearningDemo:
    """机器学习交互式演示"""
    
    def __init__(self):
        self.data = None
        self.labels = None
        self.model = None
    
    def generate_data(self, n_samples=300, n_features=2, n_classes=3):
        """生成示例数据"""
        np.random.seed(42)
        
        # 生成不同类别的数据
        data = []
        labels = []
        
        for i in range(n_classes):
            # 每个类别的中心点
            center = np.random.uniform(-5, 5, n_features)
            
            # 生成该类别的样本
            class_data = np.random.multivariate_normal(
                center, np.eye(n_features), n_samples // n_classes
            )
            
            data.extend(class_data)
            labels.extend([i] * (n_samples // n_classes))
        
        self.data = np.array(data)
        self.labels = np.array(labels)
    
    def visualize_data(self):
        """可视化数据"""
        if self.data is None:
            self.generate_data()
        
        if self.data.shape[1] == 2:
            plt.figure(figsize=(10, 8))
            
            colors = ['red', 'blue', 'green', 'yellow', 'purple']
            
            for i in range(len(np.unique(self.labels))):
                mask = self.labels == i
                plt.scatter(self.data[mask, 0], self.data[mask, 1], 
                           c=colors[i % len(colors)], alpha=0.7, label=f'类别 {i}')
            
            plt.xlabel('特征 1')
            plt.ylabel('特征 2')
            plt.title('数据可视化')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def train_decision_tree(self, max_depth=5):
        """训练决策树"""
        if self.data is None:
            self.generate_data()
        
        # 简单的决策树实现
        from sklearn.tree import DecisionTreeClassifier
        
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.model.fit(self.data, self.labels)
        
        # 计算准确率
        predictions = self.model.predict(self.data)
        accuracy = np.mean(predictions == self.labels)
        
        print(f"决策树训练完成，准确率: {accuracy:.3f}")
        
        return accuracy
    
    def visualize_decision_boundary(self):
        """可视化决策边界"""
        if self.model is None or self.data.shape[1] != 2:
            return
        
        plt.figure(figsize=(12, 8))
        
        # 创建网格
        h = 0.02
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 预测网格点
        grid_predictions = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        grid_predictions = grid_predictions.reshape(xx.shape)
        
        # 绘制决策边界
        plt.contourf(xx, yy, grid_predictions, alpha=0.3, cmap='viridis')
        
        # 绘制数据点
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        
        for i in range(len(np.unique(self.labels))):
            mask = self.labels == i
            plt.scatter(self.data[mask, 0], self.data[mask, 1], 
                       c=colors[i % len(colors)], alpha=0.8, 
                       edgecolors='black', label=f'类别 {i}')
        
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title('决策树决策边界')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

class EducationalPlatform:
    """教育平台主类"""
    
    def __init__(self):
        self.tutorials = {
            'search': SearchAlgorithmTutorial(),
            'ml': MachineLearningTutorial()
        }
        self.user_progress = {}
        self.current_user = None
    
    def register_user(self, username: str):
        """注册用户"""
        if username not in self.user_progress:
            self.user_progress[username] = LearningProgress(
                user_name=username,
                completed_tutorials=[],
                quiz_scores={},
                total_time_spent=0.0,
                last_activity=time.strftime('%Y-%m-%d %H:%M:%S')
            )
        self.current_user = username
    
    def start_tutorial(self, tutorial_name: str):
        """开始教程"""
        if tutorial_name not in self.tutorials:
            print(f"教程 '{tutorial_name}' 不存在")
            return
        
        tutorial = self.tutorials[tutorial_name]
        print(f"\n{'='*50}")
        print(f"开始教程: {tutorial.title}")
        print(f"难度: {tutorial.difficulty}")
        print(f"描述: {tutorial.description}")
        print(f"{'='*50}")
        
        # 显示教程内容
        for section in tutorial.get_content():
            print(f"\n{section['title']}")
            print("-" * len(section['title']))
            print(section['content'])
            
            if 'code_example' in section:
                print("\n代码示例:")
                print("```python")
                print(section['code_example'])
                print("```")
            
            input("\n按Enter键继续...")
    
    def take_quiz(self, tutorial_name: str):
        """参加测验"""
        if tutorial_name not in self.tutorials:
            print(f"教程 '{tutorial_name}' 不存在")
            return
        
        tutorial = self.tutorials[tutorial_name]
        questions = tutorial.quiz_questions
        
        if not questions:
            print("此教程没有测验题")
            return
        
        print(f"\n{'='*50}")
        print(f"{tutorial.title} - 测验")
        print(f"{'='*50}")
        
        correct_answers = 0
        total_questions = len(questions)
        
        for i, question in enumerate(questions, 1):
            print(f"\n问题 {i}/{total_questions}:")
            print(question.question)
            print()
            
            for j, option in enumerate(question.options):
                print(f"{j + 1}. {option}")
            
            while True:
                try:
                    answer = int(input("\n请选择答案 (1-4): ")) - 1
                    if 0 <= answer < len(question.options):
                        break
                    else:
                        print("请输入有效选项")
                except ValueError:
                    print("请输入数字")
            
            if answer == question.correct_answer:
                print("✅ 正确!")
                correct_answers += 1
            else:
                print("❌ 错误!")
                print(f"正确答案: {question.options[question.correct_answer]}")
            
            print(f"解释: {question.explanation}")
            input("\n按Enter键继续...")
        
        score = correct_answers / total_questions
        print(f"\n测验完成!")
        print(f"得分: {correct_answers}/{total_questions} ({score:.1%})")
        
        # 更新用户进度
        if self.current_user:
            self.user_progress[self.current_user].quiz_scores[tutorial_name] = score
    
    def show_progress(self):
        """显示学习进度"""
        if not self.current_user:
            print("请先注册用户")
            return
        
        progress = self.user_progress[self.current_user]
        
        print(f"\n{'='*50}")
        print(f"学习进度 - {progress.user_name}")
        print(f"{'='*50}")
        
        print(f"已完成教程: {len(progress.completed_tutorials)}")
        for tutorial in progress.completed_tutorials:
            print(f"  ✅ {tutorial}")
        
        print(f"\n测验成绩:")
        for tutorial, score in progress.quiz_scores.items():
            print(f"  {tutorial}: {score:.1%}")
        
        print(f"\n总学习时间: {progress.total_time_spent:.1f} 小时")
        print(f"最后活动: {progress.last_activity}")
    
    def run_interactive_demo(self, tutorial_name: str):
        """运行交互式演示"""
        if tutorial_name not in self.tutorials:
            print(f"教程 '{tutorial_name}' 不存在")
            return
        
        tutorial = self.tutorials[tutorial_name]
        demo = tutorial.get_interactive_demo()
        
        print(f"\n{'='*50}")
        print(f"交互式演示: {tutorial.title}")
        print(f"{'='*50}")
        
        if isinstance(demo, SearchAlgorithmDemo):
            demo.create_maze()
            demo.visualize_search('BFS')
            demo.visualize_search('A*')
        elif isinstance(demo, MachineLearningDemo):
            demo.generate_data()
            demo.visualize_data()
            demo.train_decision_tree()
            demo.visualize_decision_boundary()

# 演示函数
def demo_educational_platform():
    """演示教育平台"""
    print("🎓 教育平台演示")
    print("="*50)
    
    # 创建平台实例
    platform = EducationalPlatform()
    
    # 注册用户
    platform.register_user("学习者")
    
    # 开始搜索算法教程
    print("\n开始搜索算法教程...")
    platform.start_tutorial('search')
    
    # 参加测验
    print("\n参加搜索算法测验...")
    platform.take_quiz('search')
    
    # 运行交互式演示
    print("\n运行交互式演示...")
    platform.run_interactive_demo('search')
    
    # 显示学习进度
    platform.show_progress()

if __name__ == "__main__":
    demo_educational_platform() 