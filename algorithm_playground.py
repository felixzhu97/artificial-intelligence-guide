"""
算法游乐场 - 交互式算法参数调整和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import time
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 算法基类
class Algorithm(ABC):
    """算法基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}
        self.results = {}
        self.history = []
    
    @abstractmethod
    def run(self, data: Any, **kwargs) -> Any:
        """运行算法"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """获取可调整的参数"""
        pass
    
    @abstractmethod
    def visualize(self, data: Any, results: Any) -> None:
        """可视化结果"""
        pass

# 搜索算法游乐场
class SearchAlgorithmPlayground:
    """搜索算法游乐场"""
    
    def __init__(self):
        self.algorithms = {
            'BFS': BreadthFirstSearch(),
            'DFS': DepthFirstSearch(),
            'AStar': AStarSearch(),
            'Dijkstra': DijkstraSearch()
        }
        self.current_algorithm = None
        self.current_problem = None
        self.visualization_data = []
    
    def create_maze_problem(self, width: int = 10, height: int = 10, 
                           obstacle_ratio: float = 0.3) -> 'MazeProblem':
        """创建迷宫问题"""
        maze = np.zeros((height, width))
        
        # 随机生成障碍物
        for i in range(height):
            for j in range(width):
                if random.random() < obstacle_ratio:
                    maze[i][j] = 1
        
        # 确保起点和终点可通行
        maze[0][0] = 0
        maze[height-1][width-1] = 0
        
        return MazeProblem(maze, (0, 0), (height-1, width-1))
    
    def run_algorithm(self, algorithm_name: str, problem: 'MazeProblem', 
                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """运行算法"""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"未知算法: {algorithm_name}")
        
        algorithm = self.algorithms[algorithm_name]
        
        # 设置参数
        for key, value in parameters.items():
            if key in algorithm.get_parameters():
                algorithm.parameters[key] = value
        
        # 运行算法
        start_time = time.time()
        result = algorithm.run(problem)
        end_time = time.time()
        
        # 统计结果
        stats = {
            'algorithm': algorithm_name,
            'path_length': len(result['path']) if result['path'] else 0,
            'nodes_explored': result['nodes_explored'],
            'execution_time': end_time - start_time,
            'memory_usage': result['memory_usage'],
            'success': result['path'] is not None
        }
        
        return {
            'path': result['path'],
            'stats': stats,
            'exploration_order': result['exploration_order']
        }
    
    def compare_algorithms(self, problem: 'MazeProblem', 
                          algorithms: List[str]) -> pd.DataFrame:
        """比较多个算法"""
        results = []
        
        for alg_name in algorithms:
            # 使用默认参数运行算法
            default_params = self.algorithms[alg_name].get_parameters()
            result = self.run_algorithm(alg_name, problem, default_params)
            results.append(result['stats'])
        
        return pd.DataFrame(results)
    
    def create_animated_visualization(self, problem: 'MazeProblem', 
                                    exploration_order: List[Tuple[int, int]],
                                    path: List[Tuple[int, int]]) -> FuncAnimation:
        """创建动画可视化"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 初始化显示
        maze_display = problem.maze.copy()
        im = ax.imshow(maze_display, cmap='binary', alpha=0.8)
        
        # 标记起点和终点
        ax.plot(problem.start[1], problem.start[0], 'go', markersize=15, label='起点')
        ax.plot(problem.goal[1], problem.goal[0], 'ro', markersize=15, label='终点')
        
        explored_points = ax.scatter([], [], c='yellow', s=50, alpha=0.6, label='已探索')
        path_line, = ax.plot([], [], 'b-', linewidth=3, alpha=0.8, label='最优路径')
        
        ax.set_title('搜索算法动画演示')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            if frame < len(exploration_order):
                # 显示探索过程
                explored_x = [pos[1] for pos in exploration_order[:frame+1]]
                explored_y = [pos[0] for pos in exploration_order[:frame+1]]
                explored_points.set_offsets(list(zip(explored_x, explored_y)))
            
            if frame >= len(exploration_order) and path:
                # 显示最优路径
                path_x = [pos[1] for pos in path]
                path_y = [pos[0] for pos in path]
                path_line.set_data(path_x, path_y)
            
            return [explored_points, path_line]
        
        animation = FuncAnimation(fig, animate, frames=len(exploration_order)+10,
                                interval=100, blit=True, repeat=True)
        
        return animation

# 具体算法实现
class BreadthFirstSearch(Algorithm):
    """广度优先搜索"""
    
    def __init__(self):
        super().__init__("BFS")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'max_depth': 100,
            'early_stop': True
        }
    
    def run(self, problem: 'MazeProblem', **kwargs) -> Dict[str, Any]:
        """运行BFS算法"""
        from collections import deque
        
        queue = deque([(problem.start, [problem.start])])
        visited = set([problem.start])
        exploration_order = []
        nodes_explored = 0
        
        while queue:
            current, path = queue.popleft()
            exploration_order.append(current)
            nodes_explored += 1
            
            if current == problem.goal:
                return {
                    'path': path,
                    'nodes_explored': nodes_explored,
                    'exploration_order': exploration_order,
                    'memory_usage': len(queue)
                }
            
            # 探索邻居
            for neighbor in problem.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return {
            'path': None,
            'nodes_explored': nodes_explored,
            'exploration_order': exploration_order,
            'memory_usage': 0
        }
    
    def visualize(self, data: Any, results: Any) -> None:
        """可视化BFS结果"""
        pass

class DepthFirstSearch(Algorithm):
    """深度优先搜索"""
    
    def __init__(self):
        super().__init__("DFS")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'max_depth': 100,
            'randomize': False
        }
    
    def run(self, problem: 'MazeProblem', **kwargs) -> Dict[str, Any]:
        """运行DFS算法"""
        stack = [(problem.start, [problem.start])]
        visited = set()
        exploration_order = []
        nodes_explored = 0
        
        while stack:
            current, path = stack.pop()
            
            if current in visited:
                continue
            
            visited.add(current)
            exploration_order.append(current)
            nodes_explored += 1
            
            if current == problem.goal:
                return {
                    'path': path,
                    'nodes_explored': nodes_explored,
                    'exploration_order': exploration_order,
                    'memory_usage': len(stack)
                }
            
            # 探索邻居
            neighbors = problem.get_neighbors(current)
            if self.parameters.get('randomize', False):
                random.shuffle(neighbors)
            
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
        
        return {
            'path': None,
            'nodes_explored': nodes_explored,
            'exploration_order': exploration_order,
            'memory_usage': 0
        }
    
    def visualize(self, data: Any, results: Any) -> None:
        """可视化DFS结果"""
        pass

class AStarSearch(Algorithm):
    """A*搜索"""
    
    def __init__(self):
        super().__init__("A*")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'heuristic': 'manhattan',
            'weight': 1.0
        }
    
    def run(self, problem: 'MazeProblem', **kwargs) -> Dict[str, Any]:
        """运行A*算法"""
        import heapq
        
        def heuristic(pos: Tuple[int, int]) -> float:
            if self.parameters.get('heuristic') == 'manhattan':
                return abs(pos[0] - problem.goal[0]) + abs(pos[1] - problem.goal[1])
            else:  # euclidean
                return np.sqrt((pos[0] - problem.goal[0])**2 + (pos[1] - problem.goal[1])**2)
        
        open_set = [(heuristic(problem.start), 0, problem.start, [problem.start])]
        closed_set = set()
        exploration_order = []
        nodes_explored = 0
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            exploration_order.append(current)
            nodes_explored += 1
            
            if current == problem.goal:
                return {
                    'path': path,
                    'nodes_explored': nodes_explored,
                    'exploration_order': exploration_order,
                    'memory_usage': len(open_set)
                }
            
            # 探索邻居
            for neighbor in problem.get_neighbors(current):
                if neighbor not in closed_set:
                    new_g_score = g_score + 1
                    new_f_score = new_g_score + self.parameters.get('weight', 1.0) * heuristic(neighbor)
                    heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, path + [neighbor]))
        
        return {
            'path': None,
            'nodes_explored': nodes_explored,
            'exploration_order': exploration_order,
            'memory_usage': 0
        }
    
    def visualize(self, data: Any, results: Any) -> None:
        """可视化A*结果"""
        pass

class DijkstraSearch(Algorithm):
    """Dijkstra算法"""
    
    def __init__(self):
        super().__init__("Dijkstra")
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'early_stop': True
        }
    
    def run(self, problem: 'MazeProblem', **kwargs) -> Dict[str, Any]:
        """运行Dijkstra算法"""
        import heapq
        
        distances = {problem.start: 0}
        previous = {}
        open_set = [(0, problem.start)]
        closed_set = set()
        exploration_order = []
        nodes_explored = 0
        
        while open_set:
            current_dist, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            exploration_order.append(current)
            nodes_explored += 1
            
            if current == problem.goal:
                # 重构路径
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(problem.start)
                path.reverse()
                
                return {
                    'path': path,
                    'nodes_explored': nodes_explored,
                    'exploration_order': exploration_order,
                    'memory_usage': len(open_set)
                }
            
            # 探索邻居
            for neighbor in problem.get_neighbors(current):
                if neighbor not in closed_set:
                    new_dist = current_dist + 1
                    
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current
                        heapq.heappush(open_set, (new_dist, neighbor))
        
        return {
            'path': None,
            'nodes_explored': nodes_explored,
            'exploration_order': exploration_order,
            'memory_usage': 0
        }
    
    def visualize(self, data: Any, results: Any) -> None:
        """可视化Dijkstra结果"""
        pass

# 问题定义
@dataclass
class MazeProblem:
    """迷宫问题"""
    maze: np.ndarray
    start: Tuple[int, int]
    goal: Tuple[int, int]
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取邻居节点"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
        
        for dx, dy in directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            
            if (0 <= new_x < self.maze.shape[0] and 
                0 <= new_y < self.maze.shape[1] and
                self.maze[new_x][new_y] == 0):
                neighbors.append((new_x, new_y))
        
        return neighbors

# 机器学习算法游乐场
class MLAlgorithmPlayground:
    """机器学习算法游乐场"""
    
    def __init__(self):
        self.algorithms = {
            'DecisionTree': DecisionTreePlayground(),
            'KMeans': KMeansPlayground(),
            'NeuralNetwork': NeuralNetworkPlayground()
        }
    
    def generate_classification_data(self, n_samples: int = 300, 
                                   n_features: int = 2, 
                                   n_classes: int = 3,
                                   noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """生成分类数据"""
        np.random.seed(42)
        
        # 生成类中心
        centers = np.random.uniform(-5, 5, (n_classes, n_features))
        
        X = []
        y = []
        
        samples_per_class = n_samples // n_classes
        
        for i in range(n_classes):
            # 生成每个类的样本
            class_samples = np.random.multivariate_normal(
                centers[i], 
                np.eye(n_features) * noise, 
                samples_per_class
            )
            X.extend(class_samples)
            y.extend([i] * samples_per_class)
        
        return np.array(X), np.array(y)
    
    def generate_regression_data(self, n_samples: int = 100, 
                               n_features: int = 1,
                               noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """生成回归数据"""
        np.random.seed(42)
        
        X = np.random.uniform(-5, 5, (n_samples, n_features))
        
        if n_features == 1:
            y = 2 * X.ravel() + 1 + np.random.normal(0, noise, n_samples)
        else:
            # 多特征线性关系
            coeffs = np.random.uniform(-2, 2, n_features)
            y = X @ coeffs + np.random.normal(0, noise, n_samples)
        
        return X, y

class DecisionTreePlayground:
    """决策树游乐场"""
    
    def __init__(self):
        self.tree = None
        self.training_history = []
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              max_depth: int = 5, 
              min_samples_split: int = 2,
              criterion: str = 'gini') -> Dict[str, Any]:
        """训练决策树"""
        # 简化的决策树实现
        from collections import Counter
        
        def entropy(labels):
            counts = Counter(labels)
            total = len(labels)
            return -sum((count/total) * np.log2(count/total) for count in counts.values() if count > 0)
        
        def gini(labels):
            counts = Counter(labels)
            total = len(labels)
            return 1 - sum((count/total)**2 for count in counts.values())
        
        def build_tree(X, y, depth=0):
            if depth >= max_depth or len(set(y)) == 1 or len(y) < min_samples_split:
                return {'prediction': Counter(y).most_common(1)[0][0], 'samples': len(y)}
            
            best_feature, best_threshold = None, None
            best_score = float('inf')
            
            n_features = X.shape[1]
            
            for feature in range(n_features):
                thresholds = np.unique(X[:, feature])
                
                for threshold in thresholds:
                    left_mask = X[:, feature] <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue
                    
                    left_y, right_y = y[left_mask], y[right_mask]
                    
                    if criterion == 'entropy':
                        score = (len(left_y) * entropy(left_y) + 
                                len(right_y) * entropy(right_y)) / len(y)
                    else:  # gini
                        score = (len(left_y) * gini(left_y) + 
                                len(right_y) * gini(right_y)) / len(y)
                    
                    if score < best_score:
                        best_score = score
                        best_feature = feature
                        best_threshold = threshold
            
            if best_feature is None:
                return {'prediction': Counter(y).most_common(1)[0][0], 'samples': len(y)}
            
            left_mask = X[:, best_feature] <= best_threshold
            right_mask = ~left_mask
            
            return {
                'feature': best_feature,
                'threshold': best_threshold,
                'left': build_tree(X[left_mask], y[left_mask], depth + 1),
                'right': build_tree(X[right_mask], y[right_mask], depth + 1),
                'samples': len(y)
            }
        
        self.tree = build_tree(X, y)
        
        # 计算准确率
        predictions = [self.predict_single(x) for x in X]
        accuracy = np.mean(predictions == y)
        
        return {
            'accuracy': accuracy,
            'tree_depth': self.get_tree_depth(self.tree),
            'n_nodes': self.count_nodes(self.tree)
        }
    
    def predict_single(self, x: np.ndarray) -> int:
        """单个样本预测"""
        if self.tree is None:
            return 0
        
        node = self.tree
        while 'feature' in node:
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        
        return node['prediction']
    
    def get_tree_depth(self, node: Dict) -> int:
        """获取树的深度"""
        if 'feature' not in node:
            return 1
        
        left_depth = self.get_tree_depth(node['left'])
        right_depth = self.get_tree_depth(node['right'])
        
        return 1 + max(left_depth, right_depth)
    
    def count_nodes(self, node: Dict) -> int:
        """计算节点数量"""
        if 'feature' not in node:
            return 1
        
        return 1 + self.count_nodes(node['left']) + self.count_nodes(node['right'])

class KMeansPlayground:
    """K-means游乐场"""
    
    def __init__(self):
        self.centroids = None
        self.labels = None
        self.history = []
    
    def fit(self, X: np.ndarray, k: int = 3, 
            max_iters: int = 100, 
            init_method: str = 'random') -> Dict[str, Any]:
        """训练K-means"""
        n_samples, n_features = X.shape
        
        # 初始化质心
        if init_method == 'random':
            self.centroids = X[np.random.choice(n_samples, k, replace=False)]
        elif init_method == 'kmeans++':
            self.centroids = self._kmeans_plus_plus_init(X, k)
        
        self.history = [self.centroids.copy()]
        
        for iteration in range(max_iters):
            # 分配样本到最近的质心
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # 更新质心
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(k)])
            
            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
            self.history.append(self.centroids.copy())
        
        # 计算SSE
        sse = sum(np.sum((X[self.labels == i] - self.centroids[i])**2) 
                 for i in range(k))
        
        return {
            'sse': sse,
            'iterations': len(self.history),
            'centroids': self.centroids,
            'labels': self.labels
        }
    
    def _kmeans_plus_plus_init(self, X: np.ndarray, k: int) -> np.ndarray:
        """K-means++初始化"""
        centroids = [X[np.random.randint(X.shape[0])]]
        
        for _ in range(k - 1):
            distances = np.array([min([np.sum((x - c)**2) for c in centroids]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            
            for i, prob in enumerate(cumulative_probs):
                if r < prob:
                    centroids.append(X[i])
                    break
        
        return np.array(centroids)

class NeuralNetworkPlayground:
    """神经网络游乐场"""
    
    def __init__(self):
        self.weights = []
        self.biases = []
        self.training_history = []
    
    def init_network(self, layer_sizes: List[int]):
        """初始化网络"""
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """前向传播"""
        a = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.sigmoid(z)
        return a
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              hidden_layers: List[int] = [10],
              learning_rate: float = 0.1,
              epochs: int = 100) -> Dict[str, Any]:
        """训练神经网络"""
        # 构建网络结构
        layer_sizes = [X.shape[1]] + hidden_layers + [len(np.unique(y))]
        self.init_network(layer_sizes)
        
        # 转换标签为one-hot编码
        n_classes = len(np.unique(y))
        y_one_hot = np.eye(n_classes)[y]
        
        self.training_history = []
        
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失
            loss = np.mean((output - y_one_hot)**2)
            
            # 计算准确率
            predictions = np.argmax(output, axis=1)
            accuracy = np.mean(predictions == y)
            
            self.training_history.append({
                'epoch': epoch,
                'loss': loss,
                'accuracy': accuracy
            })
            
            # 简化的反向传播（仅用于演示）
            # 实际应用中需要完整的梯度计算
            
        return {
            'final_loss': loss,
            'final_accuracy': accuracy,
            'training_history': self.training_history
        }

# 演示函数
def demo_search_playground():
    """演示搜索算法游乐场"""
    print("🔍 搜索算法游乐场演示")
    print("="*50)
    
    # 创建游乐场实例
    playground = SearchAlgorithmPlayground()
    
    # 创建迷宫问题
    problem = playground.create_maze_problem(width=8, height=8, obstacle_ratio=0.2)
    
    print(f"迷宫大小: {problem.maze.shape}")
    print(f"起点: {problem.start}")
    print(f"终点: {problem.goal}")
    
    # 测试不同算法
    algorithms = ['BFS', 'DFS', 'AStar', 'Dijkstra']
    
    for alg_name in algorithms:
        print(f"\n测试 {alg_name} 算法:")
        
        # 获取默认参数
        default_params = playground.algorithms[alg_name].get_parameters()
        
        # 运行算法
        result = playground.run_algorithm(alg_name, problem, default_params)
        
        print(f"  路径长度: {result['stats']['path_length']}")
        print(f"  探索节点数: {result['stats']['nodes_explored']}")
        print(f"  执行时间: {result['stats']['execution_time']:.4f}秒")
        print(f"  成功: {result['stats']['success']}")
    
    # 算法比较
    print("\n📊 算法比较:")
    comparison_df = playground.compare_algorithms(problem, algorithms)
    print(comparison_df.to_string(index=False))

def demo_ml_playground():
    """演示机器学习游乐场"""
    print("\n🧠 机器学习算法游乐场演示")
    print("="*50)
    
    # 创建游乐场实例
    playground = MLAlgorithmPlayground()
    
    # 生成数据
    X, y = playground.generate_classification_data(n_samples=200, n_classes=3)
    
    print(f"数据集大小: {X.shape}")
    print(f"类别数: {len(np.unique(y))}")
    
    # 决策树演示
    print("\n🌳 决策树演示:")
    dt = DecisionTreePlayground()
    
    # 测试不同参数
    depths = [3, 5, 7]
    
    for depth in depths:
        result = dt.train(X, y, max_depth=depth)
        print(f"  深度={depth}: 准确率={result['accuracy']:.3f}, 节点数={result['n_nodes']}")
    
    # K-means演示
    print("\n🎯 K-means演示:")
    kmeans = KMeansPlayground()
    
    # 测试不同K值
    k_values = [2, 3, 4, 5]
    
    for k in k_values:
        result = kmeans.fit(X, k=k)
        print(f"  K={k}: SSE={result['sse']:.2f}, 迭代次数={result['iterations']}")

if __name__ == "__main__":
    # 运行演示
    demo_search_playground()
    demo_ml_playground() 