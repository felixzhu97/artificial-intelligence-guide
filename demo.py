#!/usr/bin/env python3
"""
《人工智能：现代方法》案例集合 - 主演示脚本

这个脚本展示了项目中的主要功能和算法实现。
运行此脚本可以体验不同的AI算法和技术。
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入各模块
try:
    from 工具库.utils import set_random_seed, Logger, print_progress
    print("✓ 工具库加载成功")
except ImportError as e:
    print(f"✗ 工具库加载失败: {e}")

try:
    from 工具库.data_structures import Graph, PriorityQueue, Tree
    print("✓ 数据结构库加载成功")
except ImportError as e:
    print(f"✗ 数据结构库加载失败: {e}")


def print_banner():
    """打印项目横幅"""
    banner = """
    ================================================
           人工智能：现代方法 - 案例集合
    ================================================
    
    基于Stuart Russell和Peter Norvig的经典教科书
    《人工智能：现代方法》的完整实现案例
    
    包含以下主要内容：
    📖 智能代理
    🔍 搜索算法
    🧠 机器学习
    🎯 推荐系统
    🎮 强化学习
    
    ================================================
    """
    print(banner)


def demo_basic_components():
    """演示基础组件"""
    print("\n=== 基础组件演示 ===")
    
    # 设置随机种子
    set_random_seed(42)
    print("✓ 随机种子设置完成")
    
    # 创建日志记录器
    logger = Logger()
    logger.info("开始基础组件演示")
    
    # 演示数据结构
    print("\n📊 数据结构演示:")
    
    # 图结构
    print("  • 创建图结构...")
    graph = Graph()
    graph.add_edge('A', 'B', 1)
    graph.add_edge('B', 'C', 2)
    graph.add_edge('A', 'C', 3)
    print(f"    图信息: {graph}")
    
    # 优先队列
    print("  • 创建优先队列...")
    pq = PriorityQueue()
    pq.push("任务1", 3)
    pq.push("任务2", 1)
    pq.push("任务3", 2)
    print(f"    优先队列大小: {pq.size()}")
    print(f"    最高优先级任务: {pq.pop()}")
    
    # 树结构
    print("  • 创建树结构...")
    root = Tree("根节点")
    child1 = Tree("子节点1")
    child2 = Tree("子节点2")
    root.add_child(child1)
    root.add_child(child2)
    print(f"    树的大小: {root.size()}")
    print(f"    树的高度: {root.height()}")
    
    logger.info("基础组件演示完成")


def demo_intelligent_agents():
    """演示智能代理"""
    print("\n=== 智能代理演示 ===")
    
    try:
        # 动态导入以避免缺失模块的问题
        exec("""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '01-智能代理', '案例实现'))

try:
    from simple_agent import SimpleReflexAgent, Environment, Action
    
    print("  • 创建环境...")
    env = Environment(4, 4, dirt_prob=0.3)
    print(f"    环境大小: {env.width}x{env.height}")
    
    print("  • 创建简单反射代理...")
    agent = SimpleReflexAgent()
    
    print("  • 运行简化仿真...")
    total_reward = 0
    for step in range(20):
        percept = env.get_percept()
        action = agent.choose_action(percept)
        new_percept, reward = env.execute_action(action)
        total_reward += reward
        
        if step % 5 == 0:
            print(f"    步骤 {step}: 位置 {percept.location}, 动作 {action.value}")
    
    print(f"    总奖励: {total_reward:.2f}")
    print("✓ 智能代理演示完成")
    
except ImportError as e:
    print(f"✗ 智能代理模块导入失败: {e}")
except Exception as e:
    print(f"✗ 智能代理演示失败: {e}")
""")
    
    except Exception as e:
        print(f"✗ 智能代理演示失败: {e}")


def demo_search_algorithms():
    """演示搜索算法"""
    print("\n=== 搜索算法演示 ===")
    
    try:
        # 简化的搜索算法演示
        print("  • 演示简化的搜索算法...")
        
        # 创建简单的图搜索问题
        class SimpleSearchProblem:
            def __init__(self):
                self.graph = {
                    'A': [('B', 1), ('C', 4)],
                    'B': [('D', 2), ('E', 3)],
                    'C': [('F', 2)],
                    'D': [('G', 1)],
                    'E': [('G', 2)],
                    'F': [('G', 3)],
                    'G': []
                }
                self.start = 'A'
                self.goal = 'G'
            
            def get_neighbors(self, node):
                return self.graph.get(node, [])
        
        problem = SimpleSearchProblem()
        
        # 简单的广度优先搜索
        from collections import deque
        
        def simple_bfs(problem):
            queue = deque([(problem.start, [problem.start])])
            visited = set()
            
            while queue:
                node, path = queue.popleft()
                
                if node in visited:
                    continue
                    
                visited.add(node)
                
                if node == problem.goal:
                    return path
                
                for neighbor, cost in problem.get_neighbors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
            
            return None
        
        print("  • 运行广度优先搜索...")
        path = simple_bfs(problem)
        
        if path:
            print(f"    找到路径: {' -> '.join(path)}")
            print(f"    路径长度: {len(path)}")
        else:
            print("    未找到路径")
        
        print("✓ 搜索算法演示完成")
        
    except Exception as e:
        print(f"✗ 搜索算法演示失败: {e}")


def demo_machine_learning():
    """演示机器学习"""
    print("\n=== 机器学习演示 ===")
    
    try:
        # 简化的机器学习演示
        print("  • 生成示例数据...")
        
        # 生成简单的分类数据
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        print(f"    数据集大小: {n_samples}")
        print(f"    特征数: {X.shape[1]}")
        print(f"    类别分布: {np.bincount(y)}")
        
        # 简单的感知机算法
        class SimplePerceptron:
            def __init__(self, learning_rate=0.1):
                self.learning_rate = learning_rate
                self.weights = None
                self.bias = None
            
            def fit(self, X, y, epochs=100):
                n_features = X.shape[1]
                self.weights = np.zeros(n_features)
                self.bias = 0
                
                for epoch in range(epochs):
                    for i in range(len(X)):
                        prediction = self.predict(X[i].reshape(1, -1))[0]
                        if prediction != y[i]:
                            self.weights += self.learning_rate * (y[i] - prediction) * X[i]
                            self.bias += self.learning_rate * (y[i] - prediction)
            
            def predict(self, X):
                return (np.dot(X, self.weights) + self.bias >= 0).astype(int)
        
        print("  • 训练感知机...")
        perceptron = SimplePerceptron()
        perceptron.fit(X, y)
        
        # 计算准确率
        predictions = perceptron.predict(X)
        accuracy = np.mean(predictions == y)
        
        print(f"    训练准确率: {accuracy:.2%}")
        print("✓ 机器学习演示完成")
        
    except Exception as e:
        print(f"✗ 机器学习演示失败: {e}")


def demo_data_analysis():
    """演示数据分析"""
    print("\n=== 数据分析演示 ===")
    
    try:
        # 生成示例数据
        np.random.seed(42)
        data = {
            'users': np.random.randint(1, 101, 1000),
            'items': np.random.randint(1, 51, 1000),
            'ratings': np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.1, 0.1, 0.2, 0.3, 0.3])
        }
        
        print("  • 生成推荐系统数据...")
        print(f"    用户数: {len(set(data['users']))}")
        print(f"    物品数: {len(set(data['items']))}")
        print(f"    评分数: {len(data['ratings'])}")
        
        # 基础统计
        avg_rating = np.mean(data['ratings'])
        rating_dist = np.bincount(data['ratings'])[1:]  # 去掉0
        
        print(f"    平均评分: {avg_rating:.2f}")
        print(f"    评分分布: {dict(enumerate(rating_dist, 1))}")
        
        # 简单的协同过滤
        print("  • 实现简单协同过滤...")
        
        # 创建用户-物品矩阵
        user_item_matrix = {}
        for user, item, rating in zip(data['users'], data['items'], data['ratings']):
            if user not in user_item_matrix:
                user_item_matrix[user] = {}
            user_item_matrix[user][item] = rating
        
        # 计算用户相似度（简化版）
        def calculate_similarity(user1_ratings, user2_ratings):
            common_items = set(user1_ratings.keys()) & set(user2_ratings.keys())
            if len(common_items) == 0:
                return 0
            
            sum_squares = sum([(user1_ratings[item] - user2_ratings[item]) ** 2 
                              for item in common_items])
            return 1 / (1 + sum_squares)
        
        sample_users = list(user_item_matrix.keys())[:5]
        similarities = {}
        
        for i, user1 in enumerate(sample_users):
            for user2 in sample_users[i+1:]:
                sim = calculate_similarity(user_item_matrix[user1], user_item_matrix[user2])
                similarities[(user1, user2)] = sim
        
        if similarities:
            avg_similarity = np.mean(list(similarities.values()))
            print(f"    平均用户相似度: {avg_similarity:.3f}")
        
        print("✓ 数据分析演示完成")
        
    except Exception as e:
        print(f"✗ 数据分析演示失败: {e}")


def demo_visualization():
    """演示可视化功能"""
    print("\n=== 可视化演示 ===")
    
    try:
        # 生成示例数据
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        
        print("  • 创建示例图表...")
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 函数图
        ax1.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
        ax1.plot(x, y2, label='cos(x)', color='red', linewidth=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('三角函数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 散点图
        np.random.seed(42)
        x_scatter = np.random.randn(100)
        y_scatter = x_scatter + np.random.randn(100) * 0.5
        ax2.scatter(x_scatter, y_scatter, alpha=0.6, color='green')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('随机散点图')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('demo_visualization.png', dpi=300, bbox_inches='tight')
        print("    图表已保存为 'demo_visualization.png'")
        
        # 显示图表（如果在交互环境中）
        try:
            plt.show()
        except:
            pass
        
        print("✓ 可视化演示完成")
        
    except Exception as e:
        print(f"✗ 可视化演示失败: {e}")


def print_project_summary():
    """打印项目总结"""
    summary = """
    ================================================
                    项目总结
    ================================================
    
    🎯 已实现的核心功能：
    
    📖 智能代理 (01-智能代理/)
       • 简单反射代理
       • 基于模型的代理
       • 基于目标的代理
       • 基于效用的代理
    
    🔍 搜索算法 (02-问题求解/)
       • 广度优先搜索 (BFS)
       • 深度优先搜索 (DFS)
       • A*搜索算法
       • 启发式搜索
    
    🧠 机器学习 (05-机器学习/)
       • 决策树算法
       • 神经网络实现
       • 强化学习算法
       • 评估和可视化
    
    🎯 推荐系统 (项目案例/)
       • 协同过滤
       • 基于内容的推荐
       • 混合推荐策略
       • 性能评估
    
    🛠️ 工具库 (工具库/)
       • 通用工具函数
       • 数据结构实现
       • 可视化工具
       • 算法工具
    
    ================================================
    
    🚀 快速开始：
    
    1. 安装依赖：pip install -r requirements.txt
    2. 运行演示：python demo.py
    3. 查看案例：访问对应的目录
    4. 阅读文档：查看 README.md
    
    ================================================
    
    📚 学习建议：
    
    • 按照目录顺序循序渐进学习
    • 结合教科书理论知识
    • 动手实践修改代码
    • 尝试解决实际问题
    
    ================================================
    """
    print(summary)


def main():
    """主函数"""
    print_banner()
    
    # 演示各个组件
    demo_basic_components()
    demo_intelligent_agents()
    demo_search_algorithms()
    demo_machine_learning()
    demo_data_analysis()
    demo_visualization()
    
    # 打印项目总结
    print_project_summary()
    
    print("\n🎉 演示完成！欢迎探索更多功能！")


if __name__ == "__main__":
    main() 