# 第 2 章：问题求解的搜索 (Problem Solving)

## 章节概述

本章节实现了《人工智能：现代方法》第 2 章的核心内容，介绍了如何将问题形式化为状态空间搜索问题，并通过系统化的搜索方法找到解决方案。

## 核心内容

### 1. 问题定义

- 状态空间表示
- 初始状态和目标状态
- 动作和转移模型
- 路径代价函数

### 2. 经典问题

- 8 数码问题
- N 皇后问题
- 罗马尼亚地图问题
- 传教士与食人族问题

### 3. 搜索策略

- 搜索树和搜索图
- 边界管理
- 已访问状态检查
- 性能评估标准

## 实现算法

### 核心类和函数

- `Problem`: 问题抽象类
- `State`: 状态表示
- `Action`: 动作表示
- `Node`: 搜索节点
- `EightPuzzle`: 8 数码问题实现
- `NQueens`: N 皇后问题实现
- `RomaniaMap`: 罗马尼亚地图问题实现

### 搜索算法

- 广度优先搜索 (BFS)
- 深度优先搜索 (DFS)
- 一致代价搜索 (UCS)
- 深度限制搜索 (DLS)
- 迭代深化搜索 (IDS)

## 文件结构

```
02-problem-solving/
├── README.md                    # 本文件
└── implementations/
    └── problem_solving.py      # 主要实现文件
```

## 使用方法

### 运行基本演示

```bash
# 进入章节目录
cd 02-problem-solving

# 运行问题求解演示
python implementations/problem_solving.py
```

### 代码示例

```python
from implementations.problem_solving import EightPuzzle, breadth_first_search

# 创建8数码问题
initial_state = [1, 2, 3, 4, 0, 5, 6, 7, 8]
goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
problem = EightPuzzle(initial_state, goal_state)

# 使用广度优先搜索求解
solution = breadth_first_search(problem)
print(f"解决方案: {solution}")
```

## 学习目标

通过本章节的学习，你将能够：

1. 理解问题的状态空间表示
2. 掌握经典 AI 问题的建模方法
3. 实现基本的搜索算法
4. 分析不同搜索策略的性能
5. 选择合适的搜索方法解决具体问题

## 相关章节

- **前置知识**: 第 1 章（智能代理）
- **后续章节**:
  - 第 3 章：搜索算法
  - 第 4 章：复杂环境搜索
  - 第 5 章：对抗性搜索

## 扩展阅读

- Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Chapter 3.
- Korf, R. E. (1985). Depth-first iterative-deepening: An optimal admissible tree search.
- Pearl, J. (1984). Heuristics: Intelligent Search Strategies for Computer Problem Solving.

## 注意事项

- 状态空间的大小直接影响搜索效率
- 需要避免重复访问相同状态
- 问题的表示方法对搜索性能有重要影响
