# 第 3 章：搜索算法 (Search Algorithms)

## 章节概述

本章节实现了《人工智能：现代方法》第 3 章的核心内容，重点介绍了启发式搜索算法，特别是 A\*算法及其变体。这些算法通过使用启发式函数来指导搜索过程，显著提高了搜索效率。

## 核心内容

### 1. 启发式搜索

- 启发式函数的概念
- 可容许性和一致性
- 启发式函数的设计

### 2. 最佳优先搜索

- 贪心最佳优先搜索
- A\*算法
- 加权 A\*算法
- IDA\*算法

### 3. 启发式函数

- 曼哈顿距离
- 欧几里得距离
- 汉明距离
- 模式数据库

### 4. 算法性能分析

- 时间复杂度
- 空间复杂度
- 最优性保证
- 完备性分析

## 实现算法

### 核心类和函数

- `HeuristicFunction`: 启发式函数抽象类
- `AStarSearch`: A\*搜索算法
- `GreedyBestFirstSearch`: 贪心最佳优先搜索
- `WeightedAStar`: 加权 A\*搜索
- `IDAStarSearch`: IDA\*搜索
- `BidirectionalSearch`: 双向搜索

### 启发式函数

- `ManhattanDistance`: 曼哈顿距离
- `EuclideanDistance`: 欧几里得距离
- `HammingDistance`: 汉明距离
- `PatternDatabase`: 模式数据库

## 文件结构

```
03-search-algorithms/
├── README.md                    # 本文件
└── implementations/
    └── search_algorithms.py    # 主要实现文件
```

## 使用方法

### 运行基本演示

```bash
# 进入章节目录
cd 03-search-algorithms

# 运行搜索算法演示
python implementations/search_algorithms.py
```

### 代码示例

```python
from implementations.search_algorithms import AStarSearch, ManhattanDistance

# 创建启发式函数
heuristic = ManhattanDistance()

# 创建A*搜索
astar = AStarSearch(heuristic)

# 求解问题
solution = astar.search(problem)
print(f"A*解决方案: {solution}")
```

## 学习目标

通过本章节的学习，你将能够：

1. 理解启发式搜索的基本原理
2. 掌握 A\*算法的实现和优化
3. 设计有效的启发式函数
4. 分析算法的性能特征
5. 选择合适的搜索策略

## 相关章节

- **前置知识**: 第 2 章（问题求解）
- **后续章节**:
  - 第 4 章：复杂环境搜索
  - 第 5 章：对抗性搜索
  - 第 6 章：约束满足问题

## 扩展阅读

- Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Chapter 3.
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths.
- Korf, R. E. (1985). Depth-first iterative-deepening: An optimal admissible tree search.

## 注意事项

- 启发式函数必须是可容许的以保证最优解
- 一致性条件比可容许性更强
- 启发式函数的质量直接影响搜索效率
