# 第 6 章：约束满足问题 (Constraint Satisfaction Problems)

## 章节概述

本章节实现了《人工智能：现代方法》第 6 章的核心内容，介绍了约束满足问题(CSP)的求解方法。CSP 是一类重要的搜索问题，通过约束传播和回溯搜索来高效求解。

## 核心内容

### 1. CSP 基本概念

- **变量(Variables)**: 需要赋值的对象
- **域(Domains)**: 每个变量的可能取值
- **约束(Constraints)**: 变量间的限制关系
- **完整赋值**: 所有变量都有值的解

### 2. 约束类型

- **一元约束**: 限制单个变量的值
- **二元约束**: 涉及两个变量的约束
- **全局约束**: 涉及多个变量的复杂约束
- **软约束**: 可以违反但有代价的约束

### 3. 求解算法

- **回溯搜索**: 深度优先的系统搜索
- **约束传播**: 通过推理减少搜索空间
- **弧一致性(AC-3)**: 最重要的约束传播算法
- **前向检查**: 在搜索过程中的约束检查

### 4. 经典问题

- **N 皇后问题**: 在 N×N 棋盘上放置 N 个皇后
- **数独**: 9×9 网格的数字填充问题
- **图着色**: 为图的顶点分配颜色
- **课程安排**: 时间表调度问题

## 实现算法

### 核心类和函数

- `CSP`: 约束满足问题抽象类
- `Variable`: 变量表示类
- `Constraint`: 约束抽象类
- `BinaryConstraint`: 二元约束实现
- `BacktrackingSearch`: 回溯搜索算法
- `AC3`: 弧一致性算法
- `ForwardChecking`: 前向检查算法

### 具体问题实现

- `NQueensProblem`: N 皇后问题
- `SudokuProblem`: 数独问题
- `GraphColoringProblem`: 图着色问题
- `SchedulingProblem`: 调度问题

### 启发式方法

- **变量选择**: MRV(最小剩余值)、度启发式
- **值选择**: LCV(最小约束值)
- **约束传播**: AC-3、前向检查、维持弧一致性

## 文件结构

```
06-constraint-satisfaction/
├── README.md                                # 本文件
└── implementations/
    └── constraint_satisfaction.py          # 主要实现文件
```

## 使用方法

### 运行基本演示

```bash
# 进入章节目录
cd 06-constraint-satisfaction

# 运行约束满足问题演示
python implementations/constraint_satisfaction.py
```

### 代码示例

```python
from implementations.constraint_satisfaction import NQueensProblem, BacktrackingSearch

# 创建8皇后问题
n_queens = NQueensProblem(n=8)

# 使用回溯搜索求解
solver = BacktrackingSearch(n_queens)
solution = solver.solve()

if solution:
    print("8皇后问题的解:")
    n_queens.display_solution(solution)
else:
    print("无解")

# 数独求解示例
from implementations.constraint_satisfaction import SudokuProblem

# 创建数独问题(部分填充的9x9网格)
initial_grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    # ... 更多行
]

sudoku = SudokuProblem(initial_grid)
sudoku_solver = BacktrackingSearch(sudoku, use_ac3=True)
sudoku_solution = sudoku_solver.solve()
```

## 学习目标

通过本章节的学习，你将能够：

1. 理解约束满足问题的基本概念
2. 掌握回溯搜索算法的实现
3. 理解约束传播的原理和应用
4. 实现 AC-3 算法和前向检查
5. 设计有效的变量和值选择启发式
6. 解决经典的 CSP 问题
7. 分析算法的时间复杂度和优化方法

## 相关章节

- **前置知识**:
  - 第 2 章：问题求解
  - 第 3 章：搜索算法
- **后续章节**:
  - 第 7 章：逻辑代理
  - 第 19 章：从样本学习

## 扩展阅读

- Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Chapter 6.
- Dechter, R. (2003). Constraint Processing.
- Rossi, F., Van Beek, P., & Walsh, T. (Eds.). (2006). Handbook of constraint programming.
- Mackworth, A. K. (1977). Consistency in networks of relations.

## 注意事项

- 变量和值的选择顺序显著影响搜索效率
- 约束传播可以大大减少搜索空间
- AC-3 算法的时间复杂度是 O(cd³)，其中 c 是约束数，d 是域大小
- 对于某些问题，局部搜索可能比回溯搜索更有效
