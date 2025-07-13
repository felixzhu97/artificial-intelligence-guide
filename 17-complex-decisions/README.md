# 第 17 章：复杂决策 (Complex Decisions)

## 章节概述

本章节实现了《人工智能：现代方法》第 17 章的核心内容，重点介绍了马尔可夫决策过程(MDP)及其求解方法。MDP 是处理不确定性环境中序列决策问题的重要框架。

## 核心内容

### 1. 马尔可夫决策过程

- **MDP 的数学定义**: 状态、动作、转移概率、奖励函数
- **马尔可夫性质**: 未来只依赖于当前状态
- **策略**: 从状态到动作的映射
- **价值函数**: 评估状态和策略的好坏

### 2. 求解方法

- **价值迭代算法**: 通过迭代更新价值函数求解最优策略
- **策略迭代算法**: 策略评估和策略改进的交替过程
- **线性规划方法**: 将 MDP 转化为线性规划问题
- **修正策略迭代**: 结合价值迭代和策略迭代的优点

### 3. 算法分析

- **收敛性**: 算法的收敛保证和条件
- **计算复杂度**: 时间和空间复杂度分析
- **近似方法**: 处理大状态空间的近似算法
- **参数敏感性**: 折扣因子等参数的影响

### 4. 应用领域

- **机器人导航**: 路径规划和避障
- **资源管理**: 库存控制和资源分配
- **投资决策**: 投资组合优化
- **游戏 AI**: 策略游戏中的决策

## 实现算法

### 核心类和函数

- `MDP`: 马尔可夫决策过程类
- `State`: 状态表示
- `Action`: 动作表示
- `ValueIteration`: 价值迭代算法
- `PolicyIteration`: 策略迭代算法
- `MDPVisualization`: MDP 可视化工具

### 求解算法

- **价值迭代**: 通过迭代更新价值函数直到收敛
- **策略迭代**: 策略评估和策略改进的交替过程
- **修正策略迭代**: 结合两种方法的优点
- **异步动态规划**: 异步更新状态价值

### 示例环境

- `GridWorld`: 网格世界 MDP
- `RaceTrack`: 赛车轨道问题
- `InventoryControl`: 库存控制问题
- `InvestmentMDP`: 投资决策问题

## 文件结构

```
17-complex-decisions/
├── README.md                    # 本文件
└── implementations/
    └── complex_decisions.py    # 主要实现文件
```

## 使用方法

### 运行基本演示

```bash
# 进入章节目录
cd 17-complex-decisions

# 运行复杂决策演示
python implementations/complex_decisions.py
```

### 代码示例

```python
from implementations.complex_decisions import MDP, ValueIteration, PolicyIteration

# 创建网格世界MDP
mdp = MDP(width=4, height=3, discount_factor=0.9)

# 设置终端状态和奖励
mdp.set_terminal_state(3, 2, 1.0)   # 目标状态：奖励+1
mdp.set_terminal_state(3, 1, -1.0)  # 陷阱状态：奖励-1

# 设置障碍物
mdp.set_obstacle(1, 1)

# 使用价值迭代求解
vi = ValueIteration(mdp, tolerance=1e-6)
vi_values = vi.solve()
print(f"价值迭代收敛次数: {vi.iteration_count}")

# 使用策略迭代求解
pi = PolicyIteration(mdp, tolerance=1e-6)
pi_values, pi_policy = pi.solve()
print(f"策略迭代收敛次数: {pi.iteration_count}")

# 可视化结果
from implementations.complex_decisions import MDPVisualization
viz = MDPVisualization(mdp)
viz.plot_values(vi_values, "价值迭代 - 状态价值函数")
viz.plot_policy(vi.policy, "价值迭代 - 最优策略")
viz.plot_convergence(vi.convergence_history)
```

## 学习目标

通过本章节的学习，你将能够：

1. 理解马尔可夫决策过程的数学基础
2. 掌握价值迭代和策略迭代算法
3. 分析算法的收敛性和复杂度
4. 设计 MDP 模型来解决实际问题
5. 评估不同策略的性能
6. 理解折扣因子对最优策略的影响
7. 应用 MDP 解决实际决策问题

## 相关章节

- **前置知识**:
  - 第 16 章：简单决策
  - 第 12-13 章：概率推理
  - 第 4 章：复杂环境
- **后续章节**:
  - 第 18 章：多代理决策
  - 第 22 章：强化学习

## 扩展阅读

- Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Chapter 17.
- Bellman, R. (1957). Dynamic Programming.
- Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming.
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.

## 注意事项

- 折扣因子的选择影响最优策略的特性
- 状态空间的大小决定了算法的计算复杂度
- 转移概率的准确性对结果的可靠性很重要
- 奖励函数的设计需要仔细考虑业务目标
- 对于大规模问题需要考虑近似方法
