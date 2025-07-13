# 第 5 章：对抗性搜索 (Adversarial Search)

## 章节概述

本章节实现了《人工智能：现代方法》第 5 章的核心内容，介绍了在竞争环境中的搜索算法。这些算法主要用于双人零和博弈，其中一个代理的收益就是另一个代理的损失。

## 核心内容

### 1. 博弈理论基础

- **零和博弈**: 一方收益等于另一方损失
- **完美信息博弈**: 所有信息对双方都可见
- **博弈树**: 表示所有可能的博弈状态
- **终端效用**: 博弈结束时的收益评估

### 2. 核心算法

- **Minimax 算法**: 最优决策的基础算法
- **Alpha-Beta 剪枝**: Minimax 的优化版本
- **期望 Minimax**: 处理随机因素的博弈
- **蒙特卡洛树搜索(MCTS)**: 现代博弈 AI 的核心

### 3. 实际应用

- **井字棋(Tic-Tac-Toe)**: 经典的完美信息博弈
- **四子连珠(Connect Four)**: 稍复杂的策略游戏
- **国际象棋**: 复杂的完美信息博弈
- **围棋**: MCTS 的重要应用领域

## 实现算法

### 核心类和函数

- `Game`: 博弈抽象基类
- `GameState`: 博弈状态表示
- `Player`: 玩家类型枚举
- `MinimaxAgent`: Minimax 算法实现
- `AlphaBetaAgent`: Alpha-Beta 剪枝实现
- `MCTSAgent`: 蒙特卡洛树搜索实现
- `MCTSNode`: MCTS 节点类

### 具体游戏实现

- `TicTacToe`: 井字棋游戏
- `ConnectFour`: 四子连珠游戏
- `GameTree`: 通用博弈树结构
- `EvaluationFunction`: 启发式评估函数

## 文件结构

```
05-adversarial-search/
├── README.md                        # 本文件
└── implementations/
    └── adversarial_search.py       # 主要实现文件
```

## 使用方法

### 运行基本演示

```bash
# 进入章节目录
cd 05-adversarial-search

# 运行对抗性搜索演示
python implementations/adversarial_search.py
```

### 代码示例

```python
from implementations.adversarial_search import TicTacToe, MinimaxAgent, MCTSAgent

# 创建井字棋游戏
game = TicTacToe()

# 创建智能体
minimax_agent = MinimaxAgent(depth=9)
mcts_agent = MCTSAgent(simulations=1000)

# 运行游戏
state = game.initial_state()
while not game.is_terminal(state):
    if state.to_move == 'X':
        action = minimax_agent.get_action(state)
    else:
        action = mcts_agent.get_action(state)

    state = game.result(state, action)
    print(f"执行动作: {action}")
    game.display(state)

winner = game.utility(state, 'X')
print(f"游戏结果: {winner}")
```

## 学习目标

通过本章节的学习，你将能够：

1. 理解博弈论的基本概念和原理
2. 掌握 Minimax 算法的实现和优化
3. 理解 Alpha-Beta 剪枝的工作原理
4. 实现蒙特卡洛树搜索算法
5. 设计有效的博弈评估函数
6. 分析不同算法的时间复杂度
7. 应用算法解决实际博弈问题

## 相关章节

- **前置知识**:
  - 第 2 章：问题求解
  - 第 3 章：搜索算法
- **后续章节**:
  - 第 18 章：多代理决策
  - 第 22 章：强化学习

## 扩展阅读

- Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Chapter 5.
- Browne, C., et al. (2012). A Survey of Monte Carlo Tree Search Methods.
- Campbell, M., Hoane Jr, A. J., & Hsu, F. H. (2002). Deep Blue.
- Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search.

## 注意事项

- Minimax 假设对手是完全理性的
- Alpha-Beta 剪枝不影响最终结果，只提高效率
- MCTS 在大状态空间中表现更好
- 评估函数的质量直接影响算法性能
- 搜索深度的选择需要平衡精度和效率
