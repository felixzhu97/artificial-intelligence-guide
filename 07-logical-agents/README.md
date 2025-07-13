# 第 7 章：逻辑代理 (Logical Agents)

## 章节概述

本章节实现了《人工智能：现代方法》第 7 章的核心内容，介绍了基于逻辑的智能代理。这些代理使用命题逻辑来表示知识、进行推理并做出决策。

## 核心内容

### 1. 命题逻辑基础

- **语法**: 原子命题、逻辑连接词、复合句子
- **语义**: 真值、模型、蕴含关系
- **推理**: 从已知事实推导新知识
- **知识库**: 存储和管理逻辑知识

### 2. 逻辑连接词

- **否定(¬)**: NOT 运算
- **合取(∧)**: AND 运算
- **析取(∨)**: OR 运算
- **蕴含(→)**: IF-THEN 关系
- **双条件(↔)**: IF-AND-ONLY-IF 关系

### 3. 推理方法

- **模型检查**: 通过枚举所有模型验证蕴含
- **推理规则**: 肯定前件、否定后件等
- **归结推理**: 基于归结的自动推理
- **前向链接**: 从事实推导结论
- **后向链接**: 从目标回溯到事实

### 4. 应用场景

- **Wumpus 世界**: 经典的逻辑推理问题
- **专家系统**: 基于规则的知识系统
- **自动定理证明**: 数学定理的机器证明
- **知识表示**: 常识知识的形式化

## 实现算法

### 核心类和函数

- `Sentence`: 命题逻辑句子基类
- `AtomicSentence`: 原子命题
- `ComplexSentence`: 复合句子
- `LogicalOperator`: 逻辑运算符枚举
- `KnowledgeBase`: 知识库管理
- `InferenceEngine`: 推理引擎
- `TruthTable`: 真值表构造和检查

### 推理算法

- `ModelChecking`: 模型检查算法
- `ResolutionInference`: 归结推理
- `ForwardChaining`: 前向链接推理
- `BackwardChaining`: 后向链接推理
- `DPLL`: Davis-Putnam-Logemann-Loveland 算法

### Wumpus 世界

- `WumpusWorldLogic`: Wumpus 世界的逻辑表示
- `WumpusAgent`: 基于逻辑的 Wumpus 代理
- `SafetyInference`: 安全性推理
- `LocationReasoning`: 位置推理

## 文件结构

```
07-logical-agents/
├── README.md                      # 本文件
└── implementations/
    └── logical_agents.py         # 主要实现文件
```

## 使用方法

### 运行基本演示

```bash
# 进入章节目录
cd 07-logical-agents

# 运行逻辑代理演示
python implementations/logical_agents.py
```

### 代码示例

```python
from implementations.logical_agents import KnowledgeBase, Sentence

# 创建知识库
kb = KnowledgeBase()

# 添加知识
kb.tell("A → B")  # 如果A则B
kb.tell("B → C")  # 如果B则C
kb.tell("A")      # A为真

# 进行推理
result = kb.ask("C")  # 询问C是否为真
print(f"C是否为真: {result}")

# Wumpus世界示例
from implementations.logical_agents import WumpusWorldLogic, WumpusAgent

# 创建Wumpus世界
world = WumpusWorldLogic(size=4)
agent = WumpusAgent(world)

# 代理探索
while not agent.done:
    # 感知当前环境
    percept = world.perceive(agent.location)

    # 更新知识库
    agent.update_knowledge(percept)

    # 推理并选择行动
    action = agent.choose_action()

    # 执行行动
    world.execute_action(action)
```

## 学习目标

通过本章节的学习，你将能够：

1. 理解命题逻辑的语法和语义
2. 掌握基本的逻辑推理方法
3. 实现知识库和推理引擎
4. 应用逻辑推理解决 Wumpus 世界问题
5. 理解前向和后向链接推理
6. 掌握归结推理的原理和实现
7. 设计基于逻辑的智能代理

## 相关章节

- **前置知识**:
  - 第 1 章：智能代理
  - 第 4 章：复杂环境
- **后续章节**:
  - 第 8 章：一阶逻辑
  - 第 9 章：一阶逻辑推理
  - 第 10 章：知识表示

## 扩展阅读

- Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Chapter 7.
- Shoenfield, J. R. (2001). Mathematical Logic.
- Chang, C. L., & Lee, R. C. T. (1973). Symbolic Logic and Mechanical Theorem Proving.
- Robinson, J. A. (1965). A machine-oriented logic based on the resolution principle.

## 注意事项

- 命题逻辑只能表示有限的知识类型
- 模型检查的复杂度是指数级的
- 归结推理要求将知识转换为合取范式
- 逻辑代理假设环境是静态的
- 需要考虑不完全信息下的推理
