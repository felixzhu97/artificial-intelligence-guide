# 第 8 章：一阶逻辑 (First-Order Logic)

## 章节概述

本章节实现了《人工智能：现代方法》第 8 章的核心内容，介绍一阶逻辑的语法、语义和推理方法。一阶逻辑扩展了命题逻辑，能够表示对象、属性和关系，是知识表示的重要工具。

## 核心内容

### 1. 一阶逻辑语法

- **原子句子**: 谓词和项的组合
- **复合句子**: 使用逻辑连接词连接的句子
- **量词**: 全称量词(∀)和存在量词(∃)
- **项**: 常量、变量和函数

### 2. 语义解释

- **解释**: 将语法符号映射到现实世界对象
- **满足**: 句子在解释下的真值
- **有效性**: 在所有解释下都为真的句子
- **可满足性**: 存在解释使句子为真

### 3. 知识工程

- **领域建模**: 将现实世界问题形式化
- **谓词设计**: 选择合适的谓词和函数
- **公理化**: 建立领域的基本规则
- **查询处理**: 回答关于领域的问题

## 实现算法

### 核心类和函数

- `Term`: 项的抽象类
- `Constant`: 常量项
- `Variable`: 变量项
- `Function`: 函数项
- `Predicate`: 谓词类
- `Sentence`: 一阶逻辑句子
- `Quantifier`: 量词类
- `Interpretation`: 解释类

### 语法处理

- `Parser`: 一阶逻辑表达式解析器
- `Lexer`: 词法分析器
- `SyntaxTree`: 语法树构建
- `TypeChecker`: 类型检查器

### 语义操作

- `Substitution`: 变量替换
- `Unification`: 合一算法（预备知识）
- `ModelChecker`: 模型检查
- `Satisfiability`: 可满足性检查

## 文件结构

```
08-first-order-logic/
├── README.md                      # 本文件
└── implementations/
    └── first_order_logic.py      # 主要实现文件
```

## 使用方法

### 运行基本演示

```bash
# 进入章节目录
cd 08-first-order-logic

# 运行一阶逻辑演示
python implementations/first_order_logic.py
```

### 代码示例

```python
from implementations.first_order_logic import Predicate, Constant, Variable, Sentence

# 创建谓词和项
loves = Predicate("Loves", 2)
john = Constant("John")
mary = Constant("Mary")
x = Variable("x")

# 构建原子句子
john_loves_mary = loves(john, mary)
print(f"原子句子: {john_loves_mary}")

# 使用量词
from implementations.first_order_logic import ForAll, Exists

# ∀x Loves(x, Mary) - 每个人都爱Mary
everyone_loves_mary = ForAll(x, loves(x, mary))
print(f"全称量化: {everyone_loves_mary}")

# ∃x Loves(John, x) - John爱某个人
john_loves_someone = Exists(x, loves(john, x))
print(f"存在量化: {john_loves_someone}")
```

## 学习目标

通过本章节的学习，你将能够：

1. 理解一阶逻辑的语法和语义
2. 掌握谓词、项和量词的使用
3. 学会将现实世界问题形式化为一阶逻辑
4. 理解解释和模型的概念
5. 实现基本的语法分析和语义检查
6. 进行简单的一阶逻辑推理

## 相关章节

- **前置知识**:
  - 第 7 章：逻辑代理
- **后续章节**:
  - 第 9 章：一阶逻辑推理
  - 第 10 章：知识表示

## 扩展阅读

- Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Chapter 8.
- Shoenfield, J. R. (2001). Mathematical Logic.
- Van Dalen, D. (2004). Logic and Structure.
- Mendelson, E. (2009). Mathematical Logic.

## 注意事项

- 一阶逻辑比命题逻辑表达能力更强但计算复杂度更高
- 量词的作用域需要仔细处理
- 变量绑定和自由变量的区别很重要
- 合适的谓词设计对知识表示质量影响很大
