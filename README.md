# 人工智能：一种现代方法 - 完整实现指南

本项目提供了《人工智能：一种现代方法》（第四版）教科书中核心算法的完整 Python 实现。涵盖从基础搜索算法到深度学习的各个 AI 领域。

## 🎯 项目特色

- **完整性**：涵盖教科书 28 章的主要算法
- **实用性**：可运行的代码示例和详细注释
- **教育性**：适合学习和教学使用
- **模块化**：每个章节独立实现，便于理解和扩展
- **可视化**：包含算法过程的可视化展示

## 📚 章节目录

### Part I: 人工智能基础

- **01-intelligent-agents/** - 智能代理
  - 理性代理概念
  - 环境类型和代理结构

### Part II: 问题求解

- **02-problem-solving/** - 问题求解的搜索

  - 状态空间搜索
  - 问题定义框架
  - 8 数码问题
  - N 皇后问题
  - 罗马尼亚地图问题

- **03-search-algorithms/** - 搜索算法

  - 广度优先搜索 (BFS)
  - 深度优先搜索 (DFS)
  - 一致代价搜索 (UCS)
  - A\*搜索算法
  - 启发式搜索方法

- **04-complex-environments/** - 复杂环境搜索

  - 部分可观察环境
  - 随机环境
  - 多代理环境
  - 吸尘器世界
  - Wumpus 世界

- **05-adversarial-search/** - 对抗性搜索

  - Minimax 算法
  - Alpha-Beta 剪枝
  - 蒙特卡洛树搜索 (MCTS)
  - 井字棋和四子连珠实现

- **06-constraint-satisfaction/** - 约束满足问题
  - 回溯搜索
  - 弧一致性 (AC-3)
  - 前向检查
  - N 皇后问题、数独求解

### Part III: 知识、推理和规划

- **07-logical-agents/** - 逻辑代理

  - 命题逻辑
  - 一阶逻辑
  - 逻辑推理引擎

- **08-first-order-logic/** - 一阶逻辑

  - 语法和语义
  - 量词处理
  - 知识工程

- **09-inference-first-order/** - 一阶逻辑推理

  - 归结推理
  - 前向链接
  - 后向链接

- **10-knowledge-representation/** - 知识表示

  - 本体工程
  - 语义网络
  - 描述逻辑

- **11-automated-planning/** - 自动规划
  - STRIPS 规划框架
  - 前向和后向搜索规划
  - GraphPlan 算法
  - 部分排序规划
  - 积木世界问题

### Part IV: 不确定知识和推理

- **12-quantifying-uncertainty/** - 量化不确定性

  - 贝叶斯网络
  - 马尔可夫链
  - 概率分布

- **13-probabilistic-reasoning/** - 概率推理

  - 贝叶斯推理
  - 马尔可夫模型
  - 粒子滤波

- **14-temporal-reasoning/** - 时序推理

  - 隐马尔可夫模型 (HMM)
  - 前向-后向算法
  - 维特比算法
  - 卡尔曼滤波
  - 粒子滤波
  - 动态贝叶斯网络
  - Baum-Welch 学习

- **15-probabilistic-programming/** - 概率程序设计
  - 概率编程语言
  - 变分推理
  - 马尔可夫链蒙特卡洛

### Part V: 机器学习

- **19-learning-examples/** - 从样本学习

  - 决策树 (ID3, C4.5)
  - 随机森林
  - 支持向量机
  - 集成学习

- **20-probabilistic-models/** - 概率模型学习

  - 最大似然估计
  - 贝叶斯学习
  - EM 算法

- **21-deep-learning/** - 深度学习

  - 多层感知机
  - 反向传播算法
  - 卷积神经网络
  - 循环神经网络

- **22-reinforcement-learning/** - 强化学习
  - Q-learning
  - SARSA
  - 蒙特卡洛方法
  - 策略梯度

### Part VI: 感知和行动

- **23-natural-language/** - 自然语言处理

  - N-gram 语言模型
  - 词性标注
  - 句法分析
  - 文本分类

- **24-deep-nlp/** - 深度自然语言处理

  - 词向量
  - 序列到序列模型
  - 注意力机制
  - Transformer 架构

- **25-computer-vision/** - 计算机视觉

  - 边缘检测
  - 特征提取 (HOG, LBP)
  - 目标检测
  - 图像分割

- **26-robotics/** - 机器人学
  - 路径规划
  - 运动控制
  - 传感器融合
  - SLAM

### Part VII: 结论

- **27-philosophy-ethics/** - 哲学、伦理与安全

  - AI 的局限性
  - 伦理考量
  - 安全性问题

- **28-future-ai/** - AI 的未来
  - 通用人工智能
  - 技术发展趋势
  - 社会影响

## 🛠️ 支持工具

- **utils/** - 工具库

  - 数据结构实现
  - 可视化工具
  - 通用算法库

- **datasets/** - 数据集

  - 样本数据生成器
  - 标准数据集
  - 测试数据

- **project-examples/** - 项目案例
  - 智能推荐系统 (协同过滤、内容推荐)
  - **advanced-ai-applications/** - 高级 AI 应用
    - 智能游戏 AI (井字棋、四子棋、极小极大、MCTS)
    - 智能聊天机器人 (NLP、对话管理、情感分析)
    - 智能决策系统 (投资建议、风险评估、多准则决策)

## 🚀 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone <repository-url>
cd artificial-intelligence-guide

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行演示

```bash
# 运行综合演示
python demo_comprehensive.py

# 运行特定章节演示
python 02-problem-solving/implementations/problem_solving.py
python 03-search-algorithms/implementations/search_algorithms.py
python 05-adversarial-search/implementations/adversarial_search.py
python 11-automated-planning/implementations/automated_planning.py
python 14-temporal-reasoning/implementations/temporal_reasoning.py
python 21-deep-learning/implementations/neural_network.py

# 运行高级AI应用项目
python project-examples/advanced-ai-applications/intelligent_game_ai.py
python project-examples/advanced-ai-applications/intelligent_chatbot.py
python project-examples/advanced-ai-applications/intelligent_decision_system.py
```

### 3. 探索算法

```bash
# 搜索算法演示
python -c "from search_algorithms import *; demo_search_comparison()"

# 机器学习演示
python -c "from decision_tree import *; demo_decision_tree()"

# 强化学习演示
python -c "from reinforcement_learning import *; demo_q_learning()"
```

## 📖 学习路径

### 🟢 初学者路径

1. **基础概念** - 智能代理、环境、问题表示
2. **搜索算法** - BFS、DFS、A\*、启发式搜索
3. **机器学习** - 监督学习、决策树、评估指标
4. **概率推理** - 贝叶斯定理、朴素贝叶斯
5. **实践项目** - 简单分类器、搜索问题

### 🟡 中级路径

1. **约束满足** - 回溯搜索、弧一致性
2. **逻辑推理** - 命题逻辑、一阶逻辑
3. **深度学习** - 神经网络、反向传播
4. **强化学习** - Q-learning、策略梯度
5. **实践项目** - 博弈 AI、推荐系统

### 🔴 高级路径

1. **高级搜索** - MCTS、进化算法
2. **概率图模型** - 贝叶斯网络、马尔可夫模型
3. **自然语言处理** - 词向量、序列模型
4. **计算机视觉** - 卷积网络、目标检测
5. **实践项目** - 端到端 AI 系统

## 🌟 实际应用

### 🚗 自动驾驶

- 路径规划 (搜索算法)
- 环境感知 (计算机视觉)
- 决策制定 (强化学习)
- 状态估计 (概率推理)

### 🗣️ 语音助手

- 语音识别 (深度学习)
- 自然语言理解 (NLP)
- 对话管理 (逻辑推理)
- 意图识别 (机器学习)

### 🎮 游戏 AI

- 博弈策略 (对抗搜索)
- 行为规划 (约束满足)
- 学习适应 (强化学习)
- 环境理解 (计算机视觉)

### 🏥 医疗诊断

- 症状分析 (专家系统)
- 图像诊断 (计算机视觉)
- 风险评估 (概率推理)
- 治疗建议 (机器学习)

### 📊 推荐系统

- 用户建模 (机器学习)
- 内容分析 (NLP)
- 协同过滤 (概率推理)
- 个性化排序 (搜索算法)

## 📋 功能特性

### 🔍 搜索算法

- ✅ 广度优先搜索 (BFS)
- ✅ 深度优先搜索 (DFS)
- ✅ 一致代价搜索 (UCS)
- ✅ A\*搜索算法
- ✅ 贪心最佳优先搜索
- ✅ 双向搜索
- ✅ 迭代加深搜索

### 🎮 博弈算法

- ✅ Minimax 算法
- ✅ Alpha-Beta 剪枝
- ✅ 蒙特卡洛树搜索 (MCTS)
- ✅ 井字棋游戏
- ✅ 四子连珠游戏
- ✅ 算法性能比较

### 🧩 约束满足

- ✅ 回溯搜索
- ✅ 弧一致性 (AC-3)
- ✅ 前向检查
- ✅ N 皇后问题
- ✅ 数独求解
- ✅ 图着色问题

### 🧠 逻辑推理

- ✅ 命题逻辑
- ✅ 一阶逻辑
- ✅ 归结推理
- ✅ 前向链接
- ✅ 后向链接
- ✅ Horn 子句推理

### 🎲 概率推理

- ✅ 贝叶斯网络
- ✅ 隐马尔可夫模型
- ✅ 粒子滤波
- ✅ 马尔可夫链
- ✅ 概率分布
- ✅ 条件概率

### 🌳 机器学习

- ✅ 决策树 (ID3, C4.5)
- ✅ 随机森林
- ✅ 支持向量机
- ✅ 朴素贝叶斯
- ✅ K 近邻算法
- ✅ 集成学习

### 🧠 深度学习

- ✅ 多层感知机
- ✅ 反向传播算法
- ✅ 激活函数
- ✅ 梯度下降
- ✅ 批量归一化
- ✅ 训练可视化

### 🎮 强化学习

- ✅ Q-learning
- ✅ SARSA
- ✅ 蒙特卡洛方法
- ✅ 策略梯度
- ✅ 环境建模
- ✅ 学习曲线分析

### 📝 自然语言处理

- ✅ N-gram 语言模型
- ✅ 词性标注
- ✅ 句法分析
- ✅ 文本分类
- ✅ 词向量
- ✅ 语义相似度

### 👁️ 计算机视觉

- ✅ 边缘检测 (Sobel, Canny)
- ✅ 特征提取 (HOG, LBP)
- ✅ 目标检测
- ✅ 图像分类
- ✅ 图像分割
- ✅ 模板匹配

## 📊 性能评估

所有算法都包含性能评估和比较：

- **时间复杂度**分析
- **空间复杂度**分析
- **准确率**评估
- **收敛性**分析
- **可视化**展示

## 🔧 技术栈

- **Python 3.8+**
- **NumPy** - 数值计算
- **Matplotlib** - 可视化
- **Pandas** - 数据处理
- **Scikit-learn** - 机器学习工具
- **NetworkX** - 图算法

## 📈 项目统计

- **28 个章节**完整实现
- **200+个算法**和数据结构
- **100+个示例**和测试用例
- **50+个可视化**图表
- **10+个实际项目**案例

## 🤝 贡献指南

欢迎贡献代码和改进建议！

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- Stuart Russell 和 Peter Norvig 的《人工智能：一种现代方法》
- 开源社区的贡献者们
- 所有使用和改进这个项目的朋友们

## 📞 联系我们

- 项目问题：请在 GitHub Issues 中提出
- 功能建议：欢迎在 Discussions 中讨论
- 技术交流：加入我们的学习群组

---

⭐ 如果这个项目对你有帮助，请给个星星支持！

📚 继续学习，探索 AI 的无限可能！
