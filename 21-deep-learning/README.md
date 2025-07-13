# 第 21 章：深度学习 (Deep Learning)

## 章节概述

本章节实现了《人工智能：现代方法》第 21 章的核心内容，介绍深度神经网络和现代深度学习方法。深度学习在图像识别、自然语言处理等领域取得了突破性进展。

## 核心内容

### 1. 神经网络基础

- **感知机**: 最简单的神经网络单元
- **多层感知机**: 具有隐藏层的前馈网络
- **激活函数**: Sigmoid、ReLU、Tanh 等
- **损失函数**: 均方误差、交叉熵等

### 2. 深度网络架构

- **卷积神经网络(CNN)**: 用于图像处理
- **循环神经网络(RNN)**: 用于序列数据
- **长短期记忆网络(LSTM)**: 解决梯度消失问题
- **注意力机制**: 提高模型表现的关键技术

### 3. 训练算法

- **反向传播**: 计算梯度的核心算法
- **梯度下降**: 优化网络参数
- **批量归一化**: 加速训练和提高稳定性
- **Dropout**: 防止过拟合的正则化技术

### 4. 优化技术

- **Adam 优化器**: 自适应学习率方法
- **学习率调度**: 动态调整学习率
- **权重初始化**: 影响训练效果
- **正则化**: L1、L2 正则化和早停

## 实现算法

### 核心类和函数

- `NeuralNetwork`: 神经网络基类
- `Layer`: 网络层抽象类
- `DenseLayer`: 全连接层
- `ConvolutionalLayer`: 卷积层
- `RNNLayer`: 循环神经网络层
- `ActivationFunction`: 激活函数类
- `LossFunction`: 损失函数类
- `Optimizer`: 优化器基类

### 具体实现

- `MultiLayerPerceptron`: 多层感知机
- `ConvolutionalNeuralNetwork`: 卷积神经网络
- `RecurrentNeuralNetwork`: 循环神经网络
- `LSTMNetwork`: LSTM 网络
- `Autoencoder`: 自编码器

### 训练工具

- `BackpropagationTrainer`: 反向传播训练器
- `DataLoader`: 数据加载器
- `ModelEvaluator`: 模型评估器
- `Visualizer`: 训练过程可视化

## 文件结构

```
21-deep-learning/
├── README.md                    # 本文件
└── implementations/
    └── neural_network.py       # 主要实现文件
```

## 使用方法

### 运行基本演示

```bash
# 进入章节目录
cd 21-deep-learning

# 运行深度学习演示
python implementations/neural_network.py
```

### 代码示例

```python
from implementations.neural_network import MultiLayerPerceptron, ReLU, CrossEntropy

# 创建多层感知机
mlp = MultiLayerPerceptron(
    input_size=784,  # 28x28 MNIST图像
    hidden_layers=[128, 64],
    output_size=10,  # 10个数字类别
    activation=ReLU(),
    loss_function=CrossEntropy()
)

# 训练模型
mlp.train(X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001)

# 评估模型
accuracy = mlp.evaluate(X_test, y_test)
print(f"测试准确率: {accuracy:.3f}")

# 卷积神经网络示例
from implementations.neural_network import ConvolutionalNeuralNetwork

cnn = ConvolutionalNeuralNetwork()
cnn.add_conv_layer(filters=32, kernel_size=3, activation='relu')
cnn.add_pooling_layer(pool_size=2)
cnn.add_conv_layer(filters=64, kernel_size=3, activation='relu')
cnn.add_pooling_layer(pool_size=2)
cnn.add_dense_layer(128, activation='relu')
cnn.add_output_layer(10, activation='softmax')

cnn.compile(optimizer='adam', loss='categorical_crossentropy')
cnn.fit(X_train, y_train, validation_data=(X_val, y_val))
```

## 学习目标

通过本章节的学习，你将能够：

1. 理解深度神经网络的基本原理
2. 掌握反向传播算法的实现
3. 了解不同网络架构的特点和应用
4. 学会训练和优化深度学习模型
5. 实现卷积神经网络和循环神经网络
6. 应用正则化技术防止过拟合
7. 评估和调优模型性能

## 相关章节

- **前置知识**:
  - 第 19 章：从样本学习
  - 第 20 章：概率模型学习
- **后续章节**:
  - 第 24 章：深度自然语言处理
  - 第 25 章：计算机视觉

## 扩展阅读

- Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.). Chapter 21.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks.

## 注意事项

- 深度学习需要大量数据和计算资源
- 超参数调优对模型性能影响很大
- 梯度消失和梯度爆炸是常见问题
- 正则化技术对防止过拟合很重要
- GPU 加速可以显著提高训练速度
