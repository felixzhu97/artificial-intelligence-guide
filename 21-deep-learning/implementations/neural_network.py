"""
神经网络案例实现

本模块演示了《人工智能：现代方法》第20章中的神经网络算法：
1. 多层感知机 (MLP)
2. 前向传播
3. 反向传播
4. 不同激活函数
5. 优化算法
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional
from abc import ABC, abstractmethod
import seaborn as sns
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time


class ActivationFunction(ABC):
    """激活函数基类"""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        pass
    
    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """反向传播（导数）"""
        pass


class Sigmoid(ActivationFunction):
    """Sigmoid激活函数"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)


class ReLU(ActivationFunction):
    """ReLU激活函数"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Tanh(ActivationFunction):
    """Tanh激活函数"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class LeakyReLU(ActivationFunction):
    """LeakyReLU激活函数"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)


class Layer:
    """神经网络层"""
    
    def __init__(self, input_size: int, output_size: int, activation: ActivationFunction):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # 初始化权重和偏置
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        
        # 缓存用于反向传播
        self.last_input = None
        self.last_output = None
        self.last_z = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.last_input = x
        self.last_z = np.dot(x, self.weights) + self.bias
        self.last_output = self.activation.forward(self.last_z)
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """反向传播"""
        # 计算激活函数的梯度
        activation_grad = self.activation.backward(self.last_z)
        
        # 计算误差项
        delta = grad_output * activation_grad
        
        # 计算权重和偏置的梯度
        self.grad_weights = np.dot(self.last_input.T, delta)
        self.grad_bias = np.sum(delta, axis=0, keepdims=True)
        
        # 计算传递给前一层的梯度
        grad_input = np.dot(delta, self.weights.T)
        
        return grad_input
    
    def update_weights(self, learning_rate: float):
        """更新权重和偏置"""
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias


class NeuralNetwork:
    """多层感知机神经网络"""
    
    def __init__(self, layers: List[int], activations: List[str] = None):
        self.layers = layers
        self.network = []
        
        # 默认激活函数
        if activations is None:
            activations = ['relu'] * (len(layers) - 2) + ['sigmoid']
        
        # 创建激活函数映射
        activation_map = {
            'sigmoid': Sigmoid(),
            'relu': ReLU(),
            'tanh': Tanh(),
            'leaky_relu': LeakyReLU()
        }
        
        # 构建网络
        for i in range(len(layers) - 1):
            activation = activation_map[activations[i]]
            layer = Layer(layers[i], layers[i + 1], activation)
            self.network.append(layer)
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        for layer in self.network:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output: np.ndarray):
        """反向传播"""
        for layer in reversed(self.network):
            grad_output = layer.backward(grad_output)
    
    def update_weights(self, learning_rate: float):
        """更新所有层的权重"""
        for layer in self.network:
            layer.update_weights(learning_rate)
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算损失函数（交叉熵）"""
        # 避免数值不稳定
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        if targets.ndim == 1:
            # 二分类
            loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        else:
            # 多分类
            loss = -np.mean(np.sum(targets * np.log(predictions), axis=1))
        
        return loss
    
    def compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算准确率"""
        if targets.ndim == 1:
            # 二分类
            pred_classes = (predictions > 0.5).astype(int).flatten()
            return np.mean(pred_classes == targets)
        else:
            # 多分类
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(targets, axis=1)
            return np.mean(pred_classes == true_classes)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, learning_rate: float = 0.01,
              batch_size: int = 32, verbose: bool = True):
        """训练网络"""
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # 小批量训练
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:end_idx]
                y_batch = y_train_shuffled[i:end_idx]
                
                # 前向传播
                predictions = self.forward(X_batch)
                
                # 计算损失梯度
                if y_batch.ndim == 1:
                    # 二分类
                    grad_output = (predictions.flatten() - y_batch.flatten()).reshape(-1, 1)
                else:
                    # 多分类
                    grad_output = predictions - y_batch
                
                grad_output /= X_batch.shape[0]  # 平均梯度
                
                # 反向传播
                self.backward(grad_output)
                
                # 更新权重
                self.update_weights(learning_rate)
            
            # 记录训练历史
            train_pred = self.forward(X_train)
            train_loss = self.compute_loss(train_pred, y_train)
            train_acc = self.compute_accuracy(train_pred, y_train)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(val_pred, y_val)
                val_acc = self.compute_accuracy(val_pred, y_val)
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.forward(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        predictions = self.predict(X)
        if predictions.shape[1] == 1:
            # 二分类
            return (predictions > 0.5).astype(int).flatten()
        else:
            # 多分类
            return np.argmax(predictions, axis=1)


def create_dataset(dataset_type: str = 'classification', n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """创建数据集"""
    if dataset_type == 'classification':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                 n_informative=2, random_state=42, n_clusters_per_class=1)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=0.1, random_state=42)
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return X, y


def plot_training_history(nn: NeuralNetwork):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失
    ax1.plot(nn.train_losses, label='训练损失', color='blue')
    if nn.val_losses:
        ax1.plot(nn.val_losses, label='验证损失', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('损失函数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率
    ax2.plot(nn.train_accuracies, label='训练准确率', color='blue')
    if nn.val_accuracies:
        ax2.plot(nn.val_accuracies, label='验证准确率', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('准确率')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(nn: NeuralNetwork, X: np.ndarray, y: np.ndarray, title: str = "决策边界"):
    """绘制决策边界"""
    plt.figure(figsize=(10, 8))
    
    # 创建网格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.predict(grid_points)
    if Z.shape[1] == 1:
        Z = Z.reshape(xx.shape)
    else:
        Z = np.argmax(Z, axis=1).reshape(xx.shape)
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title(title)
    plt.show()


def compare_activation_functions():
    """比较不同激活函数的性能"""
    print("=== 激活函数性能比较 ===")
    
    # 创建数据
    X, y = create_dataset('moons', 1000)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 不同激活函数
    activations = ['sigmoid', 'relu', 'tanh', 'leaky_relu']
    results = {}
    
    for activation in activations:
        print(f"\n训练使用 {activation} 激活函数的网络...")
        
        # 创建网络
        nn = NeuralNetwork([2, 10, 8, 1], [activation, activation, 'sigmoid'])
        
        # 训练
        start_time = time.time()
        nn.train(X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.01, verbose=False)
        training_time = time.time() - start_time
        
        # 评估
        test_predictions = nn.predict(X_test)
        test_accuracy = nn.compute_accuracy(test_predictions, y_test)
        
        results[activation] = {
            'accuracy': test_accuracy,
            'training_time': training_time,
            'final_loss': nn.train_losses[-1],
            'model': nn
        }
        
        print(f"  测试准确率: {test_accuracy:.4f}")
        print(f"  训练时间: {training_time:.2f}s")
    
    # 绘制比较结果
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 准确率比较
    acts = list(results.keys())
    accs = [results[act]['accuracy'] for act in acts]
    bars1 = ax1.bar(acts, accs, color=['skyblue', 'lightgreen', 'salmon', 'orange'])
    ax1.set_title('测试准确率比较')
    ax1.set_ylabel('准确率')
    ax1.set_ylim(0, 1)
    
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 训练时间比较
    times = [results[act]['training_time'] for act in acts]
    bars2 = ax2.bar(acts, times, color=['skyblue', 'lightgreen', 'salmon', 'orange'])
    ax2.set_title('训练时间比较')
    ax2.set_ylabel('时间 (秒)')
    
    for bar, time_val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    # 损失函数比较
    losses = [results[act]['final_loss'] for act in acts]
    bars3 = ax3.bar(acts, losses, color=['skyblue', 'lightgreen', 'salmon', 'orange'])
    ax3.set_title('最终训练损失比较')
    ax3.set_ylabel('损失')
    
    for bar, loss in zip(bars3, losses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.3f}', ha='center', va='bottom')
    
    # 训练历史比较
    for act in acts:
        ax4.plot(results[act]['model'].train_losses, label=f'{act}', linewidth=2)
    ax4.set_title('训练损失历史')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def demo_neural_network():
    """神经网络演示"""
    print("=== 神经网络演示 ===")
    
    # 创建数据
    X, y = create_dataset('circles', 1000)
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"特征数: {X_train.shape[1]}")
    
    # 创建网络
    nn = NeuralNetwork([2, 10, 8, 1], ['relu', 'relu', 'sigmoid'])
    
    print("\n网络架构:")
    print(f"输入层: {nn.layers[0]} 个神经元")
    for i, layer in enumerate(nn.network):
        print(f"隐藏层 {i+1}: {layer.output_size} 个神经元, 激活函数: {layer.activation.__class__.__name__}")
    
    # 训练
    print("\n开始训练...")
    nn.train(X_train, y_train, X_test, y_test, epochs=200, learning_rate=0.01, batch_size=32)
    
    # 评估
    test_predictions = nn.predict(X_test)
    test_accuracy = nn.compute_accuracy(test_predictions, y_test)
    
    print(f"\n最终测试准确率: {test_accuracy:.4f}")
    
    # 可视化
    plot_training_history(nn)
    plot_decision_boundary(nn, X, y, "神经网络决策边界")
    
    return nn


def demo_multiclass_classification():
    """多分类演示"""
    print("\n=== 多分类问题演示 ===")
    
    # 创建多分类数据
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=3, 
                              n_clusters_per_class=1, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 转换为one-hot编码
    n_classes = len(np.unique(y))
    y_onehot = np.eye(n_classes)[y]
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)
    
    # 创建网络
    nn = NeuralNetwork([2, 15, 10, 3], ['relu', 'relu', 'sigmoid'])
    
    # 训练
    print("开始训练多分类网络...")
    nn.train(X_train, y_train, X_test, y_test, epochs=150, learning_rate=0.01)
    
    # 评估
    test_predictions = nn.predict(X_test)
    test_accuracy = nn.compute_accuracy(test_predictions, y_test)
    
    print(f"多分类测试准确率: {test_accuracy:.4f}")
    
    # 可视化
    plot_training_history(nn)
    plot_decision_boundary(nn, X, y, "多分类神经网络决策边界")
    
    return nn


if __name__ == "__main__":
    # 基本演示
    nn = demo_neural_network()
    
    # 多分类演示
    nn_multiclass = demo_multiclass_classification()
    
    # 激活函数比较
    activation_results = compare_activation_functions()
    
    print("\n=== 神经网络演示完成 ===")
    print("已生成训练历史和决策边界可视化图表") 