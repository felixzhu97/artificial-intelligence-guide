"""
通用工具函数模块

包含项目中使用的通用工具函数。
"""

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional, Union
import functools
import json
import pickle
import os


def set_random_seed(seed: int = 42):
    """设置随机数种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)


def timer(func):
    """装饰器：计算函数执行时间"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper


def save_object(obj: Any, filename: str):
    """保存对象到文件"""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_object(filename: str) -> Any:
    """从文件加载对象"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, filename: str):
    """保存字典到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filename: str) -> Dict:
    """从JSON文件加载字典"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_directory(path: str):
    """创建目录，如果不存在的话"""
    if not os.path.exists(path):
        os.makedirs(path)


def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """计算两点间的欧几里得距离"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def manhattan_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """计算两点间的曼哈顿距离"""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def normalize_data(data: np.ndarray) -> np.ndarray:
    """数据标准化"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def split_data(data: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """分割数据集为训练集和测试集"""
    n_samples = len(data)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return data[train_indices], data[test_indices], labels[train_indices], labels[test_indices]


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算分类准确率"""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """计算混淆矩阵"""
    classes = np.unique(y_true)
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes))
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return matrix


def print_progress(current: int, total: int, prefix: str = "进度"):
    """打印进度条"""
    percent = (current / total) * 100
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})', end='')
    if current == total:
        print()


class Logger:
    """简单的日志记录器"""
    
    def __init__(self, filename: str = None):
        self.filename = filename
        self.logs = []
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.logs.append(log_entry)
        print(log_entry)
        
        if self.filename:
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
    
    def info(self, message: str):
        """记录信息级别日志"""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """记录警告级别日志"""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """记录错误级别日志"""
        self.log(message, "ERROR")


def moving_average(data: List[float], window_size: int) -> List[float]:
    """计算移动平均"""
    if len(data) < window_size:
        return data
    
    result = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        result.append(sum(window) / window_size)
    
    return result


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU激活函数"""
    return np.maximum(0, x)


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """独热编码"""
    encoded = np.zeros((len(labels), num_classes))
    encoded[np.arange(len(labels)), labels] = 1
    return encoded


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """交叉熵损失函数"""
    # 避免log(0)的情况
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方误差"""
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R²决定系数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def generate_dataset(n_samples: int, n_features: int, n_classes: int = 2, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """生成示例数据集"""
    np.random.seed(42)
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成权重
    weights = np.random.randn(n_features, n_classes)
    
    # 计算线性组合
    linear_combination = X @ weights
    
    # 添加噪声
    linear_combination += np.random.normal(0, noise, linear_combination.shape)
    
    # 生成标签
    if n_classes == 2:
        y = (linear_combination[:, 0] > 0).astype(int)
    else:
        y = np.argmax(linear_combination, axis=1)
    
    return X, y 