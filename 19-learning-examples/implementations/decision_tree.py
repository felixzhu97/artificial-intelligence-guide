"""
决策树算法案例实现

本模块演示了《人工智能：现代方法》第18章中的决策树学习算法：
1. ID3算法
2. C4.5算法
3. 决策树剪枝
4. 随机森林
5. 可视化和分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter
from abc import ABC, abstractmethod
import math
import random
from dataclasses import dataclass
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class DecisionNode:
    """决策树节点"""
    feature: Optional[str] = None
    threshold: Optional[float] = None
    left: Optional['DecisionNode'] = None
    right: Optional['DecisionNode'] = None
    value: Optional[Any] = None
    samples: int = 0
    impurity: float = 0.0
    
    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return self.value is not None


class DecisionTreeBase(ABC):
    """决策树基类"""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, criterion: str = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.feature_names = None
        self.n_features = None
        self.n_classes = None
        self.classes = None
    
    @abstractmethod
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """计算不纯度"""
        pass
    
    @abstractmethod
    def _calculate_leaf_value(self, y: np.ndarray) -> Any:
        """计算叶子节点的值"""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """训练决策树"""
        self.n_features = X.shape[1]
        self.feature_names = feature_names or [f'特征{i}' for i in range(self.n_features)]
        
        if hasattr(self, '_setup_classes'):
            self._setup_classes(y)
        
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionNode:
        """递归构建决策树"""
        n_samples = len(y)
        
        # 创建叶子节点的条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return DecisionNode(
                value=self._calculate_leaf_value(y),
                samples=n_samples,
                impurity=self._calculate_impurity(y)
            )
        
        # 寻找最佳分割
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return DecisionNode(
                value=self._calculate_leaf_value(y),
                samples=n_samples,
                impurity=self._calculate_impurity(y)
            )
        
        # 分割数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # 检查分割是否有效
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return DecisionNode(
                value=self._calculate_leaf_value(y),
                samples=n_samples,
                impurity=self._calculate_impurity(y)
            )
        
        # 递归构建左右子树
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(
            feature=self.feature_names[best_feature],
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            samples=n_samples,
            impurity=self._calculate_impurity(y)
        )
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """寻找最佳分割点"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(self.n_features):
            feature_values = np.unique(X[:, feature])
            
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                gain = self._calculate_information_gain(X, y, feature, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _calculate_information_gain(self, X: np.ndarray, y: np.ndarray, 
                                   feature: int, threshold: float) -> float:
        """计算信息增益"""
        parent_impurity = self._calculate_impurity(y)
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        n_total = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        left_impurity = self._calculate_impurity(y[left_mask])
        right_impurity = self._calculate_impurity(y[right_mask])
        
        weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
        
        return parent_impurity - weighted_impurity
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        predictions = []
        for sample in X:
            predictions.append(self._predict_sample(sample, self.root))
        return np.array(predictions)
    
    def _predict_sample(self, sample: np.ndarray, node: DecisionNode) -> Any:
        """预测单个样本"""
        if node.is_leaf():
            return node.value
        
        feature_index = self.feature_names.index(node.feature)
        
        if sample[feature_index] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """计算特征重要性"""
        importance = {name: 0.0 for name in self.feature_names}
        self._calculate_feature_importance(self.root, importance)
        
        # 标准化
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def _calculate_feature_importance(self, node: DecisionNode, importance: Dict[str, float]):
        """递归计算特征重要性"""
        if node.is_leaf():
            return
        
        # 计算该节点的重要性贡献
        left_samples = node.left.samples
        right_samples = node.right.samples
        total_samples = left_samples + right_samples
        
        improvement = (node.impurity - 
                      (left_samples / total_samples) * node.left.impurity - 
                      (right_samples / total_samples) * node.right.impurity)
        
        importance[node.feature] += improvement * total_samples
        
        # 递归处理子节点
        self._calculate_feature_importance(node.left, importance)
        self._calculate_feature_importance(node.right, importance)


class DecisionTreeClassifier(DecisionTreeBase):
    """决策树分类器"""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, criterion: str = 'gini'):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, criterion)
    
    def _setup_classes(self, y: np.ndarray):
        """设置类别信息"""
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """计算不纯度"""
        if len(y) == 0:
            return 0
        
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"不支持的准则: {self.criterion}")
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """计算基尼不纯度"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        impurity = 1.0
        
        for count in counts.values():
            prob = count / len(y)
            impurity -= prob ** 2
        
        return impurity
    
    def _entropy(self, y: np.ndarray) -> float:
        """计算熵"""
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        entropy = 0.0
        
        for count in counts.values():
            prob = count / len(y)
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _calculate_leaf_value(self, y: np.ndarray) -> Any:
        """计算叶子节点的值（多数投票）"""
        return Counter(y).most_common(1)[0][0]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        probabilities = []
        for sample in X:
            proba = self._predict_proba_sample(sample, self.root)
            probabilities.append(proba)
        return np.array(probabilities)
    
    def _predict_proba_sample(self, sample: np.ndarray, node: DecisionNode) -> np.ndarray:
        """预测单个样本的概率"""
        if node.is_leaf():
            # 创建概率向量
            proba = np.zeros(self.n_classes)
            class_index = np.where(self.classes == node.value)[0][0]
            proba[class_index] = 1.0
            return proba
        
        feature_index = self.feature_names.index(node.feature)
        
        if sample[feature_index] <= node.threshold:
            return self._predict_proba_sample(sample, node.left)
        else:
            return self._predict_proba_sample(sample, node.right)


class DecisionTreeRegressor(DecisionTreeBase):
    """决策树回归器"""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, criterion: str = 'mse'):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, criterion)
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """计算不纯度（均方误差）"""
        if len(y) == 0:
            return 0
        
        if self.criterion == 'mse':
            return np.var(y)
        elif self.criterion == 'mae':
            return np.mean(np.abs(y - np.mean(y)))
        else:
            raise ValueError(f"不支持的准则: {self.criterion}")
    
    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """计算叶子节点的值（平均值）"""
        return np.mean(y)


class RandomForestClassifier:
    """随机森林分类器"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', bootstrap: bool = True,
                 random_state: int = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """训练随机森林"""
        self.feature_names = feature_names or [f'特征{i}' for i in range(X.shape[1])]
        self.n_features = X.shape[1]
        
        # 计算每棵树使用的特征数
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(self.n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(self.n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = self.n_features
        
        self.trees = []
        self.feature_indices = []
        
        for i in range(self.n_estimators):
            # 随机选择特征
            feature_indices = np.random.choice(self.n_features, max_features, replace=False)
            self.feature_indices.append(feature_indices)
            
            # 创建子特征名称
            sub_feature_names = [self.feature_names[j] for j in feature_indices]
            
            # Bootstrap采样
            if self.bootstrap:
                indices = np.random.choice(len(X), len(X), replace=True)
                X_bootstrap = X[indices][:, feature_indices]
                y_bootstrap = y[indices]
            else:
                X_bootstrap = X[:, feature_indices]
                y_bootstrap = y
            
            # 训练决策树
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_bootstrap, y_bootstrap, sub_feature_names)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        predictions = []
        
        for tree, feature_indices in zip(self.trees, self.feature_indices):
            X_sub = X[:, feature_indices]
            tree_predictions = tree.predict(X_sub)
            predictions.append(tree_predictions)
        
        predictions = np.array(predictions).T
        
        # 多数投票
        final_predictions = []
        for pred_row in predictions:
            final_predictions.append(Counter(pred_row).most_common(1)[0][0])
        
        return np.array(final_predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """计算特征重要性"""
        importance = {name: 0.0 for name in self.feature_names}
        
        for tree, feature_indices in zip(self.trees, self.feature_indices):
            tree_importance = tree.get_feature_importance()
            
            for i, feature_index in enumerate(feature_indices):
                feature_name = self.feature_names[feature_index]
                sub_feature_name = tree.feature_names[i]
                importance[feature_name] += tree_importance.get(sub_feature_name, 0)
        
        # 标准化
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance


def visualize_tree(tree: DecisionTreeClassifier, max_depth: int = 3):
    """可视化决策树"""
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    def plot_node(node: DecisionNode, x: float, y: float, width: float, depth: int):
        if depth > max_depth:
            return
        
        # 绘制节点
        if node.is_leaf():
            # 叶子节点
            rect = patches.Rectangle((x - width/2, y - 0.1), width, 0.2, 
                                   facecolor='lightgreen', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, f'类别: {node.value}\n样本: {node.samples}', 
                   ha='center', va='center', fontsize=8)
        else:
            # 内部节点
            rect = patches.Rectangle((x - width/2, y - 0.1), width, 0.2, 
                                   facecolor='lightblue', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x, y, f'{node.feature} <= {node.threshold:.2f}\n样本: {node.samples}', 
                   ha='center', va='center', fontsize=8)
            
            # 绘制子节点
            child_width = width * 0.7
            child_y = y - 0.5
            
            # 左子树
            left_x = x - width/4
            ax.plot([x, left_x], [y - 0.1, child_y + 0.1], 'k-', alpha=0.5)
            plot_node(node.left, left_x, child_y, child_width, depth + 1)
            
            # 右子树
            right_x = x + width/4
            ax.plot([x, right_x], [y - 0.1, child_y + 0.1], 'k-', alpha=0.5)
            plot_node(node.right, right_x, child_y, child_width, depth + 1)
    
    plot_node(tree.root, 0, 0, 2, 0)
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('决策树可视化', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance: Dict[str, float], title: str = "特征重要性"):
    """绘制特征重要性"""
    features = list(importance.keys())
    values = list(importance.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, values, color='skyblue', alpha=0.8)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('重要性', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def compare_algorithms():
    """比较不同决策树算法"""
    print("=== 决策树算法比较 ===")
    
    # 生成示例数据
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                              n_informative=5, random_state=42)
    
    feature_names = [f'特征{i+1}' for i in range(X.shape[1])]
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 测试不同算法
    algorithms = {
        'ID3 (熵)': DecisionTreeClassifier(criterion='entropy', max_depth=10),
        'CART (基尼)': DecisionTreeClassifier(criterion='gini', max_depth=10),
        '随机森林': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    results = {}
    
    for name, model in algorithms.items():
        print(f"\n训练 {name}...")
        
        # 训练模型
        model.fit(X_train, y_train, feature_names)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算准确率
        accuracy = np.mean(y_pred == y_test)
        
        # 获取特征重要性
        importance = model.get_feature_importance()
        
        results[name] = {
            'accuracy': accuracy,
            'importance': importance,
            'model': model
        }
        
        print(f"  准确率: {accuracy:.4f}")
        print(f"  前3个重要特征: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # 绘制比较结果
    plt.figure(figsize=(12, 8))
    
    # 准确率比较
    plt.subplot(2, 2, 1)
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    bars = plt.bar(names, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('算法准确率比较')
    plt.ylabel('准确率')
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 特征重要性比较
    for i, (name, result) in enumerate(results.items(), 2):
        plt.subplot(2, 2, i)
        importance = result['importance']
        features = list(importance.keys())
        values = list(importance.values())
        
        # 只显示前5个重要特征
        top_indices = np.argsort(values)[-5:]
        top_features = [features[i] for i in top_indices]
        top_values = [values[i] for i in top_indices]
        
        plt.barh(top_features, top_values, color='orange', alpha=0.7)
        plt.title(f'{name} - 特征重要性')
        plt.xlabel('重要性')
    
    plt.tight_layout()
    plt.show()
    
    return results


def demo_decision_tree():
    """决策树演示"""
    print("=== 决策树演示 ===")
    
    # 创建简单的分类数据
    data = {
        '天气': ['晴', '晴', '阴', '雨', '雨', '雨', '阴', '晴', '晴', '雨', '晴', '阴', '阴', '雨'],
        '温度': ['热', '热', '热', '适中', '凉', '凉', '凉', '适中', '凉', '适中', '适中', '适中', '热', '适中'],
        '湿度': ['高', '高', '高', '高', '正常', '正常', '正常', '高', '正常', '正常', '正常', '高', '正常', '高'],
        '风': ['弱', '强', '弱', '弱', '弱', '强', '强', '弱', '弱', '弱', '强', '强', '弱', '强'],
        '打网球': ['否', '否', '是', '是', '是', '否', '是', '否', '是', '是', '是', '是', '是', '否']
    }
    
    df = pd.DataFrame(data)
    print("数据集:")
    print(df)
    
    # 编码分类变量
    le_dict = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
    
    # 准备数据
    X = df.drop('打网球', axis=1).values
    y = df['打网球'].values
    feature_names = ['天气', '温度', '湿度', '风']
    
    # 训练决策树
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    tree.fit(X, y, feature_names)
    
    # 预测
    predictions = tree.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"\n训练准确率: {accuracy:.4f}")
    
    # 特征重要性
    importance = tree.get_feature_importance()
    print("\n特征重要性:")
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {imp:.4f}")
    
    # 可视化
    visualize_tree(tree, max_depth=3)
    plot_feature_importance(importance, "网球数据集特征重要性")
    
    return tree


if __name__ == "__main__":
    # 基本演示
    tree = demo_decision_tree()
    
    # 算法比较
    results = compare_algorithms()
    
    print("\n=== 演示完成 ===")
    print("已生成决策树可视化和性能比较图表") 