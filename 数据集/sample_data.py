"""
示例数据生成脚本

为各种AI算法提供测试数据集。
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import random
import json
import os


def generate_iris_like_data(n_samples: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成类似鸢尾花数据集的数据
    
    Args:
        n_samples: 样本数量
    
    Returns:
        (features, labels): 特征和标签
    """
    np.random.seed(42)
    
    # 生成3个类别的数据
    features = []
    labels = []
    
    # 类别1：较小的花瓣和花萼
    for _ in range(n_samples // 3):
        features.append([
            np.random.normal(5.0, 0.5),  # 花萼长度
            np.random.normal(3.5, 0.4),  # 花萼宽度
            np.random.normal(1.5, 0.3),  # 花瓣长度
            np.random.normal(0.3, 0.1)   # 花瓣宽度
        ])
        labels.append(0)
    
    # 类别2：中等大小
    for _ in range(n_samples // 3):
        features.append([
            np.random.normal(6.0, 0.6),
            np.random.normal(2.8, 0.4),
            np.random.normal(4.5, 0.5),
            np.random.normal(1.5, 0.3)
        ])
        labels.append(1)
    
    # 类别3：较大的花瓣和花萼
    for _ in range(n_samples // 3):
        features.append([
            np.random.normal(6.5, 0.7),
            np.random.normal(3.0, 0.5),
            np.random.normal(5.5, 0.6),
            np.random.normal(2.0, 0.4)
        ])
        labels.append(2)
    
    return np.array(features), np.array(labels)


def generate_recommendation_data(n_users: int = 100, n_items: int = 50, 
                                n_ratings: int = 1000) -> pd.DataFrame:
    """
    生成推荐系统数据
    
    Args:
        n_users: 用户数量
        n_items: 物品数量
        n_ratings: 评分数量
    
    Returns:
        评分数据DataFrame
    """
    np.random.seed(42)
    
    # 生成用户-物品评分数据
    user_ids = np.random.choice(range(1, n_users + 1), n_ratings)
    item_ids = np.random.choice(range(1, n_items + 1), n_ratings)
    
    # 生成评分（1-5分）
    ratings = []
    for user_id, item_id in zip(user_ids, item_ids):
        # 基于用户和物品特征生成评分
        user_bias = np.random.normal(0, 0.5)
        item_bias = np.random.normal(0, 0.5)
        base_rating = 3 + user_bias + item_bias
        rating = np.clip(base_rating + np.random.normal(0, 0.3), 1, 5)
        ratings.append(rating)
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    # 去除重复评分
    df = df.drop_duplicates(subset=['user_id', 'item_id'])
    
    return df


def generate_item_features(n_items: int = 50) -> pd.DataFrame:
    """
    生成物品特征数据
    
    Args:
        n_items: 物品数量
    
    Returns:
        物品特征DataFrame
    """
    np.random.seed(42)
    
    categories = ['动作', '喜剧', '剧情', '恐怖', '爱情', '科幻', '纪录片']
    genres = ['冒险', '动画', '犯罪', '家庭', '奇幻', '历史', '音乐']
    
    items = []
    for i in range(1, n_items + 1):
        items.append({
            'item_id': i,
            'category': np.random.choice(categories),
            'genre': np.random.choice(genres),
            'year': np.random.randint(1990, 2024),
            'popularity': np.random.random(),
            'rating': np.random.uniform(1, 5)
        })
    
    return pd.DataFrame(items)


def generate_classification_data(n_samples: int = 1000, n_features: int = 2, 
                                n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成分类数据
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量
    
    Returns:
        (features, labels): 特征和标签
    """
    np.random.seed(42)
    
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=0,
        n_informative=min(n_features, n_classes),
        random_state=42
    )
    
    return X, y


def generate_regression_data(n_samples: int = 1000, n_features: int = 1, 
                            noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成回归数据
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        noise: 噪声水平
    
    Returns:
        (features, targets): 特征和目标值
    """
    np.random.seed(42)
    
    from sklearn.datasets import make_regression
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=42
    )
    
    return X, y


def generate_clustering_data(n_samples: int = 300, n_centers: int = 3) -> np.ndarray:
    """
    生成聚类数据
    
    Args:
        n_samples: 样本数量
        n_centers: 聚类中心数量
    
    Returns:
        特征数据
    """
    np.random.seed(42)
    
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=2,
        random_state=42
    )
    
    return X


def generate_grid_world_data(width: int = 5, height: int = 5) -> Dict[str, Any]:
    """
    生成网格世界数据
    
    Args:
        width: 网格宽度
        height: 网格高度
    
    Returns:
        网格世界配置
    """
    np.random.seed(42)
    
    # 生成障碍物位置
    obstacles = []
    n_obstacles = min(width * height // 4, 5)
    
    for _ in range(n_obstacles):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        if (x, y) not in [(0, 0), (width-1, height-1)]:  # 避免起点和终点
            obstacles.append((x, y))
    
    return {
        'width': width,
        'height': height,
        'start': (0, 0),
        'goal': (width-1, height-1),
        'obstacles': obstacles
    }


def generate_time_series_data(n_samples: int = 1000, trend: bool = True, 
                             seasonal: bool = True, noise: float = 0.1) -> np.ndarray:
    """
    生成时间序列数据
    
    Args:
        n_samples: 样本数量
        trend: 是否包含趋势
        seasonal: 是否包含季节性
        noise: 噪声水平
    
    Returns:
        时间序列数据
    """
    np.random.seed(42)
    
    t = np.linspace(0, 4 * np.pi, n_samples)
    
    # 基础信号
    signal = np.sin(t)
    
    # 添加趋势
    if trend:
        signal += 0.1 * t
    
    # 添加季节性
    if seasonal:
        signal += 0.5 * np.sin(0.1 * t)
    
    # 添加噪声
    signal += np.random.normal(0, noise, n_samples)
    
    return signal


def generate_text_data(n_samples: int = 100) -> List[Dict[str, Any]]:
    """
    生成文本分类数据
    
    Args:
        n_samples: 样本数量
    
    Returns:
        文本数据列表
    """
    np.random.seed(42)
    
    # 正面评论词汇
    positive_words = ['好', '棒', '优秀', '喜欢', '推荐', '满意', '完美', '赞']
    # 负面评论词汇
    negative_words = ['差', '糟糕', '失望', '不满', '后悔', '垃圾', '讨厌', '退货']
    # 中性词汇
    neutral_words = ['产品', '服务', '价格', '质量', '包装', '物流', '客服', '体验']
    
    texts = []
    
    for _ in range(n_samples):
        # 随机选择情感
        sentiment = np.random.choice(['positive', 'negative', 'neutral'])
        
        # 生成文本
        if sentiment == 'positive':
            words = np.random.choice(positive_words + neutral_words, 
                                   size=np.random.randint(3, 8), replace=True)
            label = 1
        elif sentiment == 'negative':
            words = np.random.choice(negative_words + neutral_words, 
                                   size=np.random.randint(3, 8), replace=True)
            label = 0
        else:
            words = np.random.choice(neutral_words, 
                                   size=np.random.randint(3, 8), replace=True)
            label = 0.5
        
        text = ' '.join(words)
        texts.append({
            'text': text,
            'sentiment': sentiment,
            'label': label
        })
    
    return texts


def generate_graph_data(n_nodes: int = 10, edge_probability: float = 0.3) -> Dict[str, Any]:
    """
    生成图数据
    
    Args:
        n_nodes: 节点数量
        edge_probability: 边的概率
    
    Returns:
        图数据
    """
    np.random.seed(42)
    
    # 生成节点
    nodes = [f'N{i}' for i in range(n_nodes)]
    
    # 生成边
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < edge_probability:
                weight = np.random.uniform(1, 10)
                edges.append((nodes[i], nodes[j], weight))
    
    # 构建邻接表
    adjacency_list = {node: [] for node in nodes}
    for node1, node2, weight in edges:
        adjacency_list[node1].append((node2, weight))
        adjacency_list[node2].append((node1, weight))
    
    return {
        'nodes': nodes,
        'edges': edges,
        'adjacency_list': adjacency_list
    }


def save_all_sample_data(data_dir: str = "数据集"):
    """
    保存所有示例数据到文件
    
    Args:
        data_dir: 数据目录
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print(f"正在生成示例数据并保存到 {data_dir}/")
    
    # 分类数据
    print("  生成分类数据...")
    X_cls, y_cls = generate_iris_like_data()
    np.savez(os.path.join(data_dir, 'iris_like_data.npz'), 
             features=X_cls, labels=y_cls)
    
    # 推荐数据
    print("  生成推荐数据...")
    ratings_df = generate_recommendation_data()
    ratings_df.to_csv(os.path.join(data_dir, 'ratings.csv'), index=False)
    
    items_df = generate_item_features()
    items_df.to_csv(os.path.join(data_dir, 'items.csv'), index=False)
    
    # 回归数据
    print("  生成回归数据...")
    X_reg, y_reg = generate_regression_data()
    np.savez(os.path.join(data_dir, 'regression_data.npz'), 
             features=X_reg, targets=y_reg)
    
    # 聚类数据
    print("  生成聚类数据...")
    X_cluster = generate_clustering_data()
    np.save(os.path.join(data_dir, 'clustering_data.npy'), X_cluster)
    
    # 时间序列数据
    print("  生成时间序列数据...")
    ts_data = generate_time_series_data()
    np.save(os.path.join(data_dir, 'time_series_data.npy'), ts_data)
    
    # 文本数据
    print("  生成文本数据...")
    text_data = generate_text_data()
    with open(os.path.join(data_dir, 'text_data.json'), 'w', encoding='utf-8') as f:
        json.dump(text_data, f, ensure_ascii=False, indent=2)
    
    # 图数据
    print("  生成图数据...")
    graph_data = generate_graph_data()
    with open(os.path.join(data_dir, 'graph_data.json'), 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    # 网格世界数据
    print("  生成网格世界数据...")
    grid_world = generate_grid_world_data()
    with open(os.path.join(data_dir, 'grid_world.json'), 'w', encoding='utf-8') as f:
        json.dump(grid_world, f, ensure_ascii=False, indent=2)
    
    print("✅ 所有示例数据生成完成！")


def load_sample_data(data_type: str, data_dir: str = "数据集"):
    """
    加载示例数据
    
    Args:
        data_type: 数据类型
        data_dir: 数据目录
    
    Returns:
        数据
    """
    if data_type == 'iris':
        data = np.load(os.path.join(data_dir, 'iris_like_data.npz'))
        return data['features'], data['labels']
    
    elif data_type == 'ratings':
        return pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    
    elif data_type == 'items':
        return pd.read_csv(os.path.join(data_dir, 'items.csv'))
    
    elif data_type == 'regression':
        data = np.load(os.path.join(data_dir, 'regression_data.npz'))
        return data['features'], data['targets']
    
    elif data_type == 'clustering':
        return np.load(os.path.join(data_dir, 'clustering_data.npy'))
    
    elif data_type == 'time_series':
        return np.load(os.path.join(data_dir, 'time_series_data.npy'))
    
    elif data_type == 'text':
        with open(os.path.join(data_dir, 'text_data.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif data_type == 'graph':
        with open(os.path.join(data_dir, 'graph_data.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif data_type == 'grid_world':
        with open(os.path.join(data_dir, 'grid_world.json'), 'r', encoding='utf-8') as f:
            return json.load(f)
    
    else:
        raise ValueError(f"未知的数据类型: {data_type}")


if __name__ == "__main__":
    # 生成并保存所有示例数据
    save_all_sample_data()
    
    # 测试加载数据
    print("\n测试数据加载:")
    
    # 测试分类数据
    X, y = load_sample_data('iris')
    print(f"鸢尾花数据: {X.shape}, 标签: {np.unique(y)}")
    
    # 测试推荐数据
    ratings = load_sample_data('ratings')
    print(f"评分数据: {ratings.shape}")
    
    # 测试图数据
    graph = load_sample_data('graph')
    print(f"图数据: {len(graph['nodes'])} 个节点, {len(graph['edges'])} 条边")
    
    print("\n✅ 所有数据加载测试完成！") 