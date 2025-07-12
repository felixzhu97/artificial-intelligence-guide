"""
智能推荐系统综合项目案例

本项目展示了如何结合《人工智能：现代方法》中的多种技术来构建一个完整的推荐系统：
1. 协同过滤
2. 基于内容的推荐
3. 深度学习推荐
4. 混合推荐策略
5. 实时推荐
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class RecommendationEngine(ABC):
    """推荐引擎基类"""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """训练推荐模型"""
        pass
    
    @abstractmethod
    def recommend(self, user_id: Any, n_recommendations: int = 10) -> List[Tuple[Any, float]]:
        """为用户推荐物品"""
        pass


class CollaborativeFiltering(RecommendationEngine):
    """协同过滤推荐引擎"""
    
    def __init__(self, method: str = 'user_based', n_neighbors: int = 20):
        self.method = method
        self.n_neighbors = n_neighbors
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.users = None
        self.items = None
    
    def fit(self, data: pd.DataFrame):
        """训练协同过滤模型"""
        # 创建用户-物品评分矩阵
        self.user_item_matrix = data.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        self.users = self.user_item_matrix.index.tolist()
        self.items = self.user_item_matrix.columns.tolist()
        
        # 计算相似度矩阵
        if self.method == 'user_based':
            self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        else:  # item_based
            self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        
        # 转换为DataFrame方便索引
        if self.method == 'user_based':
            self.similarity_matrix = pd.DataFrame(
                self.similarity_matrix, 
                index=self.users, 
                columns=self.users
            )
        else:
            self.similarity_matrix = pd.DataFrame(
                self.similarity_matrix, 
                index=self.items, 
                columns=self.items
            )
    
    def recommend(self, user_id: Any, n_recommendations: int = 10) -> List[Tuple[Any, float]]:
        """推荐物品"""
        if user_id not in self.users:
            return []
        
        if self.method == 'user_based':
            return self._user_based_recommend(user_id, n_recommendations)
        else:
            return self._item_based_recommend(user_id, n_recommendations)
    
    def _user_based_recommend(self, user_id: Any, n_recommendations: int) -> List[Tuple[Any, float]]:
        """基于用户的协同过滤"""
        user_similarities = self.similarity_matrix.loc[user_id].sort_values(ascending=False)
        similar_users = user_similarities.head(self.n_neighbors + 1).index[1:]  # 排除自己
        
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        recommendations = []
        
        for item in unrated_items:
            # 计算加权平均评分
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user in similar_users:
                if self.user_item_matrix.loc[similar_user, item] > 0:
                    weight = user_similarities[similar_user]
                    rating = self.user_item_matrix.loc[similar_user, item]
                    weighted_sum += weight * rating
                    similarity_sum += weight
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations.append((item, predicted_rating))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def _item_based_recommend(self, user_id: Any, n_recommendations: int) -> List[Tuple[Any, float]]:
        """基于物品的协同过滤"""
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        unrated_items = user_ratings[user_ratings == 0].index
        
        recommendations = []
        
        for item in unrated_items:
            # 计算与已评分物品的相似度加权评分
            weighted_sum = 0
            similarity_sum = 0
            
            for rated_item in rated_items:
                similarity = self.similarity_matrix.loc[item, rated_item]
                rating = user_ratings[rated_item]
                weighted_sum += similarity * rating
                similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations.append((item, predicted_rating))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]


class ContentBasedFiltering(RecommendationEngine):
    """基于内容的推荐引擎"""
    
    def __init__(self, content_features: List[str]):
        self.content_features = content_features
        self.item_profiles = None
        self.user_profiles = None
        self.tfidf_vectorizer = None
        self.feature_matrix = None
    
    def fit(self, data: pd.DataFrame, item_data: pd.DataFrame):
        """训练基于内容的模型"""
        # 构建物品特征矩阵
        self._build_item_profiles(item_data)
        
        # 构建用户特征矩阵
        self._build_user_profiles(data)
    
    def _build_item_profiles(self, item_data: pd.DataFrame):
        """构建物品特征矩阵"""
        # 处理文本特征
        text_features = []
        for _, row in item_data.iterrows():
            text = ' '.join([str(row[feature]) for feature in self.content_features])
            text_features.append(text)
        
        # TF-IDF向量化
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.feature_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        
        # 创建物品特征字典
        self.item_profiles = {}
        for i, item_id in enumerate(item_data['item_id']):
            self.item_profiles[item_id] = self.feature_matrix[i].toarray().flatten()
    
    def _build_user_profiles(self, data: pd.DataFrame):
        """构建用户特征矩阵"""
        self.user_profiles = {}
        
        for user_id in data['user_id'].unique():
            user_data = data[data['user_id'] == user_id]
            
            # 计算用户对每个特征的加权平均
            user_vector = np.zeros(self.feature_matrix.shape[1])
            total_weight = 0
            
            for _, row in user_data.iterrows():
                item_id = row['item_id']
                rating = row['rating']
                
                if item_id in self.item_profiles:
                    user_vector += rating * self.item_profiles[item_id]
                    total_weight += rating
            
            if total_weight > 0:
                user_vector /= total_weight
                self.user_profiles[user_id] = user_vector
    
    def recommend(self, user_id: Any, n_recommendations: int = 10) -> List[Tuple[Any, float]]:
        """推荐物品"""
        if user_id not in self.user_profiles:
            return []
        
        user_vector = self.user_profiles[user_id]
        recommendations = []
        
        for item_id, item_vector in self.item_profiles.items():
            # 计算余弦相似度
            similarity = cosine_similarity(
                user_vector.reshape(1, -1), 
                item_vector.reshape(1, -1)
            )[0][0]
            
            recommendations.append((item_id, similarity))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]


class MatrixFactorization(RecommendationEngine):
    """矩阵分解推荐引擎"""
    
    def __init__(self, n_factors: int = 50, n_epochs: int = 100, 
                 learning_rate: float = 0.01, reg_lambda: float = 0.01):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.user_mapping = None
        self.item_mapping = None
    
    def fit(self, data: pd.DataFrame):
        """训练矩阵分解模型"""
        # 创建用户和物品的映射
        unique_users = data['user_id'].unique()
        unique_items = data['item_id'].unique()
        
        self.user_mapping = {user: i for i, user in enumerate(unique_users)}
        self.item_mapping = {item: i for i, item in enumerate(unique_items)}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # 初始化参数
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = data['rating'].mean()
        
        # 训练模型
        for epoch in range(self.n_epochs):
            for _, row in data.iterrows():
                user_idx = self.user_mapping[row['user_id']]
                item_idx = self.item_mapping[row['item_id']]
                rating = row['rating']
                
                # 计算预测评分
                prediction = self._predict_rating(user_idx, item_idx)
                error = rating - prediction
                
                # 更新参数
                user_factors_old = self.user_factors[user_idx].copy()
                item_factors_old = self.item_factors[item_idx].copy()
                
                # 更新用户和物品因子
                self.user_factors[user_idx] += self.learning_rate * (
                    error * item_factors_old - self.reg_lambda * user_factors_old
                )
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_factors_old - self.reg_lambda * item_factors_old
                )
                
                # 更新偏置
                self.user_bias[user_idx] += self.learning_rate * (
                    error - self.reg_lambda * self.user_bias[user_idx]
                )
                self.item_bias[item_idx] += self.learning_rate * (
                    error - self.reg_lambda * self.item_bias[item_idx]
                )
    
    def _predict_rating(self, user_idx: int, item_idx: int) -> float:
        """预测评分"""
        prediction = (self.global_mean + 
                     self.user_bias[user_idx] + 
                     self.item_bias[item_idx] + 
                     np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        return prediction
    
    def recommend(self, user_id: Any, n_recommendations: int = 10) -> List[Tuple[Any, float]]:
        """推荐物品"""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        recommendations = []
        
        for item_id, item_idx in self.item_mapping.items():
            predicted_rating = self._predict_rating(user_idx, item_idx)
            recommendations.append((item_id, predicted_rating))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]


class HybridRecommendationSystem:
    """混合推荐系统"""
    
    def __init__(self, engines: Dict[str, RecommendationEngine], weights: Dict[str, float]):
        self.engines = engines
        self.weights = weights
        # 确保权重和为1
        total_weight = sum(weights.values())
        self.weights = {k: v/total_weight for k, v in weights.items()}
    
    def recommend(self, user_id: Any, n_recommendations: int = 10) -> List[Tuple[Any, float]]:
        """混合推荐"""
        all_recommendations = {}
        
        # 收集所有引擎的推荐
        for engine_name, engine in self.engines.items():
            recommendations = engine.recommend(user_id, n_recommendations * 2)
            weight = self.weights[engine_name]
            
            for item_id, score in recommendations:
                if item_id not in all_recommendations:
                    all_recommendations[item_id] = 0
                all_recommendations[item_id] += weight * score
        
        # 排序并返回top-N
        sorted_recommendations = sorted(
            all_recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_recommendations[:n_recommendations]


class RecommendationEvaluator:
    """推荐系统评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, engine: RecommendationEngine, test_data: pd.DataFrame, 
                 n_recommendations: int = 10) -> Dict[str, float]:
        """评估推荐系统"""
        predictions = []
        actuals = []
        
        # 预测评分
        for _, row in test_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            # 获取推荐分数
            recommendations = engine.recommend(user_id, n_recommendations)
            pred_rating = 0
            
            for rec_item, rec_score in recommendations:
                if rec_item == item_id:
                    pred_rating = rec_score
                    break
            
            predictions.append(pred_rating)
            actuals.append(actual_rating)
        
        # 计算评估指标
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 只计算有预测值的样本
        valid_mask = predictions != 0
        valid_predictions = predictions[valid_mask]
        valid_actuals = actuals[valid_mask]
        
        if len(valid_predictions) > 0:
            rmse = np.sqrt(mean_squared_error(valid_actuals, valid_predictions))
            mae = mean_absolute_error(valid_actuals, valid_predictions)
            
            # 计算覆盖率
            coverage = len(valid_predictions) / len(predictions)
        else:
            rmse = float('inf')
            mae = float('inf')
            coverage = 0
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'Coverage': coverage,
            'Valid_Predictions': len(valid_predictions)
        }


def generate_sample_data(n_users: int = 1000, n_items: int = 500, 
                        n_ratings: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成示例数据"""
    np.random.seed(42)
    
    # 生成用户-物品评分数据
    user_ids = np.random.choice(range(n_users), n_ratings)
    item_ids = np.random.choice(range(n_items), n_ratings)
    
    # 生成评分（1-5分）
    ratings = []
    for user_id, item_id in zip(user_ids, item_ids):
        # 基于用户和物品特征生成评分
        user_preference = np.random.random()
        item_quality = np.random.random()
        base_rating = 3 + 2 * (user_preference * item_quality)
        noise = np.random.normal(0, 0.5)
        rating = np.clip(base_rating + noise, 1, 5)
        ratings.append(rating)
    
    rating_data = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })
    
    # 去除重复评分
    rating_data = rating_data.drop_duplicates(subset=['user_id', 'item_id'])
    
    # 生成物品特征数据
    categories = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
    genres = ['Adventure', 'Animation', 'Crime', 'Documentary', 'Family', 'Fantasy']
    
    item_data = pd.DataFrame({
        'item_id': range(n_items),
        'category': np.random.choice(categories, n_items),
        'genre': np.random.choice(genres, n_items),
        'year': np.random.choice(range(1990, 2024), n_items),
        'popularity': np.random.random(n_items)
    })
    
    return rating_data, item_data


def visualize_recommendations(recommendations: List[Tuple[Any, float]], 
                            title: str = "推荐结果"):
    """可视化推荐结果"""
    if not recommendations:
        print("没有推荐结果")
        return
    
    items = [str(item) for item, _ in recommendations]
    scores = [score for _, score in recommendations]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(items, scores, color='skyblue', alpha=0.7)
    plt.xlabel('物品ID')
    plt.ylabel('推荐分数')
    plt.title(title)
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def compare_recommendation_engines():
    """比较不同推荐引擎的性能"""
    print("=== 推荐系统性能比较 ===")
    
    # 生成数据
    rating_data, item_data = generate_sample_data(n_users=500, n_items=200, n_ratings=5000)
    
    # 分割数据
    train_data, test_data = train_test_split(rating_data, test_size=0.2, random_state=42)
    
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"用户数: {rating_data['user_id'].nunique()}")
    print(f"物品数: {rating_data['item_id'].nunique()}")
    
    # 创建推荐引擎
    engines = {
        '用户协同过滤': CollaborativeFiltering(method='user_based'),
        '物品协同过滤': CollaborativeFiltering(method='item_based'),
        '矩阵分解': MatrixFactorization(n_factors=20, n_epochs=50)
    }
    
    # 训练和评估
    evaluator = RecommendationEvaluator()
    results = {}
    
    for name, engine in engines.items():
        print(f"\n训练 {name}...")
        
        # 训练模型
        if name == '基于内容推荐':
            engine.fit(train_data, item_data)
        else:
            engine.fit(train_data)
        
        # 评估模型
        metrics = evaluator.evaluate(engine, test_data)
        results[name] = metrics
        
        print(f"  RMSE: {metrics['RMSE']:.3f}")
        print(f"  MAE: {metrics['MAE']:.3f}")
        print(f"  覆盖率: {metrics['Coverage']:.3f}")
    
    # 可视化结果
    metrics_names = ['RMSE', 'MAE', 'Coverage']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics_names):
        engine_names = list(results.keys())
        values = [results[name][metric] for name in engine_names]
        
        bars = axes[i].bar(engine_names, values, color=['skyblue', 'lightgreen', 'salmon'])
        axes[i].set_title(f'{metric} 比较')
        axes[i].set_xlabel('推荐引擎')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return engines, results


def demo_hybrid_system():
    """演示混合推荐系统"""
    print("\n=== 混合推荐系统演示 ===")
    
    # 生成数据
    rating_data, item_data = generate_sample_data(n_users=300, n_items=100, n_ratings=3000)
    
    # 创建推荐引擎
    cf_engine = CollaborativeFiltering(method='user_based')
    mf_engine = MatrixFactorization(n_factors=15, n_epochs=30)
    
    # 训练引擎
    cf_engine.fit(rating_data)
    mf_engine.fit(rating_data)
    
    # 创建混合推荐系统
    hybrid_system = HybridRecommendationSystem(
        engines={
            'collaborative_filtering': cf_engine,
            'matrix_factorization': mf_engine
        },
        weights={
            'collaborative_filtering': 0.6,
            'matrix_factorization': 0.4
        }
    )
    
    # 为随机用户推荐
    sample_user = rating_data['user_id'].iloc[0]
    print(f"为用户 {sample_user} 推荐:")
    
    # 单独引擎推荐
    cf_recs = cf_engine.recommend(sample_user, 5)
    mf_recs = mf_engine.recommend(sample_user, 5)
    hybrid_recs = hybrid_system.recommend(sample_user, 5)
    
    print("\n协同过滤推荐:")
    for i, (item, score) in enumerate(cf_recs[:5], 1):
        print(f"  {i}. 物品 {item}: {score:.3f}")
    
    print("\n矩阵分解推荐:")
    for i, (item, score) in enumerate(mf_recs[:5], 1):
        print(f"  {i}. 物品 {item}: {score:.3f}")
    
    print("\n混合推荐:")
    for i, (item, score) in enumerate(hybrid_recs[:5], 1):
        print(f"  {i}. 物品 {item}: {score:.3f}")
    
    # 可视化对比
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 协同过滤
    if cf_recs:
        items1 = [str(item) for item, _ in cf_recs[:5]]
        scores1 = [score for _, score in cf_recs[:5]]
        ax1.bar(items1, scores1, color='skyblue', alpha=0.7)
        ax1.set_title('协同过滤推荐')
        ax1.set_xlabel('物品ID')
        ax1.set_ylabel('推荐分数')
    
    # 矩阵分解
    if mf_recs:
        items2 = [str(item) for item, _ in mf_recs[:5]]
        scores2 = [score for _, score in mf_recs[:5]]
        ax2.bar(items2, scores2, color='lightgreen', alpha=0.7)
        ax2.set_title('矩阵分解推荐')
        ax2.set_xlabel('物品ID')
        ax2.set_ylabel('推荐分数')
    
    # 混合推荐
    if hybrid_recs:
        items3 = [str(item) for item, _ in hybrid_recs[:5]]
        scores3 = [score for _, score in hybrid_recs[:5]]
        ax3.bar(items3, scores3, color='salmon', alpha=0.7)
        ax3.set_title('混合推荐')
        ax3.set_xlabel('物品ID')
        ax3.set_ylabel('推荐分数')
    
    plt.tight_layout()
    plt.show()
    
    return hybrid_system


def analyze_user_behavior():
    """分析用户行为"""
    print("\n=== 用户行为分析 ===")
    
    # 生成数据
    rating_data, item_data = generate_sample_data(n_users=200, n_items=100, n_ratings=2000)
    
    # 合并数据
    merged_data = rating_data.merge(item_data, on='item_id', how='left')
    
    # 分析用户评分分布
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 评分分布
    ax1.hist(rating_data['rating'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('评分')
    ax1.set_ylabel('频次')
    ax1.set_title('评分分布')
    
    # 用户活跃度
    user_activity = rating_data.groupby('user_id').size()
    ax2.hist(user_activity, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('评分数量')
    ax2.set_ylabel('用户数')
    ax2.set_title('用户活跃度分布')
    
    # 物品流行度
    item_popularity = rating_data.groupby('item_id').size()
    ax3.hist(item_popularity, bins=20, color='salmon', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('被评分次数')
    ax3.set_ylabel('物品数')
    ax3.set_title('物品流行度分布')
    
    # 类别偏好
    category_ratings = merged_data.groupby('category')['rating'].mean().sort_values(ascending=False)
    ax4.bar(category_ratings.index, category_ratings.values, color='orange', alpha=0.7)
    ax4.set_xlabel('类别')
    ax4.set_ylabel('平均评分')
    ax4.set_title('不同类别的平均评分')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"平均评分: {rating_data['rating'].mean():.2f}")
    print(f"评分标准差: {rating_data['rating'].std():.2f}")
    print(f"最活跃用户评分数: {user_activity.max()}")
    print(f"最受欢迎物品被评分次数: {item_popularity.max()}")
    print(f"稀疏度: {1 - len(rating_data) / (rating_data['user_id'].nunique() * rating_data['item_id'].nunique()):.3f}")


if __name__ == "__main__":
    # 用户行为分析
    analyze_user_behavior()
    
    # 推荐引擎比较
    engines, results = compare_recommendation_engines()
    
    # 混合推荐系统演示
    hybrid_system = demo_hybrid_system()
    
    print("\n=== 推荐系统项目完成 ===")
    print("已生成用户行为分析图表和推荐性能比较图表") 