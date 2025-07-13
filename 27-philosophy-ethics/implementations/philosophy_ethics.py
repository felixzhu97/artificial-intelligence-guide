"""
第27章：哲学、伦理与安全
Philosophy, Ethics, and Safety

本章实现AI系统的哲学、伦理和安全评估，包括：
1. AI伦理框架
2. 偏见检测与缓解
3. 可解释性与透明度
4. 安全性评估
5. 隐私保护
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class EthicsFramework:
    """AI伦理框架"""
    
    def __init__(self):
        self.principles = {
            'fairness': '公平性',
            'transparency': '透明度',
            'accountability': '问责制',
            'beneficence': '有益性',
            'non_maleficence': '无害性',
            'autonomy': '自主性',
            'justice': '正义性'
        }
        
    def evaluate_system(self, system_data: Dict[str, Any]) -> Dict[str, float]:
        """评估AI系统的伦理得分"""
        scores = {}
        
        # 公平性评估
        if 'bias_metrics' in system_data:
            bias_score = 1 - system_data['bias_metrics']['max_bias']
            scores['fairness'] = max(0, min(1, bias_score))
        
        # 透明度评估
        if 'interpretability' in system_data:
            scores['transparency'] = system_data['interpretability']['explainability_score']
        
        # 问责制评估
        if 'auditability' in system_data:
            scores['accountability'] = system_data['auditability']['audit_score']
        
        # 有益性评估
        if 'performance' in system_data:
            scores['beneficence'] = system_data['performance']['accuracy']
        
        # 无害性评估
        if 'safety' in system_data:
            scores['non_maleficence'] = system_data['safety']['safety_score']
        
        # 自主性评估
        if 'user_control' in system_data:
            scores['autonomy'] = system_data['user_control']['control_score']
        
        # 正义性评估
        if 'accessibility' in system_data:
            scores['justice'] = system_data['accessibility']['access_score']
        
        return scores
    
    def generate_ethics_report(self, scores: Dict[str, float]) -> str:
        """生成伦理评估报告"""
        report = "AI系统伦理评估报告\n"
        report += "=" * 40 + "\n\n"
        
        for principle, score in scores.items():
            principle_name = self.principles.get(principle, principle)
            status = "优秀" if score >= 0.8 else "良好" if score >= 0.6 else "需要改进"
            report += f"{principle_name}: {score:.2f} ({status})\n"
        
        average_score = np.mean(list(scores.values()))
        overall_status = "优秀" if average_score >= 0.8 else "良好" if average_score >= 0.6 else "需要改进"
        report += f"\n总体评分: {average_score:.2f} ({overall_status})\n"
        
        return report

class BiasDetector:
    """偏见检测器"""
    
    def __init__(self):
        self.bias_metrics = {}
    
    def detect_demographic_parity(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 protected_attribute: np.ndarray) -> float:
        """检测人口统计平等偏见"""
        groups = np.unique(protected_attribute)
        positive_rates = []
        
        for group in groups:
            group_mask = protected_attribute == group
            positive_rate = np.mean(y_pred[group_mask])
            positive_rates.append(positive_rate)
        
        # 计算最大差异
        max_diff = np.max(positive_rates) - np.min(positive_rates)
        self.bias_metrics['demographic_parity'] = max_diff
        
        return max_diff
    
    def detect_equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            protected_attribute: np.ndarray) -> float:
        """检测等化机会偏见"""
        groups = np.unique(protected_attribute)
        tpr_differences = []
        fpr_differences = []
        
        for group in groups:
            group_mask = protected_attribute == group
            
            # 计算TPR和FPR
            tp = np.sum((y_true[group_mask] == 1) & (y_pred[group_mask] == 1))
            fn = np.sum((y_true[group_mask] == 1) & (y_pred[group_mask] == 0))
            fp = np.sum((y_true[group_mask] == 0) & (y_pred[group_mask] == 1))
            tn = np.sum((y_true[group_mask] == 0) & (y_pred[group_mask] == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_differences.append(tpr)
            fpr_differences.append(fpr)
        
        tpr_diff = np.max(tpr_differences) - np.min(tpr_differences)
        fpr_diff = np.max(fpr_differences) - np.min(fpr_differences)
        
        max_diff = max(tpr_diff, fpr_diff)
        self.bias_metrics['equalized_odds'] = max_diff
        
        return max_diff
    
    def comprehensive_bias_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  protected_attribute: np.ndarray) -> Dict[str, float]:
        """综合偏见分析"""
        dp_bias = self.detect_demographic_parity(y_true, y_pred, protected_attribute)
        eo_bias = self.detect_equalized_odds(y_true, y_pred, protected_attribute)
        
        return {
            'demographic_parity': dp_bias,
            'equalized_odds': eo_bias,
            'max_bias': max(dp_bias, eo_bias)
        }

class ExplainabilityAnalyzer:
    """可解释性分析器"""
    
    def __init__(self):
        self.explanations = {}
    
    def feature_importance_explanation(self, model, feature_names: List[str]) -> Dict[str, float]:
        """基于特征重要性的解释"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
            
            # 排序并返回
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            self.explanations['feature_importance'] = sorted_importance
            
            return sorted_importance
        else:
            return {}
    
    def linear_model_explanation(self, model, feature_names: List[str]) -> Dict[str, float]:
        """线性模型解释"""
        if hasattr(model, 'coef_'):
            coefficients = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            coef_dict = {name: coef for name, coef in zip(feature_names, coefficients)}
            
            # 按绝对值排序
            sorted_coef = dict(sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True))
            self.explanations['linear_coefficients'] = sorted_coef
            
            return sorted_coef
        else:
            return {}
    
    def calculate_explainability_score(self, model, X: np.ndarray, feature_names: List[str]) -> float:
        """计算可解释性得分"""
        score = 0.0
        
        # 基于模型类型的基础得分
        if isinstance(model, LogisticRegression):
            score += 0.8  # 线性模型较易解释
        elif isinstance(model, RandomForestClassifier):
            score += 0.6  # 树模型中等解释性
        else:
            score += 0.3  # 其他模型较难解释
        
        # 基于特征数量的惩罚
        n_features = X.shape[1]
        if n_features > 100:
            score *= 0.5
        elif n_features > 50:
            score *= 0.7
        elif n_features > 20:
            score *= 0.9
        
        return min(1.0, score)

class SafetyEvaluator:
    """安全性评估器"""
    
    def __init__(self):
        self.safety_metrics = {}
    
    def robustness_test(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                       noise_level: float = 0.1) -> float:
        """鲁棒性测试"""
        # 原始性能
        original_pred = model.predict(X_test)
        original_accuracy = accuracy_score(y_test, original_pred)
        
        # 添加噪声后的性能
        noisy_X = X_test + noise_level * np.random.randn(*X_test.shape)
        noisy_pred = model.predict(noisy_X)
        noisy_accuracy = accuracy_score(y_test, noisy_pred)
        
        # 鲁棒性得分
        robustness_score = noisy_accuracy / original_accuracy
        self.safety_metrics['robustness'] = robustness_score
        
        return robustness_score
    
    def adversarial_vulnerability(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                                epsilon: float = 0.1) -> float:
        """对抗样本脆弱性评估"""
        # 简化的对抗样本生成（FGSM方法的简化版）
        perturbation = epsilon * np.random.choice([-1, 1], X_test.shape)
        adversarial_X = X_test + perturbation
        
        # 确保在有效范围内
        adversarial_X = np.clip(adversarial_X, X_test.min(), X_test.max())
        
        # 评估对抗样本性能
        adversarial_pred = model.predict(adversarial_X)
        adversarial_accuracy = accuracy_score(y_test, adversarial_pred)
        
        # 脆弱性得分（越低越脆弱）
        vulnerability_score = adversarial_accuracy
        self.safety_metrics['adversarial_robustness'] = vulnerability_score
        
        return vulnerability_score
    
    def calculate_safety_score(self, model, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """计算综合安全得分"""
        robustness = self.robustness_test(model, X_test, y_test)
        adversarial = self.adversarial_vulnerability(model, X_test, y_test)
        
        # 综合得分
        safety_score = 0.6 * robustness + 0.4 * adversarial
        self.safety_metrics['overall_safety'] = safety_score
        
        return safety_score

class PrivacyProtector:
    """隐私保护器"""
    
    def __init__(self):
        self.privacy_metrics = {}
    
    def differential_privacy_noise(self, data: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        """差分隐私噪声添加"""
        sensitivity = 1.0  # 简化假设
        scale = sensitivity / epsilon
        
        # 添加拉普拉斯噪声
        noise = np.random.laplace(0, scale, data.shape)
        noisy_data = data + noise
        
        return noisy_data
    
    def k_anonymity_check(self, data: np.ndarray, k: int = 5) -> bool:
        """K-匿名性检查"""
        # 简化的K-匿名性检查
        unique_rows, counts = np.unique(data, axis=0, return_counts=True)
        min_count = np.min(counts)
        
        is_k_anonymous = min_count >= k
        self.privacy_metrics['k_anonymity'] = is_k_anonymous
        
        return is_k_anonymous
    
    def calculate_privacy_score(self, data: np.ndarray, epsilon: float = 1.0, k: int = 5) -> float:
        """计算隐私保护得分"""
        score = 0.0
        
        # 差分隐私得分
        dp_score = min(1.0, 1.0 / epsilon)  # epsilon越小，隐私保护越好
        score += 0.6 * dp_score
        
        # K-匿名性得分
        k_anon = self.k_anonymity_check(data, k)
        k_score = 1.0 if k_anon else 0.0
        score += 0.4 * k_score
        
        self.privacy_metrics['privacy_score'] = score
        
        return score

def demonstrate_bias_detection():
    """演示偏见检测"""
    print("=" * 50)
    print("偏见检测演示")
    print("=" * 50)
    
    # 生成带有偏见的数据
    np.random.seed(42)
    n_samples = 1000
    
    # 创建protected attribute (0=Group A, 1=Group B)
    protected_attr = np.random.binomial(1, 0.3, n_samples)
    
    # 生成特征，Group B 有系统性劣势
    X = np.random.randn(n_samples, 5)
    X[protected_attr == 1] += -0.5  # Group B的特征值偏低
    
    # 生成标签，有偏见的决策
    y_true = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # 训练模型
    model = LogisticRegression(random_state=42)
    model.fit(X, y_true)
    y_pred = model.predict(X)
    
    # 偏见检测
    bias_detector = BiasDetector()
    bias_metrics = bias_detector.comprehensive_bias_analysis(y_true, y_pred, protected_attr)
    
    print(f"偏见检测结果：")
    print(f"人口统计平等偏见: {bias_metrics['demographic_parity']:.3f}")
    print(f"等化机会偏见: {bias_metrics['equalized_odds']:.3f}")
    print(f"最大偏见: {bias_metrics['max_bias']:.3f}")
    
    # 可视化偏见
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 不同群体的预测分布
    group_a_pred = y_pred[protected_attr == 0]
    group_b_pred = y_pred[protected_attr == 1]
    
    axes[0].hist(group_a_pred, alpha=0.7, label='Group A', bins=10)
    axes[0].hist(group_b_pred, alpha=0.7, label='Group B', bins=10)
    axes[0].set_xlabel('预测值')
    axes[0].set_ylabel('频数')
    axes[0].set_title('不同群体的预测分布')
    axes[0].legend()
    
    # 偏见指标可视化
    metrics_names = ['人口统计平等', '等化机会', '最大偏见']
    metrics_values = [bias_metrics['demographic_parity'], 
                     bias_metrics['equalized_odds'], 
                     bias_metrics['max_bias']]
    
    bars = axes[1].bar(metrics_names, metrics_values, color=['red', 'orange', 'darkred'])
    axes[1].set_ylabel('偏见程度')
    axes[1].set_title('偏见指标')
    axes[1].set_ylim(0, max(metrics_values) * 1.2)
    
    # 添加数值标签
    for bar, value in zip(bars, metrics_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def demonstrate_explainability():
    """演示可解释性分析"""
    print("=" * 50)
    print("可解释性分析演示")
    print("=" * 50)
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                              n_redundant=2, random_state=42)
    feature_names = [f'特征_{i+1}' for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练不同模型
    models = {
        '逻辑回归': LogisticRegression(random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    explainer = ExplainabilityAnalyzer()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (model_name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        
        # 计算可解释性得分
        explainability_score = explainer.calculate_explainability_score(model, X_test, feature_names)
        
        # 特征重要性解释
        if model_name == '随机森林':
            importance = explainer.feature_importance_explanation(model, feature_names)
            
            # 绘制特征重要性
            features = list(importance.keys())[:8]  # 取前8个特征
            values = list(importance.values())[:8]
            
            axes[i, 0].barh(features, values)
            axes[i, 0].set_xlabel('重要性')
            axes[i, 0].set_title(f'{model_name} - 特征重要性')
            
        else:  # 逻辑回归
            coefficients = explainer.linear_model_explanation(model, feature_names)
            
            # 绘制系数
            features = list(coefficients.keys())[:8]
            values = list(coefficients.values())[:8]
            
            colors = ['red' if v < 0 else 'blue' for v in values]
            axes[i, 0].barh(features, values, color=colors)
            axes[i, 0].set_xlabel('系数')
            axes[i, 0].set_title(f'{model_name} - 特征系数')
        
        # 可解释性得分
        axes[i, 1].bar([model_name], [explainability_score], color='green', alpha=0.7)
        axes[i, 1].set_ylabel('可解释性得分')
        axes[i, 1].set_title(f'{model_name} - 可解释性评估')
        axes[i, 1].set_ylim(0, 1)
        
        # 添加数值标签
        axes[i, 1].text(0, explainability_score + 0.02, f'{explainability_score:.3f}', 
                       ha='center', va='bottom')
        
        print(f"{model_name}可解释性得分: {explainability_score:.3f}")
    
    plt.tight_layout()
    plt.show()

def demonstrate_safety_evaluation():
    """演示安全性评估"""
    print("=" * 50)
    print("安全性评估演示")
    print("=" * 50)
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                              n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 安全性评估
    safety_evaluator = SafetyEvaluator()
    
    # 鲁棒性测试
    robustness_score = safety_evaluator.robustness_test(model, X_test, y_test)
    
    # 对抗样本脆弱性
    adversarial_score = safety_evaluator.adversarial_vulnerability(model, X_test, y_test)
    
    # 综合安全得分
    overall_safety = safety_evaluator.calculate_safety_score(model, X_test, y_test)
    
    print(f"安全性评估结果：")
    print(f"鲁棒性得分: {robustness_score:.3f}")
    print(f"对抗鲁棒性得分: {adversarial_score:.3f}")
    print(f"综合安全得分: {overall_safety:.3f}")
    
    # 可视化安全性指标
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 安全性指标条形图
    safety_metrics = ['鲁棒性', '对抗鲁棒性', '综合安全性']
    safety_values = [robustness_score, adversarial_score, overall_safety]
    colors = ['blue', 'orange', 'green']
    
    bars = axes[0].bar(safety_metrics, safety_values, color=colors, alpha=0.7)
    axes[0].set_ylabel('安全得分')
    axes[0].set_title('安全性评估指标')
    axes[0].set_ylim(0, 1)
    
    # 添加数值标签
    for bar, value in zip(bars, safety_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 不同噪声水平下的鲁棒性
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
    robustness_scores = []
    
    for noise in noise_levels:
        score = safety_evaluator.robustness_test(model, X_test, y_test, noise)
        robustness_scores.append(score)
    
    axes[1].plot(noise_levels, robustness_scores, 'o-', linewidth=2, markersize=8)
    axes[1].set_xlabel('噪声水平')
    axes[1].set_ylabel('鲁棒性得分')
    axes[1].set_title('不同噪声水平下的鲁棒性')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_comprehensive_ethics():
    """演示综合伦理评估"""
    print("=" * 50)
    print("综合伦理评估演示")
    print("=" * 50)
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                              n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 生成protected attribute
    protected_attr = np.random.binomial(1, 0.4, len(y_test))
    
    # 训练模型
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 各种评估
    bias_detector = BiasDetector()
    explainer = ExplainabilityAnalyzer()
    safety_evaluator = SafetyEvaluator()
    privacy_protector = PrivacyProtector()
    ethics_framework = EthicsFramework()
    
    # 收集评估数据
    bias_metrics = bias_detector.comprehensive_bias_analysis(y_test, y_pred, protected_attr)
    explainability_score = explainer.calculate_explainability_score(model, X_test, 
                                                                   [f'特征_{i}' for i in range(X_test.shape[1])])
    safety_score = safety_evaluator.calculate_safety_score(model, X_test, y_test)
    privacy_score = privacy_protector.calculate_privacy_score(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 构建系统数据
    system_data = {
        'bias_metrics': bias_metrics,
        'interpretability': {'explainability_score': explainability_score},
        'safety': {'safety_score': safety_score},
        'performance': {'accuracy': accuracy},
        'auditability': {'audit_score': 0.8},  # 示例值
        'user_control': {'control_score': 0.7},  # 示例值
        'accessibility': {'access_score': 0.9}  # 示例值
    }
    
    # 伦理评估
    ethics_scores = ethics_framework.evaluate_system(system_data)
    ethics_report = ethics_framework.generate_ethics_report(ethics_scores)
    
    print(ethics_report)
    
    # 可视化综合评估
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 伦理原则雷达图
    principles = list(ethics_scores.keys())
    values = list(ethics_scores.values())
    
    angles = np.linspace(0, 2*np.pi, len(principles), endpoint=False).tolist()
    values += values[:1]  # 闭合
    angles += angles[:1]
    
    axes[0, 0] = plt.subplot(2, 2, 1, projection='polar')
    axes[0, 0].plot(angles, values, 'o-', linewidth=2)
    axes[0, 0].fill(angles, values, alpha=0.25)
    axes[0, 0].set_xticks(angles[:-1])
    axes[0, 0].set_xticklabels([ethics_framework.principles[p] for p in principles])
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title('伦理原则评估')
    
    # 综合指标对比
    all_metrics = ['公平性', '透明度', '安全性', '隐私保护', '性能']
    all_values = [1-bias_metrics['max_bias'], explainability_score, safety_score, privacy_score, accuracy]
    
    bars = axes[0, 1].bar(all_metrics, all_values, color=['red', 'blue', 'green', 'purple', 'orange'], alpha=0.7)
    axes[0, 1].set_ylabel('得分')
    axes[0, 1].set_title('综合指标评估')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, all_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 风险评估热图
    risk_matrix = np.array([
        [bias_metrics['max_bias'], 1-safety_score, 1-privacy_score],
        [1-explainability_score, 1-accuracy, 0.3],
        [0.2, 0.1, 0.4]
    ])
    
    risk_labels = ['偏见风险', '安全风险', '隐私风险']
    system_aspects = ['公平性', '性能', '其他']
    
    im = axes[1, 0].imshow(risk_matrix, cmap='Reds', aspect='auto')
    axes[1, 0].set_xticks(range(len(risk_labels)))
    axes[1, 0].set_yticks(range(len(system_aspects)))
    axes[1, 0].set_xticklabels(risk_labels)
    axes[1, 0].set_yticklabels(system_aspects)
    axes[1, 0].set_title('风险评估矩阵')
    
    # 添加数值标签
    for i in range(len(system_aspects)):
        for j in range(len(risk_labels)):
            axes[1, 0].text(j, i, f'{risk_matrix[i, j]:.2f}', ha='center', va='center')
    
    # 改进建议
    axes[1, 1].text(0.1, 0.9, '改进建议:', transform=axes[1, 1].transAxes, fontsize=14, weight='bold')
    
    suggestions = []
    if bias_metrics['max_bias'] > 0.1:
        suggestions.append('• 实施偏见缓解策略')
    if explainability_score < 0.7:
        suggestions.append('• 增强模型可解释性')
    if safety_score < 0.8:
        suggestions.append('• 提高系统安全性')
    if privacy_score < 0.8:
        suggestions.append('• 加强隐私保护措施')
    
    if not suggestions:
        suggestions.append('• 系统整体表现良好')
    
    for i, suggestion in enumerate(suggestions):
        axes[1, 1].text(0.1, 0.8 - i*0.1, suggestion, transform=axes[1, 1].transAxes, fontsize=12)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("第27章：哲学、伦理与安全")
    print("Philosophy, Ethics, and Safety")
    print("=" * 60)
    
    # 演示各个模块
    demonstrate_bias_detection()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_explainability()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_safety_evaluation()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_comprehensive_ethics()
    
    print("\n" + "=" * 60)
    print("哲学、伦理与安全演示完成！")
    print("涵盖内容：")
    print("1. AI伦理框架与评估")
    print("2. 偏见检测与公平性")
    print("3. 可解释性与透明度")
    print("4. 安全性与鲁棒性")
    print("5. 隐私保护与合规")
    print("6. 综合伦理评估报告")

if __name__ == "__main__":
    main() 