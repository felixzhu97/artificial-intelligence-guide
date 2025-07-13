#!/usr/bin/env python3
"""
第13章：概率推理 (Probabilistic Reasoning)

本模块实现了概率推理的核心概念：
- 贝叶斯推理
- 马尔可夫模型
- 概率分布
- 条件概率
- 不确定性推理
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import itertools

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ProbabilityDistribution:
    """概率分布基类"""
    
    def __init__(self, variables: List[str]):
        self.variables = variables
        self.probabilities: Dict[Tuple, float] = {}
    
    def set_probability(self, assignment: Dict[str, Any], probability: float):
        """设置概率值"""
        key = tuple(assignment[var] for var in self.variables)
        self.probabilities[key] = probability
    
    def get_probability(self, assignment: Dict[str, Any]) -> float:
        """获取概率值"""
        key = tuple(assignment[var] for var in self.variables)
        return self.probabilities.get(key, 0.0)
    
    def normalize(self):
        """归一化概率分布"""
        total = sum(self.probabilities.values())
        if total > 0:
            for key in self.probabilities:
                self.probabilities[key] /= total
    
    def marginalize(self, variable: str) -> 'ProbabilityDistribution':
        """边际化（消除变量）"""
        remaining_vars = [v for v in self.variables if v != variable]
        marginal = ProbabilityDistribution(remaining_vars)
        
        # 获取被边际化变量的所有可能值
        values = set()
        for key in self.probabilities:
            var_index = self.variables.index(variable)
            values.add(key[var_index])
        
        # 计算边际概率
        for key in self.probabilities:
            marginal_key = tuple(key[i] for i, var in enumerate(self.variables) if var != variable)
            if marginal_key not in marginal.probabilities:
                marginal.probabilities[marginal_key] = 0
            marginal.probabilities[marginal_key] += self.probabilities[key]
        
        return marginal

class ConditionalProbabilityTable:
    """条件概率表"""
    
    def __init__(self, variable: str, parents: List[str]):
        self.variable = variable
        self.parents = parents
        self.table: Dict[Tuple, Dict[Any, float]] = {}
    
    def set_probability(self, parent_assignment: Dict[str, Any], variable_value: Any, probability: float):
        """设置条件概率"""
        parent_key = tuple(parent_assignment[parent] for parent in self.parents)
        if parent_key not in self.table:
            self.table[parent_key] = {}
        self.table[parent_key][variable_value] = probability
    
    def get_probability(self, parent_assignment: Dict[str, Any], variable_value: Any) -> float:
        """获取条件概率"""
        parent_key = tuple(parent_assignment[parent] for parent in self.parents)
        return self.table.get(parent_key, {}).get(variable_value, 0.0)
    
    def get_distribution(self, parent_assignment: Dict[str, Any]) -> Dict[Any, float]:
        """获取给定父节点值的条件分布"""
        parent_key = tuple(parent_assignment[parent] for parent in self.parents)
        return self.table.get(parent_key, {})

class BayesianNetwork:
    """贝叶斯网络"""
    
    def __init__(self):
        self.variables: Set[str] = set()
        self.parents: Dict[str, List[str]] = {}
        self.cpts: Dict[str, ConditionalProbabilityTable] = {}
        self.domains: Dict[str, List[Any]] = {}
    
    def add_variable(self, variable: str, domain: List[Any], parents: List[str] = None):
        """添加变量"""
        self.variables.add(variable)
        self.domains[variable] = domain
        self.parents[variable] = parents or []
        self.cpts[variable] = ConditionalProbabilityTable(variable, self.parents[variable])
    
    def set_probability(self, variable: str, parent_assignment: Dict[str, Any], variable_value: Any, probability: float):
        """设置条件概率"""
        self.cpts[variable].set_probability(parent_assignment, variable_value, probability)
    
    def query(self, query_var: str, evidence: Dict[str, Any] = None) -> Dict[Any, float]:
        """查询概率（枚举推理）"""
        evidence = evidence or {}
        
        # 获取所有可能的完全赋值
        all_vars = list(self.variables)
        all_assignments = []
        
        for assignment in itertools.product(*[self.domains[var] for var in all_vars]):
            assignment_dict = {var: val for var, val in zip(all_vars, assignment)}
            
            # 检查是否与证据一致
            consistent = True
            for var, val in evidence.items():
                if assignment_dict[var] != val:
                    consistent = False
                    break
            
            if consistent:
                all_assignments.append(assignment_dict)
        
        # 计算查询变量的概率分布
        query_distribution = {}
        for query_value in self.domains[query_var]:
            prob_sum = 0
            for assignment in all_assignments:
                if assignment[query_var] == query_value:
                    prob = self._calculate_joint_probability(assignment)
                    prob_sum += prob
            query_distribution[query_value] = prob_sum
        
        # 归一化
        total = sum(query_distribution.values())
        if total > 0:
            for key in query_distribution:
                query_distribution[key] /= total
        
        return query_distribution
    
    def _calculate_joint_probability(self, assignment: Dict[str, Any]) -> float:
        """计算联合概率"""
        probability = 1.0
        
        for variable in self.variables:
            parent_assignment = {parent: assignment[parent] for parent in self.parents[variable]}
            cpt = self.cpts[variable]
            prob = cpt.get_probability(parent_assignment, assignment[variable])
            probability *= prob
        
        return probability

class MarkovChain:
    """马尔可夫链"""
    
    def __init__(self, states: List[str]):
        self.states = states
        self.transition_matrix = np.zeros((len(states), len(states)))
        self.state_to_index = {state: i for i, state in enumerate(states)}
        self.initial_distribution = np.zeros(len(states))
    
    def set_transition_probability(self, from_state: str, to_state: str, probability: float):
        """设置转移概率"""
        from_idx = self.state_to_index[from_state]
        to_idx = self.state_to_index[to_state]
        self.transition_matrix[from_idx, to_idx] = probability
    
    def set_initial_probability(self, state: str, probability: float):
        """设置初始概率"""
        idx = self.state_to_index[state]
        self.initial_distribution[idx] = probability
    
    def get_stationary_distribution(self) -> Dict[str, float]:
        """计算稳态分布"""
        # 求转移矩阵的左特征向量（特征值为1）
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # 找到特征值最接近1的特征向量
        stationary_idx = np.argmin(np.abs(eigenvalues - 1))
        stationary_vector = np.real(eigenvectors[:, stationary_idx])
        
        # 归一化
        stationary_vector = np.abs(stationary_vector)
        stationary_vector /= np.sum(stationary_vector)
        
        return {state: prob for state, prob in zip(self.states, stationary_vector)}
    
    def simulate(self, steps: int, initial_state: str = None) -> List[str]:
        """模拟马尔可夫链"""
        if initial_state is None:
            # 根据初始分布选择初始状态
            current_idx = np.random.choice(len(self.states), p=self.initial_distribution)
        else:
            current_idx = self.state_to_index[initial_state]
        
        sequence = [self.states[current_idx]]
        
        for _ in range(steps - 1):
            # 根据转移概率选择下一个状态
            next_idx = np.random.choice(len(self.states), p=self.transition_matrix[current_idx])
            sequence.append(self.states[next_idx])
            current_idx = next_idx
        
        return sequence
    
    def forward_probability(self, observations: List[str], steps: int) -> np.ndarray:
        """前向概率算法"""
        alpha = np.zeros((steps, len(self.states)))
        
        # 初始化
        alpha[0] = self.initial_distribution
        
        # 递推
        for t in range(1, steps):
            for j in range(len(self.states)):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_matrix[:, j])
        
        return alpha

class NaiveBayesClassifier:
    """朴素贝叶斯分类器"""
    
    def __init__(self):
        self.class_probabilities: Dict[str, float] = {}
        self.feature_probabilities: Dict[str, Dict[str, Dict[Any, float]]] = {}
        self.classes: Set[str] = set()
        self.features: Set[str] = set()
    
    def train(self, training_data: List[Tuple[Dict[str, Any], str]]):
        """训练分类器"""
        # 统计类别和特征
        class_counts = defaultdict(int)
        feature_value_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for features, class_label in training_data:
            self.classes.add(class_label)
            class_counts[class_label] += 1
            
            for feature_name, feature_value in features.items():
                self.features.add(feature_name)
                feature_value_counts[class_label][feature_name][feature_value] += 1
        
        # 计算类别概率
        total_samples = len(training_data)
        for class_label in self.classes:
            self.class_probabilities[class_label] = class_counts[class_label] / total_samples
        
        # 计算条件概率
        for class_label in self.classes:
            self.feature_probabilities[class_label] = {}
            for feature_name in self.features:
                self.feature_probabilities[class_label][feature_name] = {}
                
                # 获取所有可能的特征值
                all_values = set()
                for features, _ in training_data:
                    if feature_name in features:
                        all_values.add(features[feature_name])
                
                # 计算条件概率（使用拉普拉斯平滑）
                total_class_samples = class_counts[class_label]
                num_values = len(all_values)
                
                for value in all_values:
                    count = feature_value_counts[class_label][feature_name][value]
                    # 拉普拉斯平滑
                    probability = (count + 1) / (total_class_samples + num_values)
                    self.feature_probabilities[class_label][feature_name][value] = probability
    
    def classify(self, features: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """分类"""
        class_scores = {}
        
        for class_label in self.classes:
            # 计算后验概率（对数空间避免下溢）
            log_prob = np.log(self.class_probabilities[class_label])
            
            for feature_name, feature_value in features.items():
                if (class_label in self.feature_probabilities and 
                    feature_name in self.feature_probabilities[class_label] and
                    feature_value in self.feature_probabilities[class_label][feature_name]):
                    
                    feature_prob = self.feature_probabilities[class_label][feature_name][feature_value]
                    log_prob += np.log(feature_prob)
                else:
                    # 处理未见过的特征值（使用很小的概率）
                    log_prob += np.log(1e-10)
            
            class_scores[class_label] = log_prob
        
        # 归一化为概率
        max_score = max(class_scores.values())
        normalized_scores = {}
        total = 0
        
        for class_label, score in class_scores.items():
            prob = np.exp(score - max_score)
            normalized_scores[class_label] = prob
            total += prob
        
        for class_label in normalized_scores:
            normalized_scores[class_label] /= total
        
        # 返回最可能的类别
        best_class = max(normalized_scores, key=normalized_scores.get)
        return best_class, normalized_scores

def demo_bayesian_network():
    """演示贝叶斯网络"""
    print("\n" + "="*50)
    print("贝叶斯网络演示")
    print("="*50)
    
    # 创建简单的医疗诊断网络
    bn = BayesianNetwork()
    
    print("\n构建医疗诊断贝叶斯网络:")
    print("变量: 感冒(Cold), 发烧(Fever), 头痛(Headache)")
    
    # 添加变量
    bn.add_variable("Cold", [True, False])
    bn.add_variable("Fever", [True, False], ["Cold"])
    bn.add_variable("Headache", [True, False], ["Cold"])
    
    # 设置先验概率 P(Cold)
    bn.set_probability("Cold", {}, True, 0.1)   # 感冒概率10%
    bn.set_probability("Cold", {}, False, 0.9)  # 不感冒概率90%
    
    # 设置条件概率 P(Fever|Cold)
    bn.set_probability("Fever", {"Cold": True}, True, 0.8)   # 感冒时发烧概率80%
    bn.set_probability("Fever", {"Cold": True}, False, 0.2)  # 感冒时不发烧概率20%
    bn.set_probability("Fever", {"Cold": False}, True, 0.1)  # 不感冒时发烧概率10%
    bn.set_probability("Fever", {"Cold": False}, False, 0.9) # 不感冒时不发烧概率90%
    
    # 设置条件概率 P(Headache|Cold)
    bn.set_probability("Headache", {"Cold": True}, True, 0.7)   # 感冒时头痛概率70%
    bn.set_probability("Headache", {"Cold": True}, False, 0.3)  # 感冒时不头痛概率30%
    bn.set_probability("Headache", {"Cold": False}, True, 0.2)  # 不感冒时头痛概率20%
    bn.set_probability("Headache", {"Cold": False}, False, 0.8) # 不感冒时不头痛概率80%
    
    # 查询：没有任何症状时感冒的概率
    print("\n查询1: P(Cold) - 先验概率")
    cold_prior = bn.query("Cold")
    for value, prob in cold_prior.items():
        print(f"  P(Cold={value}) = {prob:.3f}")
    
    # 查询：发烧时感冒的概率
    print("\n查询2: P(Cold|Fever=True) - 发烧时感冒概率")
    cold_given_fever = bn.query("Cold", {"Fever": True})
    for value, prob in cold_given_fever.items():
        print(f"  P(Cold={value}|Fever=True) = {prob:.3f}")
    
    # 查询：发烧且头痛时感冒的概率
    print("\n查询3: P(Cold|Fever=True, Headache=True) - 发烧且头痛时感冒概率")
    cold_given_symptoms = bn.query("Cold", {"Fever": True, "Headache": True})
    for value, prob in cold_given_symptoms.items():
        print(f"  P(Cold={value}|Fever=True, Headache=True) = {prob:.3f}")

def demo_markov_chain():
    """演示马尔可夫链"""
    print("\n" + "="*50)
    print("马尔可夫链演示")
    print("="*50)
    
    # 创建天气马尔可夫链
    weather_states = ["晴天", "雨天", "阴天"]
    mc = MarkovChain(weather_states)
    
    print("\n构建天气马尔可夫链:")
    
    # 设置转移概率
    # 从晴天转移
    mc.set_transition_probability("晴天", "晴天", 0.7)
    mc.set_transition_probability("晴天", "阴天", 0.2)
    mc.set_transition_probability("晴天", "雨天", 0.1)
    
    # 从阴天转移
    mc.set_transition_probability("阴天", "晴天", 0.3)
    mc.set_transition_probability("阴天", "阴天", 0.4)
    mc.set_transition_probability("阴天", "雨天", 0.3)
    
    # 从雨天转移
    mc.set_transition_probability("雨天", "晴天", 0.2)
    mc.set_transition_probability("雨天", "阴天", 0.6)
    mc.set_transition_probability("雨天", "雨天", 0.2)
    
    # 设置初始分布
    mc.set_initial_probability("晴天", 0.6)
    mc.set_initial_probability("阴天", 0.3)
    mc.set_initial_probability("雨天", 0.1)
    
    print("转移矩阵:")
    print(f"{'':>6} {'晴天':>6} {'阴天':>6} {'雨天':>6}")
    for i, from_state in enumerate(weather_states):
        row = f"{from_state:>6}"
        for j, to_state in enumerate(weather_states):
            row += f" {mc.transition_matrix[i, j]:>6.2f}"
        print(row)
    
    # 计算稳态分布
    stationary = mc.get_stationary_distribution()
    print(f"\n稳态分布:")
    for state, prob in stationary.items():
        print(f"  P({state}) = {prob:.3f}")
    
    # 模拟天气序列
    print(f"\n模拟10天天气:")
    sequence = mc.simulate(10, "晴天")
    print(f"  {' -> '.join(sequence)}")

def demo_naive_bayes():
    """演示朴素贝叶斯分类器"""
    print("\n" + "="*50)
    print("朴素贝叶斯分类器演示")
    print("="*50)
    
    # 创建文本分类数据
    training_data = [
        ({"好": True, "电影": True, "喜欢": True}, "正面"),
        ({"好": True, "电影": True, "推荐": True}, "正面"),
        ({"不错": True, "值得": True, "观看": True}, "正面"),
        ({"精彩": True, "演技": True, "很棒": True}, "正面"),
        ({"糟糕": True, "电影": True, "失望": True}, "负面"),
        ({"无聊": True, "剧情": True, "差": True}, "负面"),
        ({"浪费": True, "时间": True, "不好": True}, "负面"),
        ({"烂": True, "片": True, "不推荐": True}, "负面"),
    ]
    
    print("\n训练朴素贝叶斯分类器:")
    print("训练数据（电影评论分类）:")
    for features, label in training_data:
        feature_str = ", ".join(f for f in features.keys())
        print(f"  [{feature_str}] -> {label}")
    
    # 训练分类器
    nb = NaiveBayesClassifier()
    nb.train(training_data)
    
    print(f"\n学习到的类别概率:")
    for class_label, prob in nb.class_probabilities.items():
        print(f"  P({class_label}) = {prob:.3f}")
    
    # 测试分类
    test_cases = [
        {"好": True, "推荐": True},
        {"糟糕": True, "无聊": True},
        {"不错": True, "电影": True},
        {"烂": True, "不好": True}
    ]
    
    print(f"\n分类测试:")
    for test_features in test_cases:
        predicted_class, probabilities = nb.classify(test_features)
        feature_str = ", ".join(test_features.keys())
        print(f"  [{feature_str}] -> {predicted_class}")
        for class_label, prob in probabilities.items():
            print(f"    P({class_label}) = {prob:.3f}")

def demo_uncertainty_reasoning():
    """演示不确定性推理"""
    print("\n" + "="*50)
    print("不确定性推理演示")
    print("="*50)
    
    print("\n贝叶斯定理应用 - 医疗诊断:")
    print("设置:")
    print("  - 疾病患病率: P(Disease) = 0.01")
    print("  - 检测敏感度: P(+|Disease) = 0.95")
    print("  - 检测特异度: P(-|¬Disease) = 0.98")
    
    # 使用贝叶斯定理计算
    p_disease = 0.01
    p_no_disease = 0.99
    p_positive_given_disease = 0.95
    p_negative_given_no_disease = 0.98
    p_positive_given_no_disease = 0.02
    
    # P(+) = P(+|Disease)P(Disease) + P(+|¬Disease)P(¬Disease)
    p_positive = (p_positive_given_disease * p_disease + 
                 p_positive_given_no_disease * p_no_disease)
    
    # P(Disease|+) = P(+|Disease)P(Disease) / P(+)
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
    
    print(f"\n计算结果:")
    print(f"  P(检测阳性) = {p_positive:.4f}")
    print(f"  P(患病|检测阳性) = {p_disease_given_positive:.4f}")
    print(f"  解释: 即使检测阳性，患病概率仍然较低 ({p_disease_given_positive*100:.1f}%)")

def visualize_probability_distributions():
    """可视化概率分布"""
    print("\n" + "="*50)
    print("概率分布可视化")
    print("="*50)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('概率分布示例', fontsize=16)
    
    # 1. 二项分布
    n, p = 20, 0.3
    x = np.arange(0, n+1)
    y = [math.comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in x]
    axes[0, 0].bar(x, y, alpha=0.7)
    axes[0, 0].set_title(f'二项分布 (n={n}, p={p})')
    axes[0, 0].set_xlabel('成功次数')
    axes[0, 0].set_ylabel('概率')
    
    # 2. 正态分布
    x = np.linspace(-4, 4, 100)
    y = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
    axes[0, 1].plot(x, y, 'b-', linewidth=2)
    axes[0, 1].set_title('标准正态分布')
    axes[0, 1].set_xlabel('值')
    axes[0, 1].set_ylabel('概率密度')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 指数分布
    x = np.linspace(0, 5, 100)
    lambda_param = 1.5
    y = lambda_param * np.exp(-lambda_param * x)
    axes[1, 0].plot(x, y, 'r-', linewidth=2)
    axes[1, 0].set_title(f'指数分布 (λ={lambda_param})')
    axes[1, 0].set_xlabel('值')
    axes[1, 0].set_ylabel('概率密度')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 均匀分布
    x = np.array([0, 0, 2, 2, 4, 4])
    y = np.array([0, 0.5, 0.5, 0.5, 0.5, 0])
    axes[1, 1].plot(x, y, 'g-', linewidth=2)
    axes[1, 1].fill_between([0, 4], [0.5, 0.5], alpha=0.3)
    axes[1, 1].set_title('均匀分布 [0, 4]')
    axes[1, 1].set_xlabel('值')
    axes[1, 1].set_ylabel('概率密度')
    axes[1, 1].set_ylim(0, 0.6)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('probability_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("概率分布图已保存为 'probability_distributions.png'")

def run_comprehensive_demo():
    """运行完整演示"""
    print("🎲 第13章：概率推理 - 完整演示")
    print("="*60)
    
    # 运行各个演示
    demo_bayesian_network()
    demo_markov_chain()
    demo_naive_bayes()
    demo_uncertainty_reasoning()
    visualize_probability_distributions()
    
    print("\n" + "="*60)
    print("概率推理演示完成！")
    print("="*60)
    print("\n📚 学习要点:")
    print("• 贝叶斯网络提供了概率依赖关系的图形表示")
    print("• 马尔可夫链建模状态序列的概率演化")
    print("• 朴素贝叶斯是简单而有效的概率分类方法")
    print("• 贝叶斯定理是不确定性推理的核心")
    print("• 概率推理帮助在不确定环境中做出理性决策")

if __name__ == "__main__":
    run_comprehensive_demo() 