"""
概率推理实现

包含贝叶斯网络、马尔可夫链、隐马尔可夫模型等
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import itertools
from abc import ABC, abstractmethod


class ProbabilityDistribution:
    """概率分布类"""
    
    def __init__(self, variables: List[str], probabilities: Dict[Tuple, float]):
        self.variables = variables
        self.probabilities = probabilities
        self.normalize()
    
    def normalize(self):
        """归一化概率"""
        total = sum(self.probabilities.values())
        if total > 0:
            for key in self.probabilities:
                self.probabilities[key] /= total
    
    def get_probability(self, assignment: Dict[str, bool]) -> float:
        """获取指定赋值的概率"""
        key = tuple(assignment[var] for var in self.variables)
        return self.probabilities.get(key, 0.0)
    
    def marginalize(self, variables_to_keep: List[str]) -> 'ProbabilityDistribution':
        """边际化"""
        new_probs = defaultdict(float)
        
        for assignment, prob in self.probabilities.items():
            var_dict = dict(zip(self.variables, assignment))
            new_key = tuple(var_dict[var] for var in variables_to_keep)
            new_probs[new_key] += prob
        
        return ProbabilityDistribution(variables_to_keep, dict(new_probs))
    
    def condition(self, evidence: Dict[str, bool]) -> 'ProbabilityDistribution':
        """条件概率"""
        new_probs = {}
        
        for assignment, prob in self.probabilities.items():
            var_dict = dict(zip(self.variables, assignment))
            
            # 检查是否与证据一致
            consistent = True
            for var, value in evidence.items():
                if var in var_dict and var_dict[var] != value:
                    consistent = False
                    break
            
            if consistent:
                # 移除证据变量
                remaining_vars = [var for var in self.variables if var not in evidence]
                if remaining_vars:
                    new_key = tuple(var_dict[var] for var in remaining_vars)
                    new_probs[new_key] = prob
                else:
                    new_probs[()] = prob
        
        remaining_vars = [var for var in self.variables if var not in evidence]
        return ProbabilityDistribution(remaining_vars, new_probs)


class BayesianNetwork:
    """贝叶斯网络"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(list)
        self.parents = defaultdict(list)
        self.children = defaultdict(list)
    
    def add_node(self, name: str, parents: List[str], cpt: Dict[Tuple, float]):
        """添加节点"""
        self.nodes[name] = {
            'parents': parents,
            'cpt': cpt  # 条件概率表
        }
        
        # 更新图结构
        self.parents[name] = parents
        for parent in parents:
            self.edges[parent].append(name)
            self.children[parent].append(name)
    
    def get_probability(self, variable: str, value: bool, evidence: Dict[str, bool]) -> float:
        """获取条件概率"""
        node = self.nodes[variable]
        parents = node['parents']
        cpt = node['cpt']
        
        # 构建父节点的赋值
        parent_values = []
        for parent in parents:
            if parent in evidence:
                parent_values.append(evidence[parent])
            else:
                # 如果父节点未知，需要边际化
                return 0.5  # 简化处理
        
        # 查找条件概率表
        key = tuple(parent_values + [value])
        return cpt.get(key, 0.0)
    
    def enumerate_inference(self, query: str, evidence: Dict[str, bool]) -> float:
        """枚举推理"""
        # 获取所有变量
        all_vars = set(self.nodes.keys())
        
        # 计算分子和分母
        numerator = self.enumerate_all(all_vars, {**evidence, query: True})
        denominator = (self.enumerate_all(all_vars, {**evidence, query: True}) +
                      self.enumerate_all(all_vars, {**evidence, query: False}))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def enumerate_all(self, variables: Set[str], evidence: Dict[str, bool]) -> float:
        """枚举所有可能的赋值"""
        if not variables:
            return 1.0
        
        var = variables.pop()
        variables_copy = variables.copy()
        
        if var in evidence:
            # 变量已赋值
            prob = self.get_probability(var, evidence[var], evidence)
            return prob * self.enumerate_all(variables_copy, evidence)
        else:
            # 变量未赋值，枚举所有可能值
            prob_true = self.get_probability(var, True, evidence)
            prob_false = self.get_probability(var, False, evidence)
            
            evidence_true = {**evidence, var: True}
            evidence_false = {**evidence, var: False}
            
            return (prob_true * self.enumerate_all(variables_copy, evidence_true) +
                    prob_false * self.enumerate_all(variables_copy, evidence_false))
    
    def variable_elimination(self, query: str, evidence: Dict[str, bool]) -> float:
        """变量消除算法"""
        # 简化实现
        return self.enumerate_inference(query, evidence)
    
    def gibbs_sampling(self, query: str, evidence: Dict[str, bool], 
                      num_samples: int = 1000) -> float:
        """吉布斯采样"""
        # 初始化随机状态
        state = {}
        for var in self.nodes:
            if var in evidence:
                state[var] = evidence[var]
            else:
                state[var] = random.choice([True, False])
        
        # 非证据变量
        non_evidence_vars = [var for var in self.nodes if var not in evidence]
        
        # 采样
        samples = []
        for _ in range(num_samples):
            # 更新每个非证据变量
            for var in non_evidence_vars:
                # 计算条件概率
                prob_true = self.get_markov_blanket_probability(var, True, state)
                prob_false = self.get_markov_blanket_probability(var, False, state)
                
                # 归一化
                total = prob_true + prob_false
                if total > 0:
                    prob_true /= total
                    prob_false /= total
                
                # 采样
                state[var] = random.random() < prob_true
            
            samples.append(state[query])
        
        # 计算查询变量为True的概率
        true_count = sum(samples)
        return true_count / len(samples)
    
    def get_markov_blanket_probability(self, var: str, value: bool, state: Dict[str, bool]) -> float:
        """获取马尔可夫毯概率"""
        # 简化实现：只考虑条件概率表
        temp_state = state.copy()
        temp_state[var] = value
        
        prob = self.get_probability(var, value, temp_state)
        
        # 考虑子节点
        for child in self.children[var]:
            child_prob = self.get_probability(child, temp_state[child], temp_state)
            prob *= child_prob
        
        return prob


class HiddenMarkovModel:
    """隐马尔可夫模型"""
    
    def __init__(self, states: List[str], observations: List[str]):
        self.states = states
        self.observations = observations
        self.n_states = len(states)
        self.n_observations = len(observations)
        
        # 初始状态概率
        self.initial_probs = np.ones(self.n_states) / self.n_states
        
        # 转移概率矩阵
        self.transition_probs = np.ones((self.n_states, self.n_states)) / self.n_states
        
        # 观测概率矩阵
        self.emission_probs = np.ones((self.n_states, self.n_observations)) / self.n_observations
    
    def set_initial_probs(self, probs: np.ndarray):
        """设置初始概率"""
        self.initial_probs = probs
    
    def set_transition_probs(self, probs: np.ndarray):
        """设置转移概率"""
        self.transition_probs = probs
    
    def set_emission_probs(self, probs: np.ndarray):
        """设置观测概率"""
        self.emission_probs = probs
    
    def forward_algorithm(self, observations: List[int]) -> np.ndarray:
        """前向算法"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # 初始化
        alpha[0] = self.initial_probs * self.emission_probs[:, observations[0]]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = (np.sum(alpha[t-1] * self.transition_probs[:, j]) *
                              self.emission_probs[j, observations[t]])
        
        return alpha
    
    def backward_algorithm(self, observations: List[int]) -> np.ndarray:
        """后向算法"""
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # 初始化
        beta[T-1] = np.ones(self.n_states)
        
        # 递推
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transition_probs[i, :] *
                                   self.emission_probs[:, observations[t+1]] *
                                   beta[t+1])
        
        return beta
    
    def viterbi_algorithm(self, observations: List[int]) -> List[int]:
        """维特比算法"""
        T = len(observations)
        viterbi = np.zeros((T, self.n_states))
        path = np.zeros((T, self.n_states), dtype=int)
        
        # 初始化
        viterbi[0] = self.initial_probs * self.emission_probs[:, observations[0]]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                # 找到最大概率路径
                prev_probs = viterbi[t-1] * self.transition_probs[:, j]
                best_prev = np.argmax(prev_probs)
                
                viterbi[t, j] = prev_probs[best_prev] * self.emission_probs[j, observations[t]]
                path[t, j] = best_prev
        
        # 回溯找最优路径
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(viterbi[T-1])
        
        for t in range(T-2, -1, -1):
            best_path[t] = path[t+1, best_path[t+1]]
        
        return best_path.tolist()
    
    def baum_welch_algorithm(self, observations: List[int], max_iter: int = 100) -> None:
        """Baum-Welch算法（EM算法）"""
        T = len(observations)
        
        for iteration in range(max_iter):
            # E步：计算前向后向概率
            alpha = self.forward_algorithm(observations)
            beta = self.backward_algorithm(observations)
            
            # 计算gamma和xi
            gamma = np.zeros((T, self.n_states))
            xi = np.zeros((T-1, self.n_states, self.n_states))
            
            for t in range(T):
                gamma[t] = alpha[t] * beta[t]
                gamma[t] /= np.sum(gamma[t])
            
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * self.transition_probs[i, j] *
                                      self.emission_probs[j, observations[t+1]] *
                                      beta[t+1, j])
                
                xi[t] /= np.sum(xi[t])
            
            # M步：更新参数
            # 更新初始概率
            self.initial_probs = gamma[0]
            
            # 更新转移概率
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.transition_probs[i, j] = (np.sum(xi[:, i, j]) /
                                                  np.sum(gamma[:-1, i]))
            
            # 更新观测概率
            for j in range(self.n_states):
                for k in range(self.n_observations):
                    numerator = np.sum(gamma[[t for t in range(T) if observations[t] == k], j])
                    denominator = np.sum(gamma[:, j])
                    if denominator > 0:
                        self.emission_probs[j, k] = numerator / denominator


class ParticleFilter:
    """粒子滤波"""
    
    def __init__(self, num_particles: int, state_dim: int):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles
    
    def predict(self, motion_model, noise_std: float = 0.1):
        """预测步骤"""
        for i in range(self.num_particles):
            self.particles[i] = motion_model(self.particles[i])
            self.particles[i] += np.random.normal(0, noise_std, self.state_dim)
    
    def update(self, observation, observation_model, noise_std: float = 0.1):
        """更新步骤"""
        # 计算权重
        for i in range(self.num_particles):
            predicted_obs = observation_model(self.particles[i])
            likelihood = self.gaussian_likelihood(observation, predicted_obs, noise_std)
            self.weights[i] *= likelihood
        
        # 归一化权重
        self.weights /= np.sum(self.weights)
        
        # 重采样
        self.resample()
    
    def gaussian_likelihood(self, observation, prediction, noise_std: float) -> float:
        """高斯似然"""
        diff = observation - prediction
        return np.exp(-0.5 * np.sum(diff ** 2) / (noise_std ** 2))
    
    def resample(self):
        """重采样"""
        # 系统重采样
        cumulative_weights = np.cumsum(self.weights)
        new_particles = np.zeros_like(self.particles)
        
        u = np.random.uniform(0, 1/self.num_particles)
        i = 0
        
        for j in range(self.num_particles):
            while u > cumulative_weights[i]:
                i += 1
            new_particles[j] = self.particles[i]
            u += 1/self.num_particles
        
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_estimate(self) -> np.ndarray:
        """获取状态估计"""
        return np.average(self.particles, weights=self.weights, axis=0)


class MarkovChain:
    """马尔可夫链"""
    
    def __init__(self, states: List[str], transition_matrix: np.ndarray):
        self.states = states
        self.transition_matrix = transition_matrix
        self.n_states = len(states)
    
    def get_stationary_distribution(self) -> np.ndarray:
        """获取稳态分布"""
        # 计算转移矩阵的特征向量
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # 找到特征值为1的特征向量
        stationary_idx = np.argmax(eigenvalues)
        stationary = eigenvectors[:, stationary_idx].real
        
        # 归一化
        stationary = stationary / np.sum(stationary)
        
        return stationary
    
    def simulate(self, initial_state: int, num_steps: int) -> List[int]:
        """模拟马尔可夫链"""
        states = [initial_state]
        current_state = initial_state
        
        for _ in range(num_steps):
            # 根据转移概率选择下一个状态
            next_state = np.random.choice(
                self.n_states, 
                p=self.transition_matrix[current_state]
            )
            states.append(next_state)
            current_state = next_state
        
        return states


def demo_bayesian_network():
    """演示贝叶斯网络"""
    print("贝叶斯网络演示")
    print("=" * 30)
    
    # 创建简单的贝叶斯网络
    # 雨天 -> 草地湿润
    # 洒水器 -> 草地湿润
    bn = BayesianNetwork()
    
    # 添加节点
    bn.add_node("Rain", [], {(True,): 0.3, (False,): 0.7})
    bn.add_node("Sprinkler", [], {(True,): 0.2, (False,): 0.8})
    bn.add_node("WetGrass", ["Rain", "Sprinkler"], {
        (True, True, True): 0.99,
        (True, True, False): 0.01,
        (True, False, True): 0.8,
        (True, False, False): 0.2,
        (False, True, True): 0.9,
        (False, True, False): 0.1,
        (False, False, True): 0.1,
        (False, False, False): 0.9
    })
    
    # 推理
    evidence = {"WetGrass": True}
    prob_rain = bn.enumerate_inference("Rain", evidence)
    prob_sprinkler = bn.enumerate_inference("Sprinkler", evidence)
    
    print(f"给定草地湿润，下雨的概率: {prob_rain:.3f}")
    print(f"给定草地湿润，洒水器开启的概率: {prob_sprinkler:.3f}")
    
    # 吉布斯采样
    prob_rain_gibbs = bn.gibbs_sampling("Rain", evidence, 1000)
    print(f"吉布斯采样 - 下雨的概率: {prob_rain_gibbs:.3f}")


def demo_hmm():
    """演示隐马尔可夫模型"""
    print("隐马尔可夫模型演示")
    print("=" * 30)
    
    # 创建天气预测HMM
    states = ["Sunny", "Rainy"]
    observations = ["Dry", "Wet"]
    
    hmm = HiddenMarkovModel(states, observations)
    
    # 设置参数
    hmm.set_initial_probs(np.array([0.6, 0.4]))
    hmm.set_transition_probs(np.array([
        [0.7, 0.3],  # Sunny -> Sunny, Sunny -> Rainy
        [0.4, 0.6]   # Rainy -> Sunny, Rainy -> Rainy
    ]))
    hmm.set_emission_probs(np.array([
        [0.8, 0.2],  # Sunny -> Dry, Sunny -> Wet
        [0.3, 0.7]   # Rainy -> Dry, Rainy -> Wet
    ]))
    
    # 观测序列
    obs_sequence = [0, 1, 0, 1, 1]  # Dry, Wet, Dry, Wet, Wet
    
    # 前向算法
    alpha = hmm.forward_algorithm(obs_sequence)
    print(f"前向概率最后一步: {alpha[-1]}")
    
    # 维特比算法
    best_path = hmm.viterbi_algorithm(obs_sequence)
    print(f"最可能的状态序列: {[states[i] for i in best_path]}")


def demo_particle_filter():
    """演示粒子滤波"""
    print("粒子滤波演示")
    print("=" * 30)
    
    # 创建粒子滤波器
    pf = ParticleFilter(num_particles=100, state_dim=2)
    
    # 定义运动模型
    def motion_model(state):
        return state + np.array([0.1, 0.05])
    
    # 定义观测模型
    def observation_model(state):
        return state
    
    # 模拟轨迹
    true_trajectory = []
    estimated_trajectory = []
    
    for t in range(10):
        # 预测
        pf.predict(motion_model)
        
        # 模拟观测
        true_state = np.array([t * 0.1, t * 0.05])
        observation = true_state + np.random.normal(0, 0.1, 2)
        
        # 更新
        pf.update(observation, observation_model)
        
        # 记录轨迹
        true_trajectory.append(true_state)
        estimated_trajectory.append(pf.get_estimate())
    
    print("真实轨迹 vs 估计轨迹:")
    for i, (true, est) in enumerate(zip(true_trajectory, estimated_trajectory)):
        print(f"时刻 {i}: 真实 {true}, 估计 {est}")


def demo_markov_chain():
    """演示马尔可夫链"""
    print("马尔可夫链演示")
    print("=" * 30)
    
    # 创建天气马尔可夫链
    states = ["Sunny", "Cloudy", "Rainy"]
    transition_matrix = np.array([
        [0.8, 0.15, 0.05],  # Sunny -> Sunny, Cloudy, Rainy
        [0.3, 0.4, 0.3],    # Cloudy -> Sunny, Cloudy, Rainy
        [0.2, 0.3, 0.5]     # Rainy -> Sunny, Cloudy, Rainy
    ])
    
    mc = MarkovChain(states, transition_matrix)
    
    # 计算稳态分布
    stationary = mc.get_stationary_distribution()
    print("稳态分布:")
    for i, state in enumerate(states):
        print(f"{state}: {stationary[i]:.3f}")
    
    # 模拟
    simulation = mc.simulate(0, 20)  # 从Sunny开始
    print(f"\n模拟序列: {[states[i] for i in simulation]}")


if __name__ == "__main__":
    # 演示不同的概率推理方法
    demo_bayesian_network()
    print("\n" + "="*50)
    demo_hmm()
    print("\n" + "="*50)
    demo_particle_filter()
    print("\n" + "="*50)
    demo_markov_chain()
    
    print("\n✅ 概率推理演示完成！") 