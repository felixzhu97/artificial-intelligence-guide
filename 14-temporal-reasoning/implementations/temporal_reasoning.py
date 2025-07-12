"""
第14章：时序推理
实现了AIMA第14章中的时序推理算法：HMM、卡尔曼滤波、粒子滤波等
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any, Callable
from abc import ABC, abstractmethod
import random
from collections import defaultdict
from scipy.stats import multivariate_normal
import math


class HiddenMarkovModel:
    """隐马尔可夫模型（HMM）实现"""
    
    def __init__(self, states: List[str], observations: List[str],
                 transition_prob: Dict[Tuple[str, str], float],
                 emission_prob: Dict[Tuple[str, str], float],
                 initial_prob: Dict[str, float]):
        """
        初始化HMM
        
        Args:
            states: 隐状态列表
            observations: 观测状态列表
            transition_prob: 状态转移概率 P(s_t|s_{t-1})
            emission_prob: 观测概率 P(o_t|s_t)
            initial_prob: 初始状态概率 P(s_0)
        """
        self.states = states
        self.observations = observations
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.initial_prob = initial_prob
    
    def forward_algorithm(self, observation_sequence: List[str]) -> Tuple[np.ndarray, float]:
        """
        前向算法：计算观测序列的概率
        
        Returns:
            (alpha, probability): 前向概率矩阵和总概率
        """
        T = len(observation_sequence)
        N = len(self.states)
        
        # 初始化前向概率矩阵
        alpha = np.zeros((T, N))
        
        # 初始化（t=0）
        for i, state in enumerate(self.states):
            obs = observation_sequence[0]
            alpha[0, i] = (self.initial_prob.get(state, 0) * 
                          self.emission_prob.get((state, obs), 0))
        
        # 递推（t=1,2,...,T-1）
        for t in range(1, T):
            obs = observation_sequence[t]
            for j, curr_state in enumerate(self.states):
                alpha[t, j] = 0
                for i, prev_state in enumerate(self.states):
                    alpha[t, j] += (alpha[t-1, i] * 
                                   self.transition_prob.get((prev_state, curr_state), 0))
                alpha[t, j] *= self.emission_prob.get((curr_state, obs), 0)
        
        # 计算总概率
        total_prob = np.sum(alpha[T-1, :])
        
        return alpha, total_prob
    
    def backward_algorithm(self, observation_sequence: List[str]) -> np.ndarray:
        """
        后向算法：计算后向概率
        
        Returns:
            beta: 后向概率矩阵
        """
        T = len(observation_sequence)
        N = len(self.states)
        
        # 初始化后向概率矩阵
        beta = np.zeros((T, N))
        
        # 初始化（t=T-1）
        beta[T-1, :] = 1
        
        # 递推（t=T-2,T-3,...,0）
        for t in range(T-2, -1, -1):
            obs_next = observation_sequence[t+1]
            for i, curr_state in enumerate(self.states):
                beta[t, i] = 0
                for j, next_state in enumerate(self.states):
                    beta[t, i] += (self.transition_prob.get((curr_state, next_state), 0) *
                                  self.emission_prob.get((next_state, obs_next), 0) *
                                  beta[t+1, j])
        
        return beta
    
    def viterbi_algorithm(self, observation_sequence: List[str]) -> Tuple[List[str], float]:
        """
        维特比算法：找到最可能的隐状态序列
        
        Returns:
            (path, probability): 最可能的状态序列和其概率
        """
        T = len(observation_sequence)
        N = len(self.states)
        
        # 初始化
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        
        # 初始化（t=0）
        for i, state in enumerate(self.states):
            obs = observation_sequence[0]
            delta[0, i] = (self.initial_prob.get(state, 0) *
                          self.emission_prob.get((state, obs), 0))
            psi[0, i] = 0
        
        # 递推（t=1,2,...,T-1）
        for t in range(1, T):
            obs = observation_sequence[t]
            for j, curr_state in enumerate(self.states):
                # 找到最大概率路径
                max_prob = -1
                max_state = 0
                for i, prev_state in enumerate(self.states):
                    prob = (delta[t-1, i] * 
                           self.transition_prob.get((prev_state, curr_state), 0))
                    if prob > max_prob:
                        max_prob = prob
                        max_state = i
                
                delta[t, j] = max_prob * self.emission_prob.get((curr_state, obs), 0)
                psi[t, j] = max_state
        
        # 终止：找到最优路径
        best_prob = np.max(delta[T-1, :])
        best_last_state = np.argmax(delta[T-1, :])
        
        # 回溯最优路径
        path = [0] * T
        path[T-1] = best_last_state
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        # 转换为状态名称
        state_path = [self.states[i] for i in path]
        
        return state_path, best_prob
    
    def baum_welch_algorithm(self, observation_sequences: List[List[str]], 
                           max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Baum-Welch算法：HMM参数学习
        """
        prev_log_likelihood = float('-inf')
        
        for iteration in range(max_iterations):
            # E步：计算期望
            gamma_sum = defaultdict(float)
            xi_sum = defaultdict(float)
            obs_count = defaultdict(float)
            total_gamma = defaultdict(float)
            
            log_likelihood = 0
            
            for obs_seq in observation_sequences:
                alpha, prob = self.forward_algorithm(obs_seq)
                beta = self.backward_algorithm(obs_seq)
                log_likelihood += np.log(prob + 1e-10)
                
                T = len(obs_seq)
                N = len(self.states)
                
                # 计算gamma和xi
                for t in range(T):
                    for i, state in enumerate(self.states):
                        gamma = alpha[t, i] * beta[t, i] / (prob + 1e-10)
                        gamma_sum[state] += gamma
                        
                        if t == 0:
                            total_gamma[state] += gamma
                        
                        obs_count[(state, obs_seq[t])] += gamma
                        
                        if t < T - 1:
                            for j, next_state in enumerate(self.states):
                                xi = (alpha[t, i] * 
                                     self.transition_prob.get((state, next_state), 0) *
                                     self.emission_prob.get((next_state, obs_seq[t+1]), 0) *
                                     beta[t+1, j] / (prob + 1e-10))
                                xi_sum[(state, next_state)] += xi
            
            # M步：更新参数
            # 更新初始概率
            total_initial = sum(total_gamma.values())
            for state in self.states:
                self.initial_prob[state] = total_gamma[state] / (total_initial + 1e-10)
            
            # 更新转移概率
            for state in self.states:
                state_total = sum(xi_sum.get((state, next_state), 0) 
                                 for next_state in self.states)
                for next_state in self.states:
                    self.transition_prob[(state, next_state)] = (
                        xi_sum.get((state, next_state), 0) / (state_total + 1e-10))
            
            # 更新发射概率
            for state in self.states:
                state_total = gamma_sum[state]
                for obs in self.observations:
                    self.emission_prob[(state, obs)] = (
                        obs_count.get((state, obs), 0) / (state_total + 1e-10))
            
            # 检查收敛
            if abs(log_likelihood - prev_log_likelihood) < tolerance:
                break
            prev_log_likelihood = log_likelihood


class KalmanFilter:
    """卡尔曼滤波器实现"""
    
    def __init__(self, transition_matrix: np.ndarray, 
                 observation_matrix: np.ndarray,
                 process_noise: np.ndarray,
                 observation_noise: np.ndarray,
                 initial_state: np.ndarray,
                 initial_covariance: np.ndarray):
        """
        初始化卡尔曼滤波器
        
        Args:
            transition_matrix: 状态转移矩阵 F
            observation_matrix: 观测矩阵 H
            process_noise: 过程噪声协方差 Q
            observation_noise: 观测噪声协方差 R
            initial_state: 初始状态估计
            initial_covariance: 初始状态协方差
        """
        self.F = transition_matrix
        self.H = observation_matrix
        self.Q = process_noise
        self.R = observation_noise
        self.x = initial_state
        self.P = initial_covariance
        
        # 历史记录
        self.state_history = [self.x.copy()]
        self.covariance_history = [self.P.copy()]
    
    def predict(self, control_input: np.ndarray = None, 
                control_matrix: np.ndarray = None):
        """
        预测步骤
        
        Args:
            control_input: 控制输入
            control_matrix: 控制矩阵 B
        """
        # 状态预测
        self.x = self.F @ self.x
        if control_input is not None and control_matrix is not None:
            self.x += control_matrix @ control_input
        
        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, observation: np.ndarray):
        """
        更新步骤
        
        Args:
            observation: 观测值
        """
        # 创新（残差）
        y = observation - self.H @ self.x
        
        # 创新协方差
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新
        self.x = self.x + K @ y
        
        # 协方差更新
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
        
        # 记录历史
        self.state_history.append(self.x.copy())
        self.covariance_history.append(self.P.copy())
    
    def filter_sequence(self, observations: List[np.ndarray]) -> List[np.ndarray]:
        """
        对观测序列进行滤波
        
        Returns:
            状态估计序列
        """
        states = []
        for obs in observations:
            self.predict()
            self.update(obs)
            states.append(self.x.copy())
        return states


class Particle:
    """粒子类"""
    
    def __init__(self, state: np.ndarray, weight: float = 1.0):
        self.state = state.copy()
        self.weight = weight


class ParticleFilter:
    """粒子滤波器实现"""
    
    def __init__(self, num_particles: int, 
                 state_transition_func: Callable,
                 observation_func: Callable,
                 process_noise_func: Callable,
                 observation_likelihood_func: Callable,
                 initial_state_sampler: Callable):
        """
        初始化粒子滤波器
        
        Args:
            num_particles: 粒子数量
            state_transition_func: 状态转移函数
            observation_func: 观测函数
            process_noise_func: 过程噪声生成函数
            observation_likelihood_func: 观测似然函数
            initial_state_sampler: 初始状态采样函数
        """
        self.num_particles = num_particles
        self.state_transition = state_transition_func
        self.observation_func = observation_func
        self.process_noise = process_noise_func
        self.observation_likelihood = observation_likelihood_func
        
        # 初始化粒子
        self.particles = []
        for _ in range(num_particles):
            state = initial_state_sampler()
            self.particles.append(Particle(state, 1.0 / num_particles))
    
    def predict(self):
        """预测步骤：根据动态模型移动粒子"""
        for particle in self.particles:
            # 状态转移
            particle.state = self.state_transition(particle.state)
            # 添加过程噪声
            particle.state += self.process_noise()
    
    def update(self, observation: np.ndarray):
        """更新步骤：根据观测更新权重"""
        total_weight = 0
        
        for particle in self.particles:
            # 计算观测似然
            likelihood = self.observation_likelihood(observation, particle.state)
            particle.weight *= likelihood
            total_weight += particle.weight
        
        # 权重归一化
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
    
    def resample(self):
        """重采样：避免粒子退化"""
        # 计算有效粒子数
        weights = [p.weight for p in self.particles]
        n_eff = 1.0 / sum(w**2 for w in weights)
        
        # 如果有效粒子数太少，进行重采样
        if n_eff < self.num_particles / 2:
            new_particles = []
            
            # 累积权重
            cumulative_weights = np.cumsum(weights)
            
            # 系统重采样
            r = random.uniform(0, 1/self.num_particles)
            i = 0
            for j in range(self.num_particles):
                u = r + j / self.num_particles
                while u > cumulative_weights[i]:
                    i += 1
                
                new_particle = Particle(self.particles[i].state.copy(), 
                                       1.0 / self.num_particles)
                new_particles.append(new_particle)
            
            self.particles = new_particles
    
    def estimate(self) -> np.ndarray:
        """估计当前状态（加权平均）"""
        total_weight = sum(p.weight for p in self.particles)
        if total_weight == 0:
            return np.zeros_like(self.particles[0].state)
        
        weighted_sum = np.zeros_like(self.particles[0].state)
        for particle in self.particles:
            weighted_sum += particle.weight * particle.state
        
        return weighted_sum / total_weight
    
    def filter_sequence(self, observations: List[np.ndarray]) -> List[np.ndarray]:
        """对观测序列进行滤波"""
        estimates = []
        
        for obs in observations:
            self.predict()
            self.update(obs)
            self.resample()
            estimates.append(self.estimate())
        
        return estimates


class DynamicBayesianNetwork:
    """动态贝叶斯网络"""
    
    def __init__(self, states: List[str], 
                 transition_probabilities: Dict[Tuple[str, str], float],
                 evidence_probabilities: Dict[Tuple[str, str], float]):
        """
        初始化动态贝叶斯网络
        
        Args:
            states: 状态变量
            transition_probabilities: 时间转移概率
            evidence_probabilities: 证据概率
        """
        self.states = states
        self.transition_prob = transition_probabilities
        self.evidence_prob = evidence_probabilities
        self.belief = {state: 1.0/len(states) for state in states}
    
    def forward_step(self, evidence: Optional[str] = None):
        """前向推理一步"""
        # 预测步骤
        new_belief = {}
        for curr_state in self.states:
            new_belief[curr_state] = 0
            for prev_state in self.states:
                new_belief[curr_state] += (self.belief[prev_state] *
                                         self.transition_prob.get((prev_state, curr_state), 0))
        
        # 更新步骤（如果有证据）
        if evidence:
            for state in self.states:
                new_belief[state] *= self.evidence_prob.get((state, evidence), 0)
            
            # 归一化
            total = sum(new_belief.values())
            if total > 0:
                for state in self.states:
                    new_belief[state] /= total
        
        self.belief = new_belief
    
    def filter_sequence(self, evidence_sequence: List[Optional[str]]) -> List[Dict[str, float]]:
        """对证据序列进行滤波"""
        beliefs = []
        for evidence in evidence_sequence:
            self.forward_step(evidence)
            beliefs.append(self.belief.copy())
        return beliefs


def demonstrate_hmm():
    """演示隐马尔可夫模型"""
    print("=== 隐马尔可夫模型演示 ===")
    
    # 天气预测HMM
    states = ['Sunny', 'Rainy']
    observations = ['Walk', 'Shop', 'Clean']
    
    # 状态转移概率
    transition_prob = {
        ('Sunny', 'Sunny'): 0.7,
        ('Sunny', 'Rainy'): 0.3,
        ('Rainy', 'Sunny'): 0.4,
        ('Rainy', 'Rainy'): 0.6
    }
    
    # 观测概率
    emission_prob = {
        ('Sunny', 'Walk'): 0.6,
        ('Sunny', 'Shop'): 0.3,
        ('Sunny', 'Clean'): 0.1,
        ('Rainy', 'Walk'): 0.1,
        ('Rainy', 'Shop'): 0.4,
        ('Rainy', 'Clean'): 0.5
    }
    
    # 初始概率
    initial_prob = {'Sunny': 0.6, 'Rainy': 0.4}
    
    hmm = HiddenMarkovModel(states, observations, transition_prob, emission_prob, initial_prob)
    
    # 观测序列
    obs_sequence = ['Walk', 'Shop', 'Clean']
    
    print(f"观测序列: {obs_sequence}")
    
    # 前向算法
    alpha, prob = hmm.forward_algorithm(obs_sequence)
    print(f"观测序列概率: {prob:.6f}")
    
    # 维特比算法
    best_path, best_prob = hmm.viterbi_algorithm(obs_sequence)
    print(f"最可能的天气序列: {best_path}")
    print(f"路径概率: {best_prob:.6f}")


def demonstrate_kalman_filter():
    """演示卡尔曼滤波"""
    print("\n=== 卡尔曼滤波演示 ===")
    
    # 一维运动模型：位置和速度
    dt = 1.0  # 时间步长
    
    # 状态转移矩阵（位置、速度）
    F = np.array([[1, dt],
                  [0, 1]])
    
    # 观测矩阵（只能观测位置）
    H = np.array([[1, 0]])
    
    # 过程噪声协方差
    Q = np.array([[0.1, 0],
                  [0, 0.1]])
    
    # 观测噪声协方差
    R = np.array([[1.0]])
    
    # 初始状态和协方差
    initial_state = np.array([0, 1])  # 位置0，速度1
    initial_covariance = np.array([[1, 0],
                                  [0, 1]])
    
    kf = KalmanFilter(F, H, Q, R, initial_state, initial_covariance)
    
    # 生成模拟观测数据
    true_positions = []
    observations = []
    true_pos = 0
    true_vel = 1
    
    for t in range(10):
        true_pos += true_vel * dt
        true_positions.append(true_pos)
        
        # 添加观测噪声
        obs = true_pos + np.random.normal(0, 1)
        observations.append(np.array([obs]))
    
    # 卡尔曼滤波
    estimated_states = kf.filter_sequence(observations)
    estimated_positions = [state[0] for state in estimated_states]
    
    print("时间\t真实位置\t观测位置\t估计位置")
    for t in range(len(true_positions)):
        print(f"{t+1}\t{true_positions[t]:.2f}\t\t{observations[t][0]:.2f}\t\t{estimated_positions[t]:.2f}")


def demonstrate_particle_filter():
    """演示粒子滤波"""
    print("\n=== 粒子滤波演示 ===")
    
    # 定义系统模型
    def state_transition(state):
        # 简单的随机游走
        return state + np.array([0.1, 0.1])
    
    def observation_func(state):
        # 观测函数：带噪声的位置观测
        return state + np.random.normal(0, 0.1, size=state.shape)
    
    def process_noise():
        return np.random.normal(0, 0.05, size=2)
    
    def observation_likelihood(observation, state):
        # 高斯似然
        diff = observation - state
        return np.exp(-0.5 * np.sum(diff**2) / 0.01)
    
    def initial_state_sampler():
        return np.random.normal(0, 1, size=2)
    
    # 初始化粒子滤波器
    pf = ParticleFilter(
        num_particles=100,
        state_transition_func=state_transition,
        observation_func=observation_func,
        process_noise_func=process_noise,
        observation_likelihood_func=observation_likelihood,
        initial_state_sampler=initial_state_sampler
    )
    
    # 生成模拟数据
    true_states = []
    observations = []
    true_state = np.array([0, 0])
    
    for t in range(10):
        true_state = state_transition(true_state) + process_noise()
        true_states.append(true_state.copy())
        
        obs = observation_func(true_state)
        observations.append(obs)
    
    # 粒子滤波
    estimated_states = pf.filter_sequence(observations)
    
    print("时间\t真实状态\t\t观测状态\t\t估计状态")
    for t in range(len(true_states)):
        print(f"{t+1}\t{true_states[t]}\t{observations[t]}\t{estimated_states[t]}")


def main():
    """主演示函数"""
    print("第14章：时序推理")
    print("实现了HMM、卡尔曼滤波、粒子滤波等时序推理算法")
    
    demonstrate_hmm()
    demonstrate_kalman_filter()
    demonstrate_particle_filter()
    
    print("\n=== 时序推理算法总结 ===")
    print("1. 隐马尔可夫模型：离散状态的时序建模")
    print("2. 卡尔曼滤波：线性高斯系统的最优滤波")
    print("3. 扩展卡尔曼滤波：非线性系统的近似滤波")
    print("4. 粒子滤波：非线性非高斯系统的近似推理")
    print("5. 动态贝叶斯网络：复杂依赖关系的时序建模")


if __name__ == "__main__":
    main() 