#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第15章：概率程序设计 (Probabilistic Programming)
=====================================

本章实现概率程序设计的核心概念和算法，包括：
1. 概率编程语言基础
2. 贝叶斯推理
3. 随机变分推理
4. 马尔可夫链蒙特卡罗方法
5. 概率图模型
6. 实际应用案例

作者：AI Assistant
日期：2024年12月
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import random
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ProbabilisticProgram:
    """概率程序基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.variables = {}
        self.constraints = []
        self.observations = {}
    
    def prior(self, var_name: str, distribution: stats.rv_continuous, **params):
        """定义先验分布"""
        self.variables[var_name] = {
            'type': 'prior',
            'distribution': distribution,
            'params': params
        }
    
    def observe(self, var_name: str, value: float):
        """观察变量值"""
        self.observations[var_name] = value
    
    def sample(self, var_name: str, n_samples: int = 1000):
        """从分布中采样"""
        if var_name in self.variables:
            var_info = self.variables[var_name]
            dist = var_info['distribution']
            params = var_info['params']
            return dist.rvs(size=n_samples, **params)
        return None

class BayesianInference:
    """贝叶斯推理引擎"""
    
    def __init__(self):
        self.models = {}
    
    def add_model(self, name: str, prior: Callable, likelihood: Callable):
        """添加贝叶斯模型"""
        self.models[name] = {
            'prior': prior,
            'likelihood': likelihood
        }
    
    def posterior_sampling(self, model_name: str, data: np.ndarray, 
                          n_samples: int = 1000, **params):
        """后验采样"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model = self.models[model_name]
        samples = []
        
        # 简单的Metropolis-Hastings采样
        current = params.get('initial_value', 0.5)
        
        for i in range(n_samples):
            # 提议新状态
            proposal = current + np.random.normal(0, 0.1)
            
            # 计算接受概率
            current_prob = model['prior'](current) * np.prod(model['likelihood'](current, data))
            proposal_prob = model['prior'](proposal) * np.prod(model['likelihood'](proposal, data))
            
            acceptance_prob = min(1, proposal_prob / max(current_prob, 1e-10))
            
            if np.random.random() < acceptance_prob:
                current = proposal
            
            samples.append(current)
        
        return np.array(samples)

class VariationalInference:
    """变分推理"""
    
    def __init__(self):
        self.variational_families = {}
    
    def add_variational_family(self, name: str, family: Callable):
        """添加变分家族"""
        self.variational_families[name] = family
    
    def optimize_elbo(self, target_log_prob: Callable, initial_params: Dict,
                     n_samples: int = 1000, n_iterations: int = 100):
        """优化ELBO (Evidence Lower Bound)"""
        params = initial_params.copy()
        elbo_history = []
        
        for iteration in range(n_iterations):
            # 从变分分布采样
            samples = self._sample_variational(params, n_samples)
            
            # 计算ELBO
            elbo = self._compute_elbo(samples, target_log_prob, params)
            elbo_history.append(elbo)
            
            # 简单梯度估计和更新
            grad = self._estimate_gradient(samples, target_log_prob, params)
            
            # 更新参数
            learning_rate = 0.01
            for key in params:
                params[key] += learning_rate * grad.get(key, 0)
        
        return params, elbo_history
    
    def _sample_variational(self, params: Dict, n_samples: int):
        """从变分分布采样"""
        # 假设高斯变分分布
        mean = params.get('mean', 0.0)
        std = params.get('std', 1.0)
        return np.random.normal(mean, std, n_samples)
    
    def _compute_elbo(self, samples: np.ndarray, target_log_prob: Callable, params: Dict):
        """计算ELBO"""
        # E_q[log p(x)] - E_q[log q(x)]
        target_log_probs = np.array([target_log_prob(x) for x in samples])
        variational_log_probs = stats.norm.logpdf(samples, params['mean'], params['std'])
        
        return np.mean(target_log_probs - variational_log_probs)
    
    def _estimate_gradient(self, samples: np.ndarray, target_log_prob: Callable, params: Dict):
        """估计梯度"""
        # 简化的梯度估计
        eps = 1e-5
        grad = {}
        
        for key in params:
            params_plus = params.copy()
            params_plus[key] += eps
            
            params_minus = params.copy()
            params_minus[key] -= eps
            
            elbo_plus = self._compute_elbo(samples, target_log_prob, params_plus)
            elbo_minus = self._compute_elbo(samples, target_log_prob, params_minus)
            
            grad[key] = (elbo_plus - elbo_minus) / (2 * eps)
        
        return grad

class MCMCSampler:
    """马尔可夫链蒙特卡罗采样器"""
    
    def __init__(self):
        self.chains = {}
    
    def metropolis_hastings(self, target_log_prob: Callable, initial_state: float,
                          n_samples: int = 1000, step_size: float = 0.1):
        """Metropolis-Hastings采样"""
        samples = []
        current_state = initial_state
        n_accepted = 0
        
        for i in range(n_samples):
            # 提议新状态
            proposal = current_state + np.random.normal(0, step_size)
            
            # 计算接受概率
            current_log_prob = target_log_prob(current_state)
            proposal_log_prob = target_log_prob(proposal)
            
            log_acceptance_prob = min(0, proposal_log_prob - current_log_prob)
            
            if np.log(np.random.random()) < log_acceptance_prob:
                current_state = proposal
                n_accepted += 1
            
            samples.append(current_state)
        
        acceptance_rate = n_accepted / n_samples
        return np.array(samples), acceptance_rate
    
    def hamiltonian_monte_carlo(self, target_log_prob: Callable, grad_log_prob: Callable,
                               initial_state: float, n_samples: int = 1000,
                               step_size: float = 0.01, n_leapfrog: int = 10):
        """哈密顿蒙特卡罗采样"""
        samples = []
        current_q = initial_state
        n_accepted = 0
        
        for i in range(n_samples):
            # 初始化动量
            current_p = np.random.normal(0, 1)
            
            # 存储当前状态
            q = current_q
            p = current_p
            
            # Leapfrog积分
            for _ in range(n_leapfrog):
                p = p + 0.5 * step_size * grad_log_prob(q)
                q = q + step_size * p
                p = p + 0.5 * step_size * grad_log_prob(q)
            
            # 计算接受概率
            current_energy = -target_log_prob(current_q) + 0.5 * current_p**2
            proposal_energy = -target_log_prob(q) + 0.5 * p**2
            
            if np.log(np.random.random()) < current_energy - proposal_energy:
                current_q = q
                n_accepted += 1
            
            samples.append(current_q)
        
        acceptance_rate = n_accepted / n_samples
        return np.array(samples), acceptance_rate

class ProbabilisticGraphicalModel:
    """概率图模型"""
    
    def __init__(self):
        self.variables = {}
        self.edges = []
        self.factors = {}
    
    def add_variable(self, name: str, domain: List):
        """添加变量"""
        self.variables[name] = {
            'domain': domain,
            'parents': [],
            'children': []
        }
    
    def add_edge(self, parent: str, child: str):
        """添加边"""
        self.edges.append((parent, child))
        self.variables[parent]['children'].append(child)
        self.variables[child]['parents'].append(parent)
    
    def add_factor(self, name: str, variables: List[str], factor_func: Callable):
        """添加因子"""
        self.factors[name] = {
            'variables': variables,
            'function': factor_func
        }
    
    def belief_propagation(self, evidence: Dict = None, max_iterations: int = 100):
        """置信传播算法"""
        if evidence is None:
            evidence = {}
        
        # 初始化消息
        messages = {}
        for parent, child in self.edges:
            messages[(parent, child)] = np.ones(len(self.variables[child]['domain']))
            messages[(child, parent)] = np.ones(len(self.variables[parent]['domain']))
        
        # 迭代更新消息
        for iteration in range(max_iterations):
            old_messages = messages.copy()
            
            for parent, child in self.edges:
                # 更新消息
                messages[(parent, child)] = self._compute_message(parent, child, messages, evidence)
                messages[(child, parent)] = self._compute_message(child, parent, messages, evidence)
            
            # 检查收敛
            if self._messages_converged(old_messages, messages):
                break
        
        # 计算边际概率
        marginals = {}
        for var in self.variables:
            marginals[var] = self._compute_marginal(var, messages, evidence)
        
        return marginals
    
    def _compute_message(self, from_var: str, to_var: str, messages: Dict, evidence: Dict):
        """计算消息"""
        # 简化的消息计算
        domain_size = len(self.variables[to_var]['domain'])
        return np.ones(domain_size) / domain_size
    
    def _compute_marginal(self, var: str, messages: Dict, evidence: Dict):
        """计算边际概率"""
        domain_size = len(self.variables[var]['domain'])
        return np.ones(domain_size) / domain_size
    
    def _messages_converged(self, old_messages: Dict, new_messages: Dict, tolerance: float = 1e-6):
        """检查消息是否收敛"""
        for key in old_messages:
            if np.sum(np.abs(old_messages[key] - new_messages[key])) > tolerance:
                return False
        return True

class ProbabilisticProgrammingApplications:
    """概率程序设计应用案例"""
    
    def __init__(self):
        self.models = {}
    
    def linear_regression_model(self, X: np.ndarray, y: np.ndarray):
        """贝叶斯线性回归"""
        # 先验分布
        def prior_slope(beta):
            return stats.norm.pdf(beta, 0, 1)
        
        def prior_intercept(alpha):
            return stats.norm.pdf(alpha, 0, 1)
        
        def prior_noise(sigma):
            return stats.gamma.pdf(sigma, 1, 1)
        
        # 似然函数
        def likelihood(params, X, y):
            alpha, beta, sigma = params
            predictions = alpha + beta * X
            return stats.norm.pdf(y, predictions, sigma)
        
        # 后验采样
        def log_posterior(params):
            alpha, beta, sigma = params
            if sigma <= 0:
                return -np.inf
            
            log_prior = (np.log(prior_slope(beta)) + 
                        np.log(prior_intercept(alpha)) + 
                        np.log(prior_noise(sigma)))
            
            log_likelihood = np.sum(np.log(likelihood(params, X, y)))
            
            return log_prior + log_likelihood
        
        # MCMC采样
        mcmc = MCMCSampler()
        samples = []
        
        for _ in range(1000):
            # 采样每个参数
            alpha_sample = np.random.normal(0, 1)
            beta_sample = np.random.normal(0, 1)
            sigma_sample = np.random.gamma(1, 1)
            
            sample = [alpha_sample, beta_sample, sigma_sample]
            samples.append(sample)
        
        return np.array(samples)
    
    def mixture_model(self, data: np.ndarray, n_components: int = 2):
        """混合模型"""
        # 初始化参数
        means = np.random.randn(n_components)
        stds = np.ones(n_components)
        weights = np.ones(n_components) / n_components
        
        # EM算法
        for iteration in range(100):
            # E步：计算责任
            responsibilities = np.zeros((len(data), n_components))
            
            for i, x in enumerate(data):
                for k in range(n_components):
                    responsibilities[i, k] = (weights[k] * 
                                            stats.norm.pdf(x, means[k], stds[k]))
                
                # 归一化
                responsibilities[i] /= np.sum(responsibilities[i])
            
            # M步：更新参数
            for k in range(n_components):
                # 更新权重
                weights[k] = np.mean(responsibilities[:, k])
                
                # 更新均值
                means[k] = np.sum(responsibilities[:, k] * data) / np.sum(responsibilities[:, k])
                
                # 更新标准差
                stds[k] = np.sqrt(np.sum(responsibilities[:, k] * (data - means[k])**2) / 
                                 np.sum(responsibilities[:, k]))
        
        return means, stds, weights, responsibilities
    
    def hierarchical_model(self, groups: List[np.ndarray]):
        """分层模型"""
        # 超参数
        mu_prior = 0
        sigma_prior = 1
        
        # 组级参数
        group_means = []
        group_stds = []
        
        for group_data in groups:
            # 组级先验
            group_mean = np.random.normal(mu_prior, sigma_prior)
            group_std = np.random.gamma(1, 1)
            
            # 数据似然
            n_samples = len(group_data)
            sample_mean = np.mean(group_data)
            sample_std = np.std(group_data)
            
            # 更新后验
            posterior_precision = 1/sigma_prior**2 + n_samples/group_std**2
            posterior_mean = ((mu_prior/sigma_prior**2 + 
                             n_samples*sample_mean/group_std**2) / 
                            posterior_precision)
            posterior_std = 1/np.sqrt(posterior_precision)
            
            group_means.append(np.random.normal(posterior_mean, posterior_std))
            group_stds.append(group_std)
        
        return group_means, group_stds

def demonstrate_probabilistic_programming():
    """演示概率程序设计"""
    print("=== 概率程序设计演示 ===\n")
    
    # 1. 基本概率程序
    print("1. 基本概率程序")
    program = ProbabilisticProgram("simple_model")
    program.prior("theta", stats.beta, a=2, b=2)
    program.observe("data", 0.7)
    
    samples = program.sample("theta", 1000)
    print(f"先验采样均值: {np.mean(samples):.3f}")
    print(f"先验采样标准差: {np.std(samples):.3f}")
    
    # 2. 贝叶斯推理
    print("\n2. 贝叶斯推理")
    inference = BayesianInference()
    
    # 定义模型
    def coin_prior(theta):
        return stats.beta.pdf(theta, 1, 1)
    
    def coin_likelihood(theta, data):
        return stats.bernoulli.pmf(data, theta)
    
    inference.add_model("coin", coin_prior, coin_likelihood)
    
    # 观察数据
    coin_data = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1])
    posterior_samples = inference.posterior_sampling("coin", coin_data, 1000)
    
    print(f"后验采样均值: {np.mean(posterior_samples):.3f}")
    print(f"后验采样标准差: {np.std(posterior_samples):.3f}")
    
    # 3. 变分推理
    print("\n3. 变分推理")
    vi = VariationalInference()
    
    def target_log_prob(x):
        return stats.norm.logpdf(x, 2, 1)
    
    initial_params = {'mean': 0.0, 'std': 1.0}
    opt_params, elbo_history = vi.optimize_elbo(target_log_prob, initial_params, 
                                               n_iterations=50)
    
    print(f"优化后的均值: {opt_params['mean']:.3f}")
    print(f"优化后的标准差: {opt_params['std']:.3f}")
    
    # 4. MCMC采样
    print("\n4. MCMC采样")
    mcmc = MCMCSampler()
    
    def normal_log_prob(x):
        return stats.norm.logpdf(x, 1, 0.5)
    
    samples, acceptance_rate = mcmc.metropolis_hastings(normal_log_prob, 0.0, 1000)
    
    print(f"MCMC采样均值: {np.mean(samples):.3f}")
    print(f"MCMC采样标准差: {np.std(samples):.3f}")
    print(f"接受率: {acceptance_rate:.3f}")
    
    # 5. 应用案例
    print("\n5. 应用案例")
    apps = ProbabilisticProgrammingApplications()
    
    # 生成数据
    X = np.linspace(0, 10, 50)
    y = 2 * X + 1 + np.random.normal(0, 0.5, 50)
    
    # 贝叶斯线性回归
    regression_samples = apps.linear_regression_model(X, y)
    print(f"回归系数均值: {np.mean(regression_samples[:, 1]):.3f}")
    
    # 混合模型
    mixture_data = np.concatenate([np.random.normal(0, 1, 100), 
                                  np.random.normal(3, 1, 100)])
    means, stds, weights, resp = apps.mixture_model(mixture_data, 2)
    print(f"混合模型均值: {means}")
    print(f"混合模型权重: {weights}")
    
    # 创建可视化
    create_visualizations(samples, posterior_samples, elbo_history, 
                         mcmc_samples=samples, mixture_data=mixture_data, 
                         means=means, stds=stds, weights=weights)

def create_visualizations(prior_samples, posterior_samples, elbo_history, 
                         mcmc_samples, mixture_data, means, stds, weights):
    """创建可视化图表"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 先验vs后验分布
    axes[0, 0].hist(prior_samples, bins=50, alpha=0.7, label='先验', density=True)
    axes[0, 0].hist(posterior_samples, bins=50, alpha=0.7, label='后验', density=True)
    axes[0, 0].set_xlabel('θ')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('先验vs后验分布')
    axes[0, 0].legend()
    
    # 2. ELBO收敛
    axes[0, 1].plot(elbo_history)
    axes[0, 1].set_xlabel('迭代次数')
    axes[0, 1].set_ylabel('ELBO')
    axes[0, 1].set_title('变分推理收敛')
    
    # 3. MCMC轨迹
    axes[0, 2].plot(mcmc_samples[:500])
    axes[0, 2].set_xlabel('迭代次数')
    axes[0, 2].set_ylabel('采样值')
    axes[0, 2].set_title('MCMC采样轨迹')
    
    # 4. MCMC自相关
    from statsmodels.tsa.stattools import acf
    try:
        autocorr = acf(mcmc_samples, nlags=50)
        axes[1, 0].plot(autocorr)
        axes[1, 0].set_xlabel('滞后')
        axes[1, 0].set_ylabel('自相关')
        axes[1, 0].set_title('MCMC自相关函数')
    except:
        axes[1, 0].text(0.5, 0.5, '自相关计算失败', ha='center', va='center')
    
    # 5. 混合模型拟合
    axes[1, 1].hist(mixture_data, bins=50, alpha=0.7, density=True, label='数据')
    
    x = np.linspace(mixture_data.min(), mixture_data.max(), 100)
    mixture_pdf = np.zeros_like(x)
    for i in range(len(means)):
        component_pdf = weights[i] * stats.norm.pdf(x, means[i], stds[i])
        mixture_pdf += component_pdf
        axes[1, 1].plot(x, component_pdf, '--', alpha=0.7, label=f'组分{i+1}')
    
    axes[1, 1].plot(x, mixture_pdf, 'r-', linewidth=2, label='混合分布')
    axes[1, 1].set_xlabel('值')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].set_title('混合模型拟合')
    axes[1, 1].legend()
    
    # 6. 参数分布
    axes[1, 2].boxplot([prior_samples, posterior_samples, mcmc_samples], 
                      labels=['先验', '后验', 'MCMC'])
    axes[1, 2].set_ylabel('参数值')
    axes[1, 2].set_title('参数分布比较')
    
    plt.tight_layout()
    plt.savefig('probabilistic_programming_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    demonstrate_probabilistic_programming()
    print("\n=== 概率程序设计演示完成 ===") 