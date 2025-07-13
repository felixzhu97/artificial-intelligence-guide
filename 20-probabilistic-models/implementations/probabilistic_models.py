"""
第20章：概率模型学习
Learning Probabilistic Models

本章实现统计学习中的概率模型，包括：
1. 期望最大化算法（EM Algorithm）
2. 隐马尔可夫模型（HMM）
3. 高斯混合模型（GMM）
4. 概率主成分分析（PPCA）
5. 变分推断
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp
from sklearn.datasets import make_blobs
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GaussianMixtureModel:
    """高斯混合模型"""
    
    def __init__(self, n_components: int = 2, max_iter: int = 100, tol: float = 1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X: np.ndarray):
        """使用EM算法训练GMM"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        # 存储对数似然
        self.log_likelihood_history = []
        
        for iteration in range(self.max_iter):
            # E步：计算后验概率
            responsibilities = self._e_step(X)
            
            # M步：更新参数
            self._m_step(X, responsibilities)
            
            # 计算对数似然
            log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history.append(log_likelihood)
            
            # 检查收敛
            if iteration > 0:
                if abs(log_likelihood - self.log_likelihood_history[-2]) < self.tol:
                    print(f"EM算法在第{iteration+1}次迭代收敛")
                    break
        
        return self
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """E步：计算后验概率"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # 计算每个组件的概率密度
            try:
                responsibilities[:, k] = self.weights[k] * stats.multivariate_normal.pdf(
                    X, self.means[k], self.covariances[k]
                )
            except:
                # 处理奇异协方差矩阵
                responsibilities[:, k] = self.weights[k] * stats.multivariate_normal.pdf(
                    X, self.means[k], self.covariances[k] + 1e-6 * np.eye(X.shape[1])
                )
        
        # 归一化
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        responsibilities = np.nan_to_num(responsibilities)
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """M步：更新参数"""
        n_samples = X.shape[0]
        
        # 计算有效样本数
        Nk = responsibilities.sum(axis=0)
        
        # 更新权重
        self.weights = Nk / n_samples
        
        # 更新均值
        for k in range(self.n_components):
            self.means[k] = (responsibilities[:, k:k+1].T @ X) / Nk[k]
        
        # 更新协方差
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = (responsibilities[:, k:k+1] * diff).T @ diff / Nk[k]
            
            # 添加正则化以避免奇异矩阵
            self.covariances[k] += 1e-6 * np.eye(X.shape[1])
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """计算对数似然"""
        n_samples = X.shape[0]
        log_likelihood = 0
        
        for i in range(n_samples):
            likelihood = 0
            for k in range(self.n_components):
                try:
                    likelihood += self.weights[k] * stats.multivariate_normal.pdf(
                        X[i], self.means[k], self.covariances[k]
                    )
                except:
                    likelihood += self.weights[k] * stats.multivariate_normal.pdf(
                        X[i], self.means[k], self.covariances[k] + 1e-6 * np.eye(X.shape[1])
                    )
            log_likelihood += np.log(likelihood + 1e-10)
        
        return log_likelihood
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测样本所属的组件"""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测样本属于各组件的概率"""
        return self._e_step(X)

class HiddenMarkovModel:
    """隐马尔可夫模型"""
    
    def __init__(self, n_states: int, n_observations: int):
        self.n_states = n_states
        self.n_observations = n_observations
        
        # 初始化参数
        self.initial_probs = np.ones(n_states) / n_states
        self.transition_probs = np.ones((n_states, n_states)) / n_states
        self.emission_probs = np.ones((n_states, n_observations)) / n_observations
    
    def fit(self, observations_sequences: List[List[int]], max_iter: int = 100, tol: float = 1e-6):
        """使用Baum-Welch算法训练HMM"""
        self.log_likelihood_history = []
        
        for iteration in range(max_iter):
            # 存储所有序列的前向后向概率
            all_alpha = []
            all_beta = []
            all_gamma = []
            all_xi = []
            
            total_log_likelihood = 0
            
            for obs_seq in observations_sequences:
                # 前向算法
                alpha = self._forward(obs_seq)
                all_alpha.append(alpha)
                
                # 后向算法
                beta = self._backward(obs_seq)
                all_beta.append(beta)
                
                # 计算gamma和xi
                gamma = self._compute_gamma(alpha, beta)
                xi = self._compute_xi(obs_seq, alpha, beta)
                all_gamma.append(gamma)
                all_xi.append(xi)
                
                # 计算对数似然
                total_log_likelihood += self._compute_log_likelihood(alpha)
            
            self.log_likelihood_history.append(total_log_likelihood)
            
            # 更新参数
            self._update_parameters(observations_sequences, all_gamma, all_xi)
            
            # 检查收敛
            if iteration > 0:
                if abs(total_log_likelihood - self.log_likelihood_history[-2]) < tol:
                    print(f"Baum-Welch算法在第{iteration+1}次迭代收敛")
                    break
        
        return self
    
    def _forward(self, obs_seq: List[int]) -> np.ndarray:
        """前向算法"""
        T = len(obs_seq)
        alpha = np.zeros((T, self.n_states))
        
        # 初始化
        alpha[0] = self.initial_probs * self.emission_probs[:, obs_seq[0]]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_probs[:, j]) * \
                             self.emission_probs[j, obs_seq[t]]
        
        return alpha
    
    def _backward(self, obs_seq: List[int]) -> np.ndarray:
        """后向算法"""
        T = len(obs_seq)
        beta = np.zeros((T, self.n_states))
        
        # 初始化
        beta[T-1] = 1
        
        # 递推
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transition_probs[i] * 
                                  self.emission_probs[:, obs_seq[t+1]] * beta[t+1])
        
        return beta
    
    def _compute_gamma(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """计算gamma（状态后验概率）"""
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma
    
    def _compute_xi(self, obs_seq: List[int], alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """计算xi（状态转移后验概率）"""
        T = len(obs_seq)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T-1):
            denominator = 0
            for i in range(self.n_states):
                for j in range(self.n_states):
                    denominator += alpha[t, i] * self.transition_probs[i, j] * \
                                 self.emission_probs[j, obs_seq[t+1]] * beta[t+1, j]
            
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.transition_probs[i, j] * 
                                  self.emission_probs[j, obs_seq[t+1]] * beta[t+1, j]) / denominator
        
        return xi
    
    def _update_parameters(self, observations_sequences: List[List[int]], 
                          all_gamma: List[np.ndarray], all_xi: List[np.ndarray]):
        """更新HMM参数"""
        # 更新初始概率
        self.initial_probs = np.mean([gamma[0] for gamma in all_gamma], axis=0)
        
        # 更新转移概率
        xi_sum = np.sum([xi.sum(axis=0) for xi in all_xi], axis=0)
        gamma_sum = np.sum([gamma[:-1].sum(axis=0) for gamma in all_gamma], axis=0)
        
        for i in range(self.n_states):
            self.transition_probs[i] = xi_sum[i] / gamma_sum[i]
        
        # 更新发射概率
        for j in range(self.n_states):
            for k in range(self.n_observations):
                numerator = 0
                denominator = 0
                
                for seq_idx, obs_seq in enumerate(observations_sequences):
                    gamma = all_gamma[seq_idx]
                    for t in range(len(obs_seq)):
                        if obs_seq[t] == k:
                            numerator += gamma[t, j]
                        denominator += gamma[t, j]
                
                self.emission_probs[j, k] = numerator / denominator
    
    def _compute_log_likelihood(self, alpha: np.ndarray) -> float:
        """计算对数似然"""
        return np.log(alpha[-1].sum())
    
    def viterbi(self, obs_seq: List[int]) -> List[int]:
        """Viterbi算法：寻找最可能的状态序列"""
        T = len(obs_seq)
        
        # 初始化
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0] = self.initial_probs * self.emission_probs[:, obs_seq[0]]
        
        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                probs = delta[t-1] * self.transition_probs[:, j]
                delta[t, j] = np.max(probs) * self.emission_probs[j, obs_seq[t]]
                psi[t, j] = np.argmax(probs)
        
        # 回溯
        path = []
        last_state = np.argmax(delta[T-1])
        path.append(last_state)
        
        for t in range(T-2, -1, -1):
            last_state = psi[t+1, last_state]
            path.append(last_state)
        
        return path[::-1]

class ProbabilisticPCA:
    """概率主成分分析"""
    
    def __init__(self, n_components: int, max_iter: int = 100, tol: float = 1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X: np.ndarray):
        """使用EM算法训练PPCA"""
        n_samples, n_features = X.shape
        
        # 数据中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 初始化参数
        self.W = np.random.randn(n_features, self.n_components)
        self.sigma2 = 1.0
        
        self.log_likelihood_history = []
        
        for iteration in range(self.max_iter):
            # E步
            M = self.W.T @ self.W + self.sigma2 * np.eye(self.n_components)
            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                # 处理奇异矩阵，使用伪逆
                M_inv = np.linalg.pinv(M)
            
            # 计算期望
            Ez = M_inv @ self.W.T @ X_centered.T  # (n_components, n_samples)
            Ezz = self.sigma2 * M_inv + Ez @ Ez.T / n_samples
            
            # M步
            try:
                self.W = X_centered.T @ Ez.T @ np.linalg.inv(Ezz)
            except np.linalg.LinAlgError:
                # 处理奇异矩阵，使用伪逆
                self.W = X_centered.T @ Ez.T @ np.linalg.pinv(Ezz)
            
            residual = X_centered - Ez.T @ self.W.T
            self.sigma2 = np.trace(residual.T @ residual) / (n_samples * n_features)
            
            # 计算对数似然
            log_likelihood = self._compute_log_likelihood(X_centered)
            self.log_likelihood_history.append(log_likelihood)
            
            # 检查收敛
            if iteration > 0:
                if abs(log_likelihood - self.log_likelihood_history[-2]) < self.tol:
                    print(f"PPCA在第{iteration+1}次迭代收敛")
                    break
        
        return self
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """计算对数似然"""
        n_samples, n_features = X.shape
        
        C = self.W @ self.W.T + self.sigma2 * np.eye(n_features)
        
        try:
            L = np.linalg.cholesky(C)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            inv_C_X = np.linalg.solve(L, X.T)
            quadratic = np.sum(inv_C_X ** 2)
            
            log_likelihood = -0.5 * (n_samples * log_det + quadratic + 
                                   n_samples * n_features * np.log(2 * np.pi))
        except:
            # 数值稳定性问题时的替代计算
            log_likelihood = -np.inf
        
        return log_likelihood
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """将数据投影到潜在空间"""
        X_centered = X - self.mean
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.n_components)
        M_inv = np.linalg.inv(M)
        
        return (M_inv @ self.W.T @ X_centered.T).T
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """从潜在空间重构数据"""
        return Z @ self.W.T + self.mean

class VariationalInference:
    """变分推断"""
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
    
    def fit_variational_gmm(self, X: np.ndarray, max_iter: int = 100, tol: float = 1e-6):
        """变分贝叶斯高斯混合模型"""
        n_samples, n_features = X.shape
        
        # 初始化变分参数
        self.alpha = np.ones(self.n_components)  # Dirichlet参数
        self.beta = np.ones(self.n_components)   # Wishart参数
        self.m = np.random.randn(self.n_components, n_features)  # 均值参数
        self.W = np.array([np.eye(n_features) for _ in range(self.n_components)])  # 精度矩阵
        self.nu = n_features  # 自由度
        
        self.lower_bound_history = []
        
        for iteration in range(max_iter):
            # 变分E步
            responsibilities = self._variational_e_step(X)
            
            # 变分M步
            self._variational_m_step(X, responsibilities)
            
            # 计算变分下界
            lower_bound = self._compute_lower_bound(X, responsibilities)
            self.lower_bound_history.append(lower_bound)
            
            # 检查收敛
            if iteration > 0:
                if abs(lower_bound - self.lower_bound_history[-2]) < tol:
                    print(f"变分推断在第{iteration+1}次迭代收敛")
                    break
        
        return self
    
    def _variational_e_step(self, X: np.ndarray) -> np.ndarray:
        """变分E步"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # 计算期望对数权重
            ln_pi = self._expected_log_pi(k)
            
            # 计算期望对数似然
            ln_likelihood = self._expected_log_likelihood(X, k)
            
            responsibilities[:, k] = ln_pi + ln_likelihood
        
        # 归一化
        responsibilities = np.exp(responsibilities - logsumexp(responsibilities, axis=1, keepdims=True))
        
        return responsibilities
    
    def _variational_m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """变分M步"""
        n_samples = X.shape[0]
        
        # 更新参数
        Nk = responsibilities.sum(axis=0)
        
        # 更新Dirichlet参数
        self.alpha = self.alpha + Nk
        
        # 更新高斯参数
        for k in range(self.n_components):
            # 更新均值
            xk = (responsibilities[:, k:k+1] * X).sum(axis=0) / Nk[k]
            self.m[k] = (self.beta[k] * self.m[k] + Nk[k] * xk) / (self.beta[k] + Nk[k])
            
            # 更新精度矩阵
            diff = X - xk
            Sk = (responsibilities[:, k:k+1] * diff).T @ diff / Nk[k]
            
            self.W[k] = np.linalg.inv(np.linalg.inv(self.W[k]) + Nk[k] * Sk + 
                                    (self.beta[k] * Nk[k] / (self.beta[k] + Nk[k])) * 
                                    np.outer(xk - self.m[k], xk - self.m[k]))
            
            # 更新其他参数
            self.beta[k] = self.beta[k] + Nk[k]
            self.nu = self.nu + Nk[k]
    
    def _expected_log_pi(self, k: int) -> float:
        """期望对数权重"""
        from scipy.special import digamma
        return digamma(self.alpha[k]) - digamma(self.alpha.sum())
    
    def _expected_log_likelihood(self, X: np.ndarray, k: int) -> np.ndarray:
        """期望对数似然"""
        n_features = X.shape[1]
        
        # 计算Wishart分布的期望
        expected_log_det = np.sum([stats.gamma.logpdf(0.5 * (self.nu + 1 - i), 0.5, scale=2) 
                                 for i in range(1, n_features + 1)])
        expected_log_det += n_features * np.log(2) + np.log(np.linalg.det(self.W[k]))
        
        # 计算二次型
        diff = X - self.m[k]
        quadratic = np.sum((diff @ self.W[k]) * diff, axis=1)
        
        return 0.5 * expected_log_det - 0.5 * n_features / self.beta[k] - 0.5 * self.nu * quadratic
    
    def _compute_lower_bound(self, X: np.ndarray, responsibilities: np.ndarray) -> float:
        """计算变分下界"""
        # 简化的下界计算
        return np.sum(responsibilities * np.log(responsibilities + 1e-10))

def demonstrate_gaussian_mixture_model():
    """演示高斯混合模型"""
    print("=" * 50)
    print("高斯混合模型演示")
    print("=" * 50)
    
    # 生成测试数据
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, 
                          random_state=42, cluster_std=1.5)
    
    # 训练GMM
    gmm = GaussianMixtureModel(n_components=3, max_iter=50)
    gmm.fit(X)
    
    # 预测
    y_pred = gmm.predict(X)
    probabilities = gmm.predict_proba(X)
    
    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 原始数据
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('原始数据（真实类别）')
    axes[0, 0].set_xlabel('特征1')
    axes[0, 0].set_ylabel('特征2')
    
    # GMM聚类结果
    axes[0, 1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    
    # 绘制高斯分布轮廓
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    for k in range(gmm.n_components):
        rv = stats.multivariate_normal(gmm.means[k], gmm.covariances[k])
        axes[0, 1].contour(xx, yy, rv.pdf(np.dstack((xx, yy))), 
                          levels=3, colors=['red', 'blue', 'green'][k], alpha=0.6)
    
    axes[0, 1].set_title('GMM聚类结果')
    axes[0, 1].set_xlabel('特征1')
    axes[0, 1].set_ylabel('特征2')
    
    # 对数似然收敛过程
    axes[1, 0].plot(gmm.log_likelihood_history)
    axes[1, 0].set_title('对数似然收敛过程')
    axes[1, 0].set_xlabel('迭代次数')
    axes[1, 0].set_ylabel('对数似然')
    axes[1, 0].grid(True)
    
    # 概率分布热图
    im = axes[1, 1].imshow(probabilities, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('样本属于各组件的概率')
    axes[1, 1].set_xlabel('组件')
    axes[1, 1].set_ylabel('样本')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    print(f"GMM参数：")
    print(f"权重: {gmm.weights}")
    print(f"均值: {gmm.means}")
    print(f"协方差矩阵形状: {[cov.shape for cov in gmm.covariances]}")

def demonstrate_hidden_markov_model():
    """演示隐马尔可夫模型"""
    print("=" * 50)
    print("隐马尔可夫模型演示")
    print("=" * 50)
    
    # 创建一个简单的HMM（天气模型）
    # 状态：0=晴天，1=雨天
    # 观测：0=散步，1=购物，2=清洁
    
    # 生成训练数据
    np.random.seed(42)
    
    # 真实参数
    true_initial = np.array([0.6, 0.4])
    true_transition = np.array([[0.7, 0.3], [0.4, 0.6]])
    true_emission = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    
    # 生成观测序列
    def generate_sequence(length=50):
        states = []
        observations = []
        
        # 初始状态
        state = np.random.choice(2, p=true_initial)
        states.append(state)
        
        for _ in range(length):
            # 生成观测
            obs = np.random.choice(3, p=true_emission[state])
            observations.append(obs)
            
            # 状态转移
            state = np.random.choice(2, p=true_transition[state])
            states.append(state)
        
        return states[:-1], observations
    
    # 生成多个序列
    sequences = []
    true_sequences = []
    
    for _ in range(5):
        states, obs = generate_sequence(30)
        sequences.append(obs)
        true_sequences.append(states)
    
    # 训练HMM
    hmm = HiddenMarkovModel(n_states=2, n_observations=3)
    hmm.fit(sequences, max_iter=50)
    
    # 测试Viterbi算法
    test_sequence = sequences[0]
    predicted_states = hmm.viterbi(test_sequence)
    true_states = true_sequences[0]
    
    print(f"学习到的参数：")
    print(f"初始概率: {hmm.initial_probs}")
    print(f"转移概率:\n{hmm.transition_probs}")
    print(f"发射概率:\n{hmm.emission_probs}")
    
    print(f"\n真实参数：")
    print(f"初始概率: {true_initial}")
    print(f"转移概率:\n{true_transition}")
    print(f"发射概率:\n{true_emission}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 对数似然收敛
    axes[0, 0].plot(hmm.log_likelihood_history)
    axes[0, 0].set_title('对数似然收敛过程')
    axes[0, 0].set_xlabel('迭代次数')
    axes[0, 0].set_ylabel('对数似然')
    axes[0, 0].grid(True)
    
    # 状态预测比较
    t = range(len(test_sequence))
    axes[0, 1].plot(t, true_states, 'o-', label='真实状态', alpha=0.7)
    axes[0, 1].plot(t, predicted_states, 's-', label='预测状态', alpha=0.7)
    axes[0, 1].set_title('状态预测比较')
    axes[0, 1].set_xlabel('时间')
    axes[0, 1].set_ylabel('状态')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 转移概率矩阵
    im1 = axes[1, 0].imshow(hmm.transition_probs, cmap='Blues')
    axes[1, 0].set_title('学习到的转移概率')
    axes[1, 0].set_xlabel('目标状态')
    axes[1, 0].set_ylabel('源状态')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['晴天', '雨天'])
    axes[1, 0].set_yticklabels(['晴天', '雨天'])
    plt.colorbar(im1, ax=axes[1, 0])
    
    # 发射概率矩阵
    im2 = axes[1, 1].imshow(hmm.emission_probs, cmap='Reds')
    axes[1, 1].set_title('学习到的发射概率')
    axes[1, 1].set_xlabel('观测')
    axes[1, 1].set_ylabel('状态')
    axes[1, 1].set_xticks([0, 1, 2])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['散步', '购物', '清洁'])
    axes[1, 1].set_yticklabels(['晴天', '雨天'])
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    # 计算预测准确率
    accuracy = np.mean(np.array(predicted_states) == np.array(true_states))
    print(f"\n状态预测准确率: {accuracy:.2%}")

def demonstrate_probabilistic_pca():
    """演示概率主成分分析"""
    print("=" * 50)
    print("概率主成分分析演示")
    print("=" * 50)
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 200
    
    # 生成低维数据
    true_latent = np.random.randn(n_samples, 2)
    
    # 真实的线性变换
    true_W = np.random.randn(5, 2)
    
    # 生成观测数据
    noise = 0.1 * np.random.randn(n_samples, 5)
    X = true_latent @ true_W.T + noise
    
    # 训练PPCA
    ppca = ProbabilisticPCA(n_components=2, max_iter=50)
    ppca.fit(X)
    
    # 变换数据
    X_transformed = ppca.transform(X)
    X_reconstructed = ppca.inverse_transform(X_transformed)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 对数似然收敛
    axes[0, 0].plot(ppca.log_likelihood_history)
    axes[0, 0].set_title('对数似然收敛过程')
    axes[0, 0].set_xlabel('迭代次数')
    axes[0, 0].set_ylabel('对数似然')
    axes[0, 0].grid(True)
    
    # 潜在空间可视化
    axes[0, 1].scatter(true_latent[:, 0], true_latent[:, 1], 
                      alpha=0.7, label='真实潜在变量')
    axes[0, 1].scatter(X_transformed[:, 0], X_transformed[:, 1], 
                      alpha=0.7, label='PPCA变换结果')
    axes[0, 1].set_title('潜在空间比较')
    axes[0, 1].set_xlabel('潜在维度1')
    axes[0, 1].set_ylabel('潜在维度2')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 重构误差
    reconstruction_error = np.mean((X - X_reconstructed) ** 2, axis=1)
    axes[1, 0].hist(reconstruction_error, bins=30, alpha=0.7)
    axes[1, 0].set_title('重构误差分布')
    axes[1, 0].set_xlabel('重构误差')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].grid(True)
    
    # 原始数据 vs 重构数据（选择前两个特征）
    axes[1, 1].scatter(X[:, 0], X[:, 1], alpha=0.7, label='原始数据')
    axes[1, 1].scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], 
                      alpha=0.7, label='重构数据')
    axes[1, 1].set_title('原始数据 vs 重构数据')
    axes[1, 1].set_xlabel('特征1')
    axes[1, 1].set_ylabel('特征2')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"PPCA参数：")
    print(f"潜在维度: {ppca.n_components}")
    print(f"噪声方差: {ppca.sigma2:.4f}")
    print(f"变换矩阵形状: {ppca.W.shape}")
    print(f"平均重构误差: {np.mean(reconstruction_error):.4f}")

def demonstrate_variational_inference():
    """演示变分推断"""
    print("=" * 50)
    print("变分推断演示")
    print("=" * 50)
    
    # 生成测试数据
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=2, n_features=2, 
                     random_state=42, cluster_std=1.0)
    
    # 变分贝叶斯GMM
    vb_gmm = VariationalInference(n_components=3)
    vb_gmm.fit_variational_gmm(X, max_iter=30)
    
    # 比较普通GMM
    regular_gmm = GaussianMixtureModel(n_components=3, max_iter=30)
    regular_gmm.fit(X)
    
    # 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 变分下界收敛
    axes[0, 0].plot(vb_gmm.lower_bound_history)
    axes[0, 0].set_title('变分下界收敛过程')
    axes[0, 0].set_xlabel('迭代次数')
    axes[0, 0].set_ylabel('变分下界')
    axes[0, 0].grid(True)
    
    # 普通GMM对数似然收敛
    axes[0, 1].plot(regular_gmm.log_likelihood_history)
    axes[0, 1].set_title('普通GMM对数似然收敛')
    axes[0, 1].set_xlabel('迭代次数')
    axes[0, 1].set_ylabel('对数似然')
    axes[0, 1].grid(True)
    
    # 数据分布
    axes[1, 0].scatter(X[:, 0], X[:, 1], alpha=0.7)
    axes[1, 0].set_title('原始数据分布')
    axes[1, 0].set_xlabel('特征1')
    axes[1, 0].set_ylabel('特征2')
    axes[1, 0].grid(True)
    
    # 参数比较
    param_comparison = {
        '变分贝叶斯': {'alpha': vb_gmm.alpha, 'beta': vb_gmm.beta},
        '普通GMM': {'weights': regular_gmm.weights}
    }
    
    axes[1, 1].text(0.1, 0.9, '变分贝叶斯参数:', transform=axes[1, 1].transAxes, 
                   fontsize=12, weight='bold')
    axes[1, 1].text(0.1, 0.8, f'α: {vb_gmm.alpha}', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'β: {vb_gmm.beta}', transform=axes[1, 1].transAxes)
    
    axes[1, 1].text(0.1, 0.5, '普通GMM参数:', transform=axes[1, 1].transAxes, 
                   fontsize=12, weight='bold')
    axes[1, 1].text(0.1, 0.4, f'权重: {regular_gmm.weights}', transform=axes[1, 1].transAxes)
    
    axes[1, 1].set_title('参数比较')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("变分推断的优势：")
    print("1. 提供参数的不确定性估计")
    print("2. 自动模型选择（有效组件数）")
    print("3. 防止过拟合")
    print("4. 贝叶斯框架下的理论保证")

def main():
    """主函数"""
    print("第20章：概率模型学习")
    print("Learning Probabilistic Models")
    print("=" * 60)
    
    # 演示各个模型
    demonstrate_gaussian_mixture_model()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_hidden_markov_model()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_probabilistic_pca()
    print("\n" + "=" * 60 + "\n")
    
    demonstrate_variational_inference()
    
    print("\n" + "=" * 60)
    print("概率模型学习演示完成！")
    print("涵盖内容：")
    print("1. 高斯混合模型与EM算法")
    print("2. 隐马尔可夫模型与Baum-Welch算法")
    print("3. 概率主成分分析")
    print("4. 变分推断与变分贝叶斯")
    print("5. 参数估计与模型选择")

if __name__ == "__main__":
    main() 