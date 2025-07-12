"""
高级AI应用：智能决策系统
整合了概率推理、决策理论、多目标优化、风险评估等AI技术
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json


class DecisionOutcome:
    """决策结果"""
    
    def __init__(self, value: float, probability: float, description: str = ""):
        self.value = value
        self.probability = probability
        self.description = description
    
    def expected_value(self) -> float:
        """期望值"""
        return self.value * self.probability
    
    def __str__(self):
        return f"Outcome(value={self.value}, prob={self.probability}, desc='{self.description}')"


class DecisionAlternative:
    """决策选择"""
    
    def __init__(self, name: str, outcomes: List[DecisionOutcome]):
        self.name = name
        self.outcomes = outcomes
    
    def expected_value(self) -> float:
        """计算期望值"""
        return sum(outcome.expected_value() for outcome in self.outcomes)
    
    def variance(self) -> float:
        """计算方差（风险度量）"""
        ev = self.expected_value()
        return sum(outcome.probability * (outcome.value - ev) ** 2 for outcome in self.outcomes)
    
    def risk_measure(self) -> float:
        """风险度量（标准差）"""
        return math.sqrt(self.variance())
    
    def __str__(self):
        return f"Alternative: {self.name}, EV: {self.expected_value():.2f}, Risk: {self.risk_measure():.2f}"


class DecisionCriterion(Enum):
    """决策准则"""
    EXPECTED_VALUE = "expected_value"
    RISK_AVERSE = "risk_averse"
    MINIMAX = "minimax"
    MAXIMAX = "maximax"
    HURWICZ = "hurwicz"
    SAVAGE = "savage"


@dataclass
class Criterion:
    """决策标准"""
    name: str
    weight: float
    maximize: bool = True
    
    def normalize_score(self, score: float, min_score: float, max_score: float) -> float:
        """标准化分数"""
        if max_score == min_score:
            return 0.5
        
        normalized = (score - min_score) / (max_score - min_score)
        return normalized if self.maximize else (1 - normalized)


class MultiCriteriaDecisionMaker:
    """多准则决策分析"""
    
    def __init__(self, criteria: List[Criterion]):
        self.criteria = criteria
        self.normalize_weights()
    
    def normalize_weights(self):
        """归一化权重"""
        total_weight = sum(c.weight for c in self.criteria)
        if total_weight > 0:
            for criterion in self.criteria:
                criterion.weight /= total_weight
    
    def evaluate_alternatives(self, alternatives: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """评估备选方案
        
        Args:
            alternatives: {alternative_name: {criterion_name: score}}
        
        Returns:
            {alternative_name: total_score}
        """
        # 收集所有分数用于归一化
        criterion_scores = defaultdict(list)
        for alt_scores in alternatives.values():
            for criterion_name, score in alt_scores.items():
                criterion_scores[criterion_name].append(score)
        
        # 计算归一化分数
        normalized_scores = {}
        for alt_name, alt_scores in alternatives.items():
            normalized_scores[alt_name] = 0
            
            for criterion in self.criteria:
                if criterion.name in alt_scores:
                    score = alt_scores[criterion.name]
                    scores_list = criterion_scores[criterion.name]
                    min_score, max_score = min(scores_list), max(scores_list)
                    
                    normalized = criterion.normalize_score(score, min_score, max_score)
                    normalized_scores[alt_name] += criterion.weight * normalized
        
        return normalized_scores


class RiskAssessment:
    """风险评估模块"""
    
    def __init__(self):
        self.risk_factors = {}
        self.risk_matrix = {}
    
    def add_risk_factor(self, factor_name: str, probability: float, impact: float):
        """添加风险因子"""
        self.risk_factors[factor_name] = {
            'probability': probability,
            'impact': impact,
            'risk_score': probability * impact
        }
    
    def calculate_portfolio_risk(self, portfolio: List[str]) -> float:
        """计算投资组合风险"""
        total_risk = 0
        for factor in portfolio:
            if factor in self.risk_factors:
                total_risk += self.risk_factors[factor]['risk_score']
        return total_risk
    
    def monte_carlo_simulation(self, scenarios: List[Callable[[], float]], 
                              num_simulations: int = 10000) -> Dict[str, float]:
        """蒙特卡洛风险模拟"""
        results = []
        
        for _ in range(num_simulations):
            scenario_result = sum(scenario() for scenario in scenarios)
            results.append(scenario_result)
        
        results = np.array(results)
        
        return {
            'mean': np.mean(results),
            'std': np.std(results),
            'var_95': np.percentile(results, 5),  # 95% VaR
            'var_99': np.percentile(results, 1),  # 99% VaR
            'max_loss': np.min(results),
            'max_gain': np.max(results)
        }


class BayesianDecisionNetwork:
    """贝叶斯决策网络"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.probabilities = {}
    
    def add_chance_node(self, name: str, states: List[str], 
                       probabilities: Dict[Tuple, float]):
        """添加机会节点"""
        self.nodes[name] = {'type': 'chance', 'states': states}
        self.probabilities[name] = probabilities
    
    def add_decision_node(self, name: str, alternatives: List[str]):
        """添加决策节点"""
        self.nodes[name] = {'type': 'decision', 'alternatives': alternatives}
    
    def add_utility_node(self, name: str, utility_function: Callable):
        """添加效用节点"""
        self.nodes[name] = {'type': 'utility', 'function': utility_function}
    
    def add_edge(self, from_node: str, to_node: str):
        """添加边"""
        self.edges.append((from_node, to_node))
    
    def calculate_expected_utility(self, decision_values: Dict[str, str]) -> float:
        """计算期望效用"""
        # 简化实现：枚举所有可能的机会节点状态
        chance_nodes = [name for name, node in self.nodes.items() 
                       if node['type'] == 'chance']
        
        if not chance_nodes:
            return 0
        
        total_utility = 0
        
        # 枚举所有可能的状态组合
        from itertools import product
        
        all_states = [self.nodes[node]['states'] for node in chance_nodes]
        
        for state_combination in product(*all_states):
            # 计算这个状态组合的概率
            prob = 1.0
            state_dict = dict(zip(chance_nodes, state_combination))
            
            for node in chance_nodes:
                state = state_dict[node]
                node_prob = self.probabilities[node].get((state,), 0)
                prob *= node_prob
            
            # 计算效用
            utility_nodes = [name for name, node in self.nodes.items() 
                           if node['type'] == 'utility']
            
            for utility_node in utility_nodes:
                utility_func = self.nodes[utility_node]['function']
                utility = utility_func(state_dict, decision_values)
                total_utility += prob * utility
        
        return total_utility


class PortfolioOptimizer:
    """投资组合优化器"""
    
    def __init__(self, assets: List[str], expected_returns: List[float], 
                 covariance_matrix: np.ndarray):
        self.assets = assets
        self.expected_returns = np.array(expected_returns)
        self.covariance_matrix = covariance_matrix
        self.num_assets = len(assets)
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float]:
        """计算投资组合的期望收益和风险"""
        expected_return = np.dot(weights, self.expected_returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        portfolio_risk = math.sqrt(portfolio_variance)
        
        return expected_return, portfolio_risk
    
    def efficient_frontier(self, num_portfolios: int = 1000) -> List[Tuple[float, float, np.ndarray]]:
        """生成有效前沿"""
        results = []
        
        for _ in range(num_portfolios):
            # 随机生成权重
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)  # 归一化
            
            expected_return, risk = self.calculate_portfolio_metrics(weights)
            results.append((expected_return, risk, weights))
        
        return results
    
    def optimize_sharpe_ratio(self, risk_free_rate: float = 0.02) -> np.ndarray:
        """优化夏普比率"""
        best_sharpe = -np.inf
        best_weights = None
        
        for _ in range(10000):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            
            expected_return, risk = self.calculate_portfolio_metrics(weights)
            
            if risk > 0:
                sharpe_ratio = (expected_return - risk_free_rate) / risk
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_weights = weights
        
        return best_weights


class SmartInvestmentAdvisor:
    """智能投资顾问"""
    
    def __init__(self):
        self.user_profile = {}
        self.market_data = {}
        self.risk_tolerance = 0.5  # 0-1之间
        
    def assess_risk_tolerance(self, questionnaire_responses: Dict[str, Any]) -> float:
        """评估风险承受能力"""
        risk_score = 0
        
        # 年龄因子
        age = questionnaire_responses.get('age', 30)
        age_factor = max(0, (65 - age) / 40)  # 年龄越小，风险承受能力越强
        risk_score += age_factor * 0.3
        
        # 收入稳定性
        income_stability = questionnaire_responses.get('income_stability', 3)  # 1-5
        risk_score += (income_stability / 5) * 0.2
        
        # 投资经验
        experience = questionnaire_responses.get('investment_experience', 1)  # 1-5
        risk_score += (experience / 5) * 0.2
        
        # 投资目标
        investment_horizon = questionnaire_responses.get('investment_horizon', 5)  # 年数
        horizon_factor = min(1, investment_horizon / 10)
        risk_score += horizon_factor * 0.3
        
        return max(0, min(1, risk_score))
    
    def generate_portfolio_recommendation(self, investment_amount: float) -> Dict[str, Any]:
        """生成投资组合建议"""
        # 基于风险承受能力分配资产
        if self.risk_tolerance < 0.3:
            # 保守型
            allocation = {
                '债券': 0.7,
                '货币基金': 0.2,
                '股票': 0.1
            }
        elif self.risk_tolerance < 0.7:
            # 平衡型
            allocation = {
                '债券': 0.4,
                '股票': 0.5,
                '房地产基金': 0.1
            }
        else:
            # 激进型
            allocation = {
                '股票': 0.7,
                '成长股': 0.2,
                '债券': 0.1
            }
        
        # 计算具体投资金额
        investment_plan = {}
        for asset, ratio in allocation.items():
            investment_plan[asset] = investment_amount * ratio
        
        # 估算预期收益和风险
        expected_returns = {
            '债券': 0.04, '货币基金': 0.02, '股票': 0.08,
            '房地产基金': 0.06, '成长股': 0.12
        }
        
        portfolio_return = sum(allocation.get(asset, 0) * expected_returns.get(asset, 0) 
                             for asset in allocation)
        
        return {
            'allocation': allocation,
            'investment_plan': investment_plan,
            'expected_annual_return': portfolio_return,
            'risk_level': 'Low' if self.risk_tolerance < 0.3 else 
                         'Medium' if self.risk_tolerance < 0.7 else 'High'
        }


class BusinessDecisionSupport:
    """商业决策支持系统"""
    
    def __init__(self):
        self.criteria = [
            Criterion("盈利能力", 0.35, True),
            Criterion("风险水平", 0.25, False),  # 风险越低越好
            Criterion("市场潜力", 0.20, True),
            Criterion("投资回收期", 0.20, False)  # 回收期越短越好
        ]
        
        self.mcdm = MultiCriteriaDecisionMaker(self.criteria)
    
    def evaluate_business_projects(self, projects: Dict[str, Dict]) -> Dict[str, float]:
        """评估商业项目"""
        # 转换项目数据为MCDM格式
        alternatives = {}
        
        for project_name, project_data in projects.items():
            alternatives[project_name] = {
                "盈利能力": project_data.get("profit_margin", 0),
                "风险水平": project_data.get("risk_score", 5),
                "市场潜力": project_data.get("market_size", 0),
                "投资回收期": project_data.get("payback_period", 10)
            }
        
        return self.mcdm.evaluate_alternatives(alternatives)
    
    def sensitivity_analysis(self, base_scenario: Dict, 
                           variables: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """敏感性分析"""
        results = defaultdict(list)
        
        for var_name, var_values in variables.items():
            for value in var_values:
                scenario = base_scenario.copy()
                scenario[var_name] = value
                
                # 重新计算项目评分
                alternatives = {"project": scenario}
                score = self.mcdm.evaluate_alternatives(alternatives)["project"]
                results[var_name].append(score)
        
        return dict(results)


def demonstrate_investment_advisor():
    """演示智能投资顾问"""
    print("=== 智能投资顾问演示 ===")
    
    advisor = SmartInvestmentAdvisor()
    
    # 模拟用户画像
    user_profiles = [
        {
            'name': '年轻专业人士',
            'age': 28,
            'income_stability': 4,
            'investment_experience': 2,
            'investment_horizon': 10
        },
        {
            'name': '中年家庭',
            'age': 45,
            'income_stability': 4,
            'investment_experience': 3,
            'investment_horizon': 15
        },
        {
            'name': '临近退休',
            'age': 58,
            'income_stability': 3,
            'investment_experience': 4,
            'investment_horizon': 5
        }
    ]
    
    for profile in user_profiles:
        print(f"\n--- {profile['name']} ---")
        
        risk_tolerance = advisor.assess_risk_tolerance(profile)
        advisor.risk_tolerance = risk_tolerance
        
        print(f"风险承受能力: {risk_tolerance:.2f}")
        
        recommendation = advisor.generate_portfolio_recommendation(100000)
        print(f"推荐风险等级: {recommendation['risk_level']}")
        print(f"预期年收益率: {recommendation['expected_annual_return']:.1%}")
        print("资产配置:")
        for asset, ratio in recommendation['allocation'].items():
            print(f"  {asset}: {ratio:.1%}")


def demonstrate_business_decision():
    """演示商业决策支持"""
    print("\n=== 商业决策支持演示 ===")
    
    decision_support = BusinessDecisionSupport()
    
    # 模拟项目数据
    projects = {
        "项目A - 移动应用": {
            "profit_margin": 0.25,
            "risk_score": 6,
            "market_size": 8,
            "payback_period": 2
        },
        "项目B - 电商平台": {
            "profit_margin": 0.15,
            "risk_score": 4,
            "market_size": 9,
            "payback_period": 3
        },
        "项目C - AI产品": {
            "profit_margin": 0.35,
            "risk_score": 8,
            "market_size": 7,
            "payback_period": 4
        }
    }
    
    scores = decision_support.evaluate_business_projects(projects)
    
    print("项目评估结果:")
    for project, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{project}: {score:.3f}")
    
    # 敏感性分析
    print("\n敏感性分析 (项目A):")
    base_scenario = projects["项目A - 移动应用"]
    sensitivity_vars = {
        "profit_margin": [0.15, 0.20, 0.25, 0.30, 0.35],
        "risk_score": [4, 5, 6, 7, 8]
    }
    
    sensitivity_results = decision_support.sensitivity_analysis(base_scenario, sensitivity_vars)
    
    for var, scores in sensitivity_results.items():
        print(f"{var}: {scores}")


def demonstrate_risk_assessment():
    """演示风险评估"""
    print("\n=== 风险评估演示 ===")
    
    risk_assessment = RiskAssessment()
    
    # 添加风险因子
    risk_assessment.add_risk_factor("市场风险", 0.3, 0.8)
    risk_assessment.add_risk_factor("信用风险", 0.1, 0.9)
    risk_assessment.add_risk_factor("操作风险", 0.2, 0.6)
    risk_assessment.add_risk_factor("流动性风险", 0.15, 0.7)
    
    print("风险因子分析:")
    for factor, data in risk_assessment.risk_factors.items():
        print(f"{factor}: 概率={data['probability']:.2f}, 影响={data['impact']:.2f}, "
              f"风险分数={data['risk_score']:.3f}")
    
    # 蒙特卡洛模拟
    scenarios = [
        lambda: np.random.normal(0.08, 0.15),  # 股票收益
        lambda: np.random.normal(0.04, 0.05),  # 债券收益
        lambda: np.random.normal(-0.02, 0.03)  # 现金流风险
    ]
    
    simulation_results = risk_assessment.monte_carlo_simulation(scenarios, 5000)
    
    print("\n蒙特卡洛模拟结果:")
    for metric, value in simulation_results.items():
        print(f"{metric}: {value:.4f}")


def main():
    """主演示函数"""
    print("高级AI应用：智能决策系统")
    print("整合了概率推理、决策理论、风险评估、多目标优化等技术")
    
    demonstrate_investment_advisor()
    demonstrate_business_decision()
    demonstrate_risk_assessment()
    
    print("\n=== 智能决策系统技术总结 ===")
    print("1. 多准则决策分析: 处理复杂的多目标决策问题")
    print("2. 贝叶斯决策网络: 不确定性下的最优决策")
    print("3. 风险评估与管理: 量化和控制决策风险")
    print("4. 投资组合优化: 平衡收益与风险的资产配置")
    print("5. 敏感性分析: 评估参数变化对决策的影响")
    print("6. 蒙特卡洛模拟: 复杂系统的风险建模")


if __name__ == "__main__":
    main() 