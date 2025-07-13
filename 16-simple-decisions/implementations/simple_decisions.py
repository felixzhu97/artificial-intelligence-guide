#!/usr/bin/env python3
"""
第16章：简单决策 (Simple Decisions)

本模块实现了决策理论的核心概念：
- 决策理论基础
- 效用理论
- 决策树
- 期望效用最大化
- 风险态度
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class NodeType(Enum):
    """节点类型"""
    DECISION = "decision"      # 决策节点（方形）
    CHANCE = "chance"          # 机会节点（圆形）
    TERMINAL = "terminal"      # 终端节点（三角形）

@dataclass
class DecisionNode:
    """决策节点"""
    name: str
    node_type: NodeType
    value: float = 0.0
    children: Dict[str, 'DecisionNode'] = None
    probabilities: Dict[str, float] = None  # 仅用于机会节点
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.probabilities is None:
            self.probabilities = {}

class DecisionTree:
    """决策树"""
    
    def __init__(self, root_name: str):
        self.root = DecisionNode(root_name, NodeType.DECISION)
        self.utility_function = None
    
    def add_decision_node(self, parent_path: List[str], node_name: str, action_name: str):
        """添加决策节点"""
        parent = self._get_node(parent_path)
        if parent:
            child = DecisionNode(node_name, NodeType.DECISION)
            parent.children[action_name] = child
            return child
        return None
    
    def add_chance_node(self, parent_path: List[str], node_name: str, action_name: str, 
                       outcomes: Dict[str, float]):
        """添加机会节点"""
        parent = self._get_node(parent_path)
        if parent:
            child = DecisionNode(node_name, NodeType.CHANCE)
            child.probabilities = outcomes
            parent.children[action_name] = child
            return child
        return None
    
    def add_terminal_node(self, parent_path: List[str], value: float, outcome_name: str):
        """添加终端节点"""
        parent = self._get_node(parent_path)
        if parent:
            terminal = DecisionNode(f"Terminal_{value}", NodeType.TERMINAL, value)
            parent.children[outcome_name] = terminal
            return terminal
        return None
    
    def _get_node(self, path: List[str]) -> Optional[DecisionNode]:
        """根据路径获取节点"""
        current = self.root
        for action in path:
            if action in current.children:
                current = current.children[action]
            else:
                return None
        return current
    
    def calculate_expected_utility(self, node: DecisionNode = None) -> float:
        """计算期望效用"""
        if node is None:
            node = self.root
        
        if node.node_type == NodeType.TERMINAL:
            return node.value
        
        elif node.node_type == NodeType.CHANCE:
            expected_value = 0.0
            for outcome, probability in node.probabilities.items():
                if outcome in node.children:
                    child_value = self.calculate_expected_utility(node.children[outcome])
                    expected_value += probability * child_value
            node.value = expected_value
            return expected_value
        
        elif node.node_type == NodeType.DECISION:
            max_value = float('-inf')
            for action, child in node.children.items():
                child_value = self.calculate_expected_utility(child)
                max_value = max(max_value, child_value)
            node.value = max_value
            return max_value
        
        return 0.0
    
    def get_optimal_strategy(self, node: DecisionNode = None, path: List[str] = None) -> Dict[str, Any]:
        """获取最优策略"""
        if node is None:
            node = self.root
        if path is None:
            path = []
        
        strategy = {"path": path.copy(), "value": node.value}
        
        if node.node_type == NodeType.DECISION:
            best_action = None
            best_value = float('-inf')
            
            for action, child in node.children.items():
                child_value = child.value
                if child_value > best_value:
                    best_value = child_value
                    best_action = action
            
            strategy["best_action"] = best_action
            if best_action and best_action in node.children:
                child_strategy = self.get_optimal_strategy(
                    node.children[best_action], 
                    path + [best_action]
                )
                strategy["next"] = child_strategy
        
        return strategy

class UtilityFunction:
    """效用函数"""
    
    def __init__(self, function_type: str = "linear"):
        self.function_type = function_type
        self.parameters = {}
    
    def set_linear(self, slope: float = 1.0, intercept: float = 0.0):
        """设置线性效用函数"""
        self.function_type = "linear"
        self.parameters = {"slope": slope, "intercept": intercept}
    
    def set_logarithmic(self, base: float = np.e, scale: float = 1.0):
        """设置对数效用函数（风险规避）"""
        self.function_type = "logarithmic"
        self.parameters = {"base": base, "scale": scale}
    
    def set_exponential(self, risk_aversion: float = 0.1):
        """设置指数效用函数"""
        self.function_type = "exponential"
        self.parameters = {"risk_aversion": risk_aversion}
    
    def set_power(self, exponent: float = 0.5):
        """设置幂效用函数"""
        self.function_type = "power"
        self.parameters = {"exponent": exponent}
    
    def calculate(self, wealth: float) -> float:
        """计算效用值"""
        if self.function_type == "linear":
            return (self.parameters.get("slope", 1.0) * wealth + 
                   self.parameters.get("intercept", 0.0))
        
        elif self.function_type == "logarithmic":
            if wealth <= 0:
                return float('-inf')
            base = self.parameters.get("base", np.e)
            scale = self.parameters.get("scale", 1.0)
            return scale * np.log(wealth) / np.log(base)
        
        elif self.function_type == "exponential":
            risk_aversion = self.parameters.get("risk_aversion", 0.1)
            return -np.exp(-risk_aversion * wealth)
        
        elif self.function_type == "power":
            if wealth <= 0:
                return 0
            exponent = self.parameters.get("exponent", 0.5)
            return wealth ** exponent
        
        return wealth

class LotteryComparison:
    """彩票比较（用于研究风险态度）"""
    
    def __init__(self):
        self.utility_function = UtilityFunction()
    
    def create_lottery(self, outcomes: List[float], probabilities: List[float]) -> Dict[str, Any]:
        """创建彩票"""
        if len(outcomes) != len(probabilities):
            raise ValueError("结果数量必须等于概率数量")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("概率之和必须等于1")
        
        return {
            "outcomes": outcomes,
            "probabilities": probabilities,
            "expected_value": sum(o * p for o, p in zip(outcomes, probabilities))
        }
    
    def calculate_expected_utility(self, lottery: Dict[str, Any]) -> float:
        """计算彩票的期望效用"""
        expected_utility = 0.0
        for outcome, probability in zip(lottery["outcomes"], lottery["probabilities"]):
            utility = self.utility_function.calculate(outcome)
            expected_utility += probability * utility
        return expected_utility
    
    def compare_lotteries(self, lottery1: Dict[str, Any], lottery2: Dict[str, Any]) -> Dict[str, Any]:
        """比较两个彩票"""
        eu1 = self.calculate_expected_utility(lottery1)
        eu2 = self.calculate_expected_utility(lottery2)
        
        return {
            "lottery1_eu": eu1,
            "lottery2_eu": eu2,
            "preferred": "Lottery 1" if eu1 > eu2 else "Lottery 2" if eu2 > eu1 else "Indifferent",
            "utility_difference": eu1 - eu2
        }
    
    def certainty_equivalent(self, lottery: Dict[str, Any], precision: float = 0.01) -> float:
        """计算确定性等价"""
        lottery_eu = self.calculate_expected_utility(lottery)
        
        # 二分搜索找到确定性等价
        low = min(lottery["outcomes"])
        high = max(lottery["outcomes"])
        
        while high - low > precision:
            mid = (low + high) / 2
            mid_utility = self.utility_function.calculate(mid)
            
            if mid_utility < lottery_eu:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2

class MultiAttributeDecision:
    """多属性决策"""
    
    def __init__(self):
        self.attributes = []
        self.weights = {}
        self.alternatives = {}
    
    def add_attribute(self, name: str, weight: float, higher_better: bool = True):
        """添加属性"""
        self.attributes.append(name)
        self.weights[name] = {
            "weight": weight,
            "higher_better": higher_better
        }
    
    def add_alternative(self, name: str, attribute_values: Dict[str, float]):
        """添加备选方案"""
        self.alternatives[name] = attribute_values
    
    def normalize_weights(self):
        """归一化权重"""
        total_weight = sum(attr["weight"] for attr in self.weights.values())
        for attr in self.weights.values():
            attr["weight"] /= total_weight
    
    def simple_additive_weighting(self) -> Dict[str, float]:
        """简单加权法（SAW）"""
        self.normalize_weights()
        
        # 归一化属性值
        normalized_values = {}
        for attr in self.attributes:
            values = [alt_values[attr] for alt_values in self.alternatives.values()]
            min_val, max_val = min(values), max(values)
            
            for alt_name, alt_values in self.alternatives.items():
                if alt_name not in normalized_values:
                    normalized_values[alt_name] = {}
                
                if max_val > min_val:
                    if self.weights[attr]["higher_better"]:
                        normalized_values[alt_name][attr] = (alt_values[attr] - min_val) / (max_val - min_val)
                    else:
                        normalized_values[alt_name][attr] = (max_val - alt_values[attr]) / (max_val - min_val)
                else:
                    normalized_values[alt_name][attr] = 1.0
        
        # 计算加权得分
        scores = {}
        for alt_name in self.alternatives:
            score = 0.0
            for attr in self.attributes:
                score += (self.weights[attr]["weight"] * 
                         normalized_values[alt_name][attr])
            scores[alt_name] = score
        
        return scores

def demo_decision_tree():
    """演示决策树"""
    print("\n" + "="*50)
    print("决策树演示")
    print("="*50)
    
    print("\n投资决策问题:")
    print("- 选择：投资股票 vs 银行存款")
    print("- 股票投资：高收益(30万, 概率0.4) 或 低收益(5万, 概率0.6)")
    print("- 银行存款：确定收益(10万)")
    
    # 创建决策树
    dt = DecisionTree("投资决策")
    
    # 添加投资选择
    stock_node = dt.add_chance_node([], "股票投资", "投资股票", 
                                   {"高收益": 0.4, "低收益": 0.6})
    dt.add_terminal_node(["投资股票"], 30, "高收益")  # 30万
    dt.add_terminal_node(["投资股票"], 5, "低收益")   # 5万
    
    dt.add_terminal_node([], 10, "银行存款")  # 10万
    
    # 计算期望效用
    expected_utility = dt.calculate_expected_utility()
    print(f"\n决策树分析:")
    print(f"股票投资期望收益: {0.4 * 30 + 0.6 * 5} 万元")
    print(f"银行存款确定收益: 10 万元")
    print(f"最优期望效用: {expected_utility} 万元")
    
    # 获取最优策略
    strategy = dt.get_optimal_strategy()
    print(f"最优决策: {strategy['best_action']}")

def demo_utility_functions():
    """演示效用函数"""
    print("\n" + "="*50)
    print("效用函数演示")
    print("="*50)
    
    # 创建不同类型的效用函数
    wealth_range = np.linspace(1, 100, 100)
    
    # 线性效用函数（风险中性）
    linear_utility = UtilityFunction()
    linear_utility.set_linear(slope=1.0)
    linear_values = [linear_utility.calculate(w) for w in wealth_range]
    
    # 对数效用函数（风险规避）
    log_utility = UtilityFunction()
    log_utility.set_logarithmic(scale=10)
    log_values = [log_utility.calculate(w) for w in wealth_range]
    
    # 幂效用函数（风险规避）
    power_utility = UtilityFunction()
    power_utility.set_power(exponent=0.5)
    power_values = [power_utility.calculate(w) for w in wealth_range]
    
    # 绘制效用函数
    plt.figure(figsize=(10, 6))
    plt.plot(wealth_range, linear_values, 'b-', label='线性效用（风险中性）', linewidth=2)
    plt.plot(wealth_range, log_values, 'r-', label='对数效用（风险规避）', linewidth=2)
    plt.plot(wealth_range, power_values, 'g-', label='幂效用（风险规避）', linewidth=2)
    
    plt.xlabel('财富')
    plt.ylabel('效用')
    plt.title('不同类型的效用函数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('utility_functions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("效用函数图已保存为 'utility_functions.png'")
    
    # 风险态度分析
    print(f"\n风险态度分析（财富=50时）:")
    print(f"线性效用: {linear_utility.calculate(50):.2f}")
    print(f"对数效用: {log_utility.calculate(50):.2f}")
    print(f"幂效用: {power_utility.calculate(50):.2f}")

def demo_lottery_comparison():
    """演示彩票比较"""
    print("\n" + "="*50)
    print("彩票比较演示")
    print("="*50)
    
    lc = LotteryComparison()
    lc.utility_function.set_logarithmic(scale=1.0)
    
    print("\n彩票比较（对数效用函数）:")
    
    # 创建两个彩票
    lottery1 = lc.create_lottery([100, 0], [0.5, 0.5])    # 50%获得100，50%获得0
    lottery2 = lc.create_lottery([40], [1.0])             # 确定获得40
    
    print(f"彩票1: 50%概率获得100, 50%概率获得0")
    print(f"期望值: {lottery1['expected_value']}")
    
    print(f"彩票2: 100%概率获得40")
    print(f"期望值: {lottery2['expected_value']}")
    
    # 比较彩票
    comparison = lc.compare_lotteries(lottery1, lottery2)
    print(f"\n比较结果:")
    print(f"彩票1期望效用: {comparison['lottery1_eu']:.3f}")
    print(f"彩票2期望效用: {comparison['lottery2_eu']:.3f}")
    print(f"偏好: {comparison['preferred']}")
    
    # 计算确定性等价
    ce1 = lc.certainty_equivalent(lottery1)
    ce2 = lc.certainty_equivalent(lottery2)
    print(f"\n确定性等价:")
    print(f"彩票1的确定性等价: {ce1:.2f}")
    print(f"彩票2的确定性等价: {ce2:.2f}")
    
    risk_premium = lottery1['expected_value'] - ce1
    print(f"彩票1的风险溢价: {risk_premium:.2f}")

def demo_multiattribute_decision():
    """演示多属性决策"""
    print("\n" + "="*50)
    print("多属性决策演示")
    print("="*50)
    
    print("\n汽车购买决策:")
    print("属性: 价格(越低越好), 油耗(越低越好), 安全性(越高越好), 舒适度(越高越好)")
    
    # 创建多属性决策模型
    mad = MultiAttributeDecision()
    
    # 添加属性
    mad.add_attribute("价格", 0.3, False)     # 价格越低越好
    mad.add_attribute("油耗", 0.2, False)     # 油耗越低越好
    mad.add_attribute("安全性", 0.3, True)    # 安全性越高越好
    mad.add_attribute("舒适度", 0.2, True)    # 舒适度越高越好
    
    # 添加备选方案
    mad.add_alternative("车型A", {"价格": 20, "油耗": 8, "安全性": 90, "舒适度": 85})
    mad.add_alternative("车型B", {"价格": 25, "油耗": 6, "安全性": 95, "舒适度": 90})
    mad.add_alternative("车型C", {"价格": 15, "油耗": 10, "安全性": 80, "舒适度": 75})
    mad.add_alternative("车型D", {"价格": 30, "油耗": 5, "安全性": 98, "舒适度": 95})
    
    print("\n备选方案:")
    for name, values in mad.alternatives.items():
        print(f"{name}: 价格{values['价格']}万, 油耗{values['油耗']}L/100km, "
              f"安全性{values['安全性']}分, 舒适度{values['舒适度']}分")
    
    # 计算得分
    scores = mad.simple_additive_weighting()
    
    print(f"\n简单加权法得分:")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(sorted_scores, 1):
        print(f"{i}. {name}: {score:.3f}")
    
    print(f"\n推荐选择: {sorted_scores[0][0]}")

def demo_decision_under_uncertainty():
    """演示不确定性下的决策"""
    print("\n" + "="*50)
    print("不确定性下的决策演示")
    print("="*50)
    
    print("\n农民种植决策问题:")
    print("作物选择: 玉米, 小麦, 大豆")
    print("天气状况: 干旱, 正常, 多雨")
    
    # 收益矩阵（万元）
    payoff_matrix = {
        "玉米": {"干旱": 2, "正常": 8, "多雨": 5},
        "小麦": {"干旱": 4, "正常": 6, "多雨": 3},
        "大豆": {"干旱": 1, "正常": 7, "多雨": 9}
    }
    
    # 天气概率
    weather_probs = {"干旱": 0.2, "正常": 0.6, "多雨": 0.2}
    
    print("\n收益矩阵（万元）:")
    print(f"{'作物':>6} {'干旱':>6} {'正常':>6} {'多雨':>6}")
    for crop, outcomes in payoff_matrix.items():
        print(f"{crop:>6} {outcomes['干旱']:>6} {outcomes['正常']:>6} {outcomes['多雨']:>6}")
    
    # 计算期望收益
    print(f"\n期望收益:")
    expected_payoffs = {}
    for crop, outcomes in payoff_matrix.items():
        expected = sum(outcomes[weather] * weather_probs[weather] 
                      for weather in weather_probs)
        expected_payoffs[crop] = expected
        print(f"{crop}: {expected:.2f} 万元")
    
    # 最优决策
    best_crop = max(expected_payoffs, key=expected_payoffs.get)
    print(f"\n期望收益最大化决策: {best_crop} ({expected_payoffs[best_crop]:.2f} 万元)")
    
    # 最大最小决策（悲观准则）
    print(f"\n最大最小决策（悲观准则）:")
    maximin_payoffs = {}
    for crop, outcomes in payoff_matrix.items():
        min_payoff = min(outcomes.values())
        maximin_payoffs[crop] = min_payoff
        print(f"{crop}: 最坏情况收益 {min_payoff} 万元")
    
    best_maximin = max(maximin_payoffs, key=maximin_payoffs.get)
    print(f"最大最小决策: {best_maximin}")
    
    # 最大最大决策（乐观准则）
    print(f"\n最大最大决策（乐观准则）:")
    maximax_payoffs = {}
    for crop, outcomes in payoff_matrix.items():
        max_payoff = max(outcomes.values())
        maximax_payoffs[crop] = max_payoff
        print(f"{crop}: 最好情况收益 {max_payoff} 万元")
    
    best_maximax = max(maximax_payoffs, key=maximax_payoffs.get)
    print(f"最大最大决策: {best_maximax}")

def run_comprehensive_demo():
    """运行完整演示"""
    print("⚖️ 第16章：简单决策 - 完整演示")
    print("="*60)
    
    # 运行各个演示
    demo_decision_tree()
    demo_utility_functions()
    demo_lottery_comparison()
    demo_multiattribute_decision()
    demo_decision_under_uncertainty()
    
    print("\n" + "="*60)
    print("简单决策演示完成！")
    print("="*60)
    print("\n📚 学习要点:")
    print("• 决策树为序贯决策提供了系统化分析工具")
    print("• 效用函数反映决策者的风险态度")
    print("• 期望效用最大化是理性决策的基础")
    print("• 多属性决策帮助处理复杂的权衡问题")
    print("• 不确定性下的决策需要考虑不同的决策准则")

if __name__ == "__main__":
    run_comprehensive_demo() 