"""
逻辑代理实现

包含命题逻辑、一阶逻辑、推理引擎等
"""

import re
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from collections import defaultdict
from itertools import combinations
import copy


class PropositionalLogic:
    """命题逻辑处理器"""
    
    def __init__(self):
        self.symbols = set()
        self.clauses = []
    
    def parse_formula(self, formula: str) -> List[str]:
        """解析公式为子句"""
        # 简化的解析器，处理基本的逻辑运算符
        formula = formula.replace('(', ' ( ').replace(')', ' ) ')
        formula = formula.replace('&', ' & ').replace('|', ' | ')
        formula = formula.replace('~', ' ~ ').replace('->', ' -> ')
        
        tokens = formula.split()
        return self.parse_tokens(tokens)
    
    def parse_tokens(self, tokens: List[str]) -> List[str]:
        """解析标记"""
        # 简化实现，返回标记列表
        return [token for token in tokens if token.isalpha()]
    
    def add_clause(self, clause: str):
        """添加子句"""
        symbols = self.parse_formula(clause)
        self.symbols.update(symbols)
        self.clauses.append(clause)
    
    def is_satisfiable(self) -> bool:
        """检查是否可满足"""
        # 使用DPLL算法的简化版本
        symbols = list(self.symbols)
        return self.dpll(symbols, {})
    
    def dpll(self, symbols: List[str], assignment: Dict[str, bool]) -> bool:
        """DPLL算法"""
        if not symbols:
            return self.evaluate_clauses(assignment)
        
        symbol = symbols[0]
        remaining = symbols[1:]
        
        # 尝试True
        assignment[symbol] = True
        if self.dpll(remaining, assignment):
            return True
        
        # 尝试False
        assignment[symbol] = False
        if self.dpll(remaining, assignment):
            return True
        
        # 回溯
        del assignment[symbol]
        return False
    
    def evaluate_clauses(self, assignment: Dict[str, bool]) -> bool:
        """评估子句"""
        # 简化实现，假设所有子句都是文字的析取
        for clause in self.clauses:
            if not self.evaluate_clause(clause, assignment):
                return False
        return True
    
    def evaluate_clause(self, clause: str, assignment: Dict[str, bool]) -> bool:
        """评估单个子句"""
        # 简化实现
        symbols = self.parse_formula(clause)
        for symbol in symbols:
            if symbol in assignment and assignment[symbol]:
                return True
        return False


class Literal:
    """文字（原子公式或其否定）"""
    
    def __init__(self, symbol: str, negated: bool = False):
        self.symbol = symbol
        self.negated = negated
    
    def __eq__(self, other):
        if not isinstance(other, Literal):
            return False
        return self.symbol == other.symbol and self.negated == other.negated
    
    def __hash__(self):
        return hash((self.symbol, self.negated))
    
    def __str__(self):
        return f"~{self.symbol}" if self.negated else self.symbol
    
    def __repr__(self):
        return self.__str__()
    
    def negate(self) -> 'Literal':
        """返回否定"""
        return Literal(self.symbol, not self.negated)


class Clause:
    """子句（文字的析取）"""
    
    def __init__(self, literals: List[Literal]):
        self.literals = set(literals)
    
    def __eq__(self, other):
        if not isinstance(other, Clause):
            return False
        return self.literals == other.literals
    
    def __hash__(self):
        return hash(frozenset(self.literals))
    
    def __str__(self):
        if not self.literals:
            return "⊥"
        return " ∨ ".join(str(lit) for lit in self.literals)
    
    def __repr__(self):
        return self.__str__()
    
    def is_empty(self) -> bool:
        """检查是否为空子句"""
        return len(self.literals) == 0
    
    def is_unit(self) -> bool:
        """检查是否为单元子句"""
        return len(self.literals) == 1
    
    def get_unit_literal(self) -> Optional[Literal]:
        """获取单元子句的文字"""
        if self.is_unit():
            return list(self.literals)[0]
        return None
    
    def contains_literal(self, literal: Literal) -> bool:
        """检查是否包含指定文字"""
        return literal in self.literals
    
    def remove_literal(self, literal: Literal) -> 'Clause':
        """移除文字"""
        new_literals = self.literals - {literal}
        return Clause(list(new_literals))


class CNFConverter:
    """CNF转换器"""
    
    def __init__(self):
        self.symbol_counter = 0
    
    def convert_to_cnf(self, formula: str) -> List[Clause]:
        """转换为CNF"""
        # 简化实现，假设输入已经是CNF形式
        clauses = []
        
        # 解析公式
        parts = formula.split(' & ')
        for part in parts:
            part = part.strip('() ')
            literals = []
            
            # 解析文字
            lit_parts = part.split(' | ')
            for lit_part in lit_parts:
                lit_part = lit_part.strip()
                if lit_part.startswith('~'):
                    literals.append(Literal(lit_part[1:], True))
                else:
                    literals.append(Literal(lit_part, False))
            
            clauses.append(Clause(literals))
        
        return clauses


class ResolutionEngine:
    """归结推理引擎"""
    
    def __init__(self):
        self.clauses = []
    
    def add_clause(self, clause: Clause):
        """添加子句"""
        self.clauses.append(clause)
    
    def resolve(self, clause1: Clause, clause2: Clause) -> Optional[Clause]:
        """对两个子句进行归结"""
        # 寻找可归结的文字对
        for lit1 in clause1.literals:
            for lit2 in clause2.literals:
                if lit1.symbol == lit2.symbol and lit1.negated != lit2.negated:
                    # 找到可归结的文字对
                    new_literals = []
                    new_literals.extend(clause1.literals - {lit1})
                    new_literals.extend(clause2.literals - {lit2})
                    
                    # 去重
                    new_literals = list(set(new_literals))
                    
                    # 检查是否有互补文字
                    for i, l1 in enumerate(new_literals):
                        for j, l2 in enumerate(new_literals):
                            if i != j and l1.symbol == l2.symbol and l1.negated != l2.negated:
                                # 有互补文字，子句为重言式
                                return None
                    
                    return Clause(new_literals)
        
        return None
    
    def prove(self, goal: str) -> bool:
        """证明目标"""
        # 添加目标的否定
        goal_literals = []
        if goal.startswith('~'):
            goal_literals.append(Literal(goal[1:], False))
        else:
            goal_literals.append(Literal(goal, True))
        
        goal_clause = Clause(goal_literals)
        clauses = self.clauses + [goal_clause]
        
        # 归结循环
        new_clauses = []
        
        while True:
            # 生成所有可能的归结
            for i in range(len(clauses)):
                for j in range(i + 1, len(clauses)):
                    resolvent = self.resolve(clauses[i], clauses[j])
                    
                    if resolvent is not None:
                        if resolvent.is_empty():
                            # 找到空子句，证明成功
                            return True
                        
                        new_clauses.append(resolvent)
            
            # 检查是否有新的子句
            if all(clause in clauses for clause in new_clauses):
                # 没有新的子句，证明失败
                return False
            
            # 添加新子句
            clauses.extend(new_clauses)
            new_clauses = []


class WumpusWorld:
    """Wumpus世界"""
    
    def __init__(self, size: int = 4):
        self.size = size
        self.agent_pos = (1, 1)  # 代理位置
        self.wumpus_pos = (3, 1)  # Wumpus位置
        self.gold_pos = (2, 3)  # 金块位置
        self.pits = [(3, 3), (3, 4), (4, 4)]  # 陷阱位置
        
        # 知识库
        self.kb = ResolutionEngine()
        self.init_kb()
    
    def init_kb(self):
        """初始化知识库"""
        # 规则：如果有臭味，则相邻格子有Wumpus
        # 规则：如果有微风，则相邻格子有陷阱
        
        # 添加一些基本规则
        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                # 微风规则
                breeze_clause = Clause([
                    Literal(f"Breeze_{x}_{y}", True),
                    Literal(f"Pit_{x-1}_{y}", False),
                    Literal(f"Pit_{x+1}_{y}", False),
                    Literal(f"Pit_{x}_{y-1}", False),
                    Literal(f"Pit_{x}_{y+1}", False)
                ])
                self.kb.add_clause(breeze_clause)
                
                # 臭味规则
                stench_clause = Clause([
                    Literal(f"Stench_{x}_{y}", True),
                    Literal(f"Wumpus_{x-1}_{y}", False),
                    Literal(f"Wumpus_{x+1}_{y}", False),
                    Literal(f"Wumpus_{x}_{y-1}", False),
                    Literal(f"Wumpus_{x}_{y+1}", False)
                ])
                self.kb.add_clause(stench_clause)
    
    def get_percept(self, x: int, y: int) -> Dict[str, bool]:
        """获取感知"""
        percept = {
            'stench': False,
            'breeze': False,
            'glitter': False,
            'bump': False,
            'scream': False
        }
        
        # 检查臭味
        if self.is_adjacent(x, y, self.wumpus_pos[0], self.wumpus_pos[1]):
            percept['stench'] = True
        
        # 检查微风
        for pit_x, pit_y in self.pits:
            if self.is_adjacent(x, y, pit_x, pit_y):
                percept['breeze'] = True
                break
        
        # 检查金块
        if (x, y) == self.gold_pos:
            percept['glitter'] = True
        
        return percept
    
    def is_adjacent(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """检查两个位置是否相邻"""
        return abs(x1 - x2) + abs(y1 - y2) == 1
    
    def update_kb(self, x: int, y: int, percept: Dict[str, bool]):
        """更新知识库"""
        # 添加感知信息
        if percept['stench']:
            self.kb.add_clause(Clause([Literal(f"Stench_{x}_{y}", False)]))
        else:
            self.kb.add_clause(Clause([Literal(f"Stench_{x}_{y}", True)]))
        
        if percept['breeze']:
            self.kb.add_clause(Clause([Literal(f"Breeze_{x}_{y}", False)]))
        else:
            self.kb.add_clause(Clause([Literal(f"Breeze_{x}_{y}", True)]))
    
    def is_safe(self, x: int, y: int) -> bool:
        """检查位置是否安全"""
        # 检查是否能证明没有陷阱和Wumpus
        no_pit = self.kb.prove(f"~Pit_{x}_{y}")
        no_wumpus = self.kb.prove(f"~Wumpus_{x}_{y}")
        
        return no_pit and no_wumpus


class HornClause:
    """Horn子句"""
    
    def __init__(self, premises: List[str], conclusion: str):
        self.premises = premises
        self.conclusion = conclusion
    
    def __str__(self):
        if not self.premises:
            return self.conclusion
        return f"{' ∧ '.join(self.premises)} → {self.conclusion}"
    
    def __repr__(self):
        return self.__str__()


class ForwardChaining:
    """前向链接推理"""
    
    def __init__(self):
        self.rules = []
        self.facts = set()
    
    def add_rule(self, rule: HornClause):
        """添加规则"""
        self.rules.append(rule)
    
    def add_fact(self, fact: str):
        """添加事实"""
        self.facts.add(fact)
    
    def infer(self, goal: str) -> bool:
        """推理"""
        inferred = set()
        agenda = list(self.facts)
        
        while agenda:
            fact = agenda.pop(0)
            
            if fact == goal:
                return True
            
            if fact in inferred:
                continue
            
            inferred.add(fact)
            
            # 检查所有规则
            for rule in self.rules:
                if rule.conclusion not in inferred:
                    # 检查前提是否满足
                    if all(premise in inferred for premise in rule.premises):
                        agenda.append(rule.conclusion)
        
        return goal in inferred
    
    def get_inference_path(self, goal: str) -> List[str]:
        """获取推理路径"""
        path = []
        inferred = set()
        agenda = list(self.facts)
        
        while agenda:
            fact = agenda.pop(0)
            
            if fact in inferred:
                continue
            
            inferred.add(fact)
            path.append(fact)
            
            if fact == goal:
                return path
            
            # 检查所有规则
            for rule in self.rules:
                if rule.conclusion not in inferred:
                    if all(premise in inferred for premise in rule.premises):
                        agenda.append(rule.conclusion)
        
        return path


class BackwardChaining:
    """后向链接推理"""
    
    def __init__(self):
        self.rules = []
        self.facts = set()
    
    def add_rule(self, rule: HornClause):
        """添加规则"""
        self.rules.append(rule)
    
    def add_fact(self, fact: str):
        """添加事实"""
        self.facts.add(fact)
    
    def prove(self, goal: str, visited: Optional[Set[str]] = None) -> bool:
        """证明目标"""
        if visited is None:
            visited = set()
        
        if goal in visited:
            return False  # 避免循环
        
        if goal in self.facts:
            return True
        
        visited.add(goal)
        
        # 寻找能推出目标的规则
        for rule in self.rules:
            if rule.conclusion == goal:
                # 递归证明所有前提
                if all(self.prove(premise, visited) for premise in rule.premises):
                    return True
        
        visited.remove(goal)
        return False


def demo_propositional_logic():
    """演示命题逻辑"""
    print("命题逻辑演示")
    print("=" * 30)
    
    # 创建命题逻辑处理器
    pl = PropositionalLogic()
    
    # 添加一些子句
    pl.add_clause("P")
    pl.add_clause("P -> Q")
    pl.add_clause("Q -> R")
    
    print(f"知识库包含 {len(pl.clauses)} 个子句")
    print(f"符号: {pl.symbols}")
    
    # 检查可满足性
    print(f"可满足性: {pl.is_satisfiable()}")


def demo_resolution():
    """演示归结推理"""
    print("归结推理演示")
    print("=" * 30)
    
    # 创建归结引擎
    engine = ResolutionEngine()
    
    # 添加知识库
    # 如果下雨，则地面湿润
    engine.add_clause(Clause([
        Literal("Rain", True),
        Literal("WetGround", False)
    ]))
    
    # 如果洒水器开启，则地面湿润
    engine.add_clause(Clause([
        Literal("Sprinkler", True),
        Literal("WetGround", False)
    ]))
    
    # 今天下雨
    engine.add_clause(Clause([
        Literal("Rain", False)
    ]))
    
    # 证明地面湿润
    result = engine.prove("WetGround")
    print(f"能否证明地面湿润: {result}")


def demo_wumpus_world():
    """演示Wumpus世界"""
    print("Wumpus世界演示")
    print("=" * 30)
    
    world = WumpusWorld(4)
    
    # 模拟代理移动
    positions = [(1, 1), (1, 2), (2, 2), (2, 1)]
    
    for x, y in positions:
        percept = world.get_percept(x, y)
        print(f"位置 ({x}, {y}) 的感知: {percept}")
        
        world.update_kb(x, y, percept)
        
        # 检查相邻位置的安全性
        adjacent = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        for adj_x, adj_y in adjacent:
            if 1 <= adj_x <= 4 and 1 <= adj_y <= 4:
                safe = world.is_safe(adj_x, adj_y)
                print(f"位置 ({adj_x}, {adj_y}) 安全: {safe}")
        
        print("-" * 20)


def demo_horn_clauses():
    """演示Horn子句推理"""
    print("Horn子句推理演示")
    print("=" * 30)
    
    # 前向链接
    print("前向链接:")
    fc = ForwardChaining()
    
    # 添加规则
    fc.add_rule(HornClause(["Mammal"], "Animal"))
    fc.add_rule(HornClause(["Human"], "Mammal"))
    fc.add_rule(HornClause(["Animal", "Breathes"], "Alive"))
    
    # 添加事实
    fc.add_fact("Human")
    fc.add_fact("Breathes")
    
    # 推理
    result = fc.infer("Alive")
    print(f"能否推出 'Alive': {result}")
    
    path = fc.get_inference_path("Alive")
    print(f"推理路径: {path}")
    
    # 后向链接
    print("\n后向链接:")
    bc = BackwardChaining()
    
    # 添加规则
    bc.add_rule(HornClause(["Mammal"], "Animal"))
    bc.add_rule(HornClause(["Human"], "Mammal"))
    bc.add_rule(HornClause(["Animal", "Breathes"], "Alive"))
    
    # 添加事实
    bc.add_fact("Human")
    bc.add_fact("Breathes")
    
    # 证明
    result = bc.prove("Alive")
    print(f"能否证明 'Alive': {result}")


if __name__ == "__main__":
    # 演示不同的逻辑推理方法
    demo_propositional_logic()
    print("\n" + "="*50)
    demo_resolution()
    print("\n" + "="*50)
    demo_wumpus_world()
    print("\n" + "="*50)
    demo_horn_clauses()
    
    print("\n✅ 逻辑代理演示完成！") 