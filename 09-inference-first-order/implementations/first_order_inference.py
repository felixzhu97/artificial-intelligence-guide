#!/usr/bin/env python3
"""
第9章：一阶逻辑推理 (First-Order Logic Inference)

本模块实现了一阶逻辑的推理算法：
- 归结推理 (Resolution)
- 前向链接 (Forward Chaining)
- 后向链接 (Backward Chaining)
- 合一算法 (Unification)
"""

import sys
import os
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass
from copy import deepcopy

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from implementations.first_order_logic import (
        Term, Predicate, Formula, AtomicFormula, CompoundFormula, 
        QuantifiedFormula, LogicalConnective, Quantifier, KnowledgeBase
    )
except ImportError:
    print("警告：无法导入一阶逻辑模块，将使用简化版本")

@dataclass
class Clause:
    """子句 - 文字的析取"""
    literals: List[AtomicFormula]
    
    def is_empty(self) -> bool:
        """判断是否为空子句"""
        return len(self.literals) == 0
    
    def get_variables(self) -> Set[str]:
        """获取所有变量"""
        variables = set()
        for literal in self.literals:
            variables.update(literal.get_variables())
        return variables
    
    def __str__(self):
        if self.is_empty():
            return "□"  # 空子句
        return " ∨ ".join(str(lit) for lit in self.literals)

class Substitution:
    """替换（合一结果）"""
    def __init__(self, mapping: Dict[str, Term] = None):
        self.mapping = mapping or {}
    
    def apply(self, term: Term) -> Term:
        """应用替换到项"""
        if term.is_variable() and term.name in self.mapping:
            return self.mapping[term.name]
        elif term.is_function():
            new_args = [self.apply(arg) for arg in term.args]
            return Term(term.name, new_args)
        else:
            return term
    
    def apply_to_formula(self, formula: AtomicFormula) -> AtomicFormula:
        """应用替换到原子公式"""
        new_args = [self.apply(arg) for arg in formula.predicate.args]
        return AtomicFormula(Predicate(formula.predicate.name, new_args))
    
    def compose(self, other: 'Substitution') -> 'Substitution':
        """合成两个替换"""
        new_mapping = {}
        
        # 应用other到self的值
        for var, term in self.mapping.items():
            new_term = term
            for other_var, other_term in other.mapping.items():
                new_term = self._substitute_in_term(new_term, other_var, other_term)
            new_mapping[var] = new_term
        
        # 添加other中不在self中的映射
        for var, term in other.mapping.items():
            if var not in new_mapping:
                new_mapping[var] = term
        
        return Substitution(new_mapping)
    
    def _substitute_in_term(self, term: Term, var: str, replacement: Term) -> Term:
        """在项中替换变量"""
        if term.is_variable() and term.name == var:
            return replacement
        elif term.is_function():
            new_args = [self._substitute_in_term(arg, var, replacement) for arg in term.args]
            return Term(term.name, new_args)
        else:
            return term
    
    def __str__(self):
        if not self.mapping:
            return "{}"
        items = [f"{var} → {term}" for var, term in self.mapping.items()]
        return "{" + ", ".join(items) + "}"

class UnificationEngine:
    """合一算法引擎"""
    
    @staticmethod
    def unify(term1: Term, term2: Term) -> Optional[Substitution]:
        """合一两个项"""
        return UnificationEngine._unify_recursive(term1, term2, Substitution())
    
    @staticmethod
    def _unify_recursive(term1: Term, term2: Term, subst: Substitution) -> Optional[Substitution]:
        """递归合一算法"""
        # 应用当前替换
        term1 = subst.apply(term1)
        term2 = subst.apply(term2)
        
        # 如果两项相同
        if term1.name == term2.name and len(term1.args) == len(term2.args) == 0:
            return subst
        
        # 如果term1是变量
        if term1.is_variable():
            return UnificationEngine._unify_variable(term1.name, term2, subst)
        
        # 如果term2是变量
        if term2.is_variable():
            return UnificationEngine._unify_variable(term2.name, term1, subst)
        
        # 如果都是函数且函数名相同
        if (term1.is_function() and term2.is_function() and 
            term1.name == term2.name and len(term1.args) == len(term2.args)):
            
            for arg1, arg2 in zip(term1.args, term2.args):
                subst = UnificationEngine._unify_recursive(arg1, arg2, subst)
                if subst is None:
                    return None
            return subst
        
        # 无法合一
        return None
    
    @staticmethod
    def _unify_variable(var: str, term: Term, subst: Substitution) -> Optional[Substitution]:
        """变量合一"""
        # 检查变量是否已有绑定
        if var in subst.mapping:
            return UnificationEngine._unify_recursive(subst.mapping[var], term, subst)
        
        # 检查term是否为变量且已有绑定
        if term.is_variable() and term.name in subst.mapping:
            return UnificationEngine._unify_recursive(Term(var), subst.mapping[term.name], subst)
        
        # 发生检查（避免无限循环）
        if UnificationEngine._occurs_check(var, term):
            return None
        
        # 创建新的替换
        new_mapping = subst.mapping.copy()
        new_mapping[var] = term
        return Substitution(new_mapping)
    
    @staticmethod
    def _occurs_check(var: str, term: Term) -> bool:
        """发生检查：变量是否出现在项中"""
        if term.is_variable():
            return term.name == var
        elif term.is_function():
            return any(UnificationEngine._occurs_check(var, arg) for arg in term.args)
        else:
            return False

class ResolutionEngine:
    """归结推理引擎"""
    
    def __init__(self):
        self.clauses: List[Clause] = []
    
    def add_clause(self, clause: Clause):
        """添加子句"""
        self.clauses.append(clause)
    
    def resolve(self, clause1: Clause, clause2: Clause) -> List[Clause]:
        """归结两个子句"""
        resolvents = []
        
        for i, lit1 in enumerate(clause1.literals):
            for j, lit2 in enumerate(clause2.literals):
                # 尝试合一互补文字
                if self._can_resolve(lit1, lit2):
                    # 标准化变量名
                    std_clause1 = self._standardize_variables(clause1, "1")
                    std_clause2 = self._standardize_variables(clause2, "2")
                    
                    std_lit1 = std_clause1.literals[i]
                    std_lit2 = std_clause2.literals[j]
                    
                    # 合一
                    subst = UnificationEngine.unify(
                        Term(std_lit1.predicate.name, std_lit1.predicate.args),
                        Term(std_lit2.predicate.name, std_lit2.predicate.args)
                    )
                    
                    if subst is not None:
                        # 构造归结式
                        new_literals = []
                        
                        # 添加clause1中除lit1外的文字
                        for k, lit in enumerate(std_clause1.literals):
                            if k != i:
                                new_literals.append(subst.apply_to_formula(lit))
                        
                        # 添加clause2中除lit2外的文字
                        for k, lit in enumerate(std_clause2.literals):
                            if k != j:
                                new_literals.append(subst.apply_to_formula(lit))
                        
                        # 去重
                        unique_literals = []
                        for lit in new_literals:
                            if not any(str(lit) == str(existing) for existing in unique_literals):
                                unique_literals.append(lit)
                        
                        resolvents.append(Clause(unique_literals))
        
        return resolvents
    
    def _can_resolve(self, lit1: AtomicFormula, lit2: AtomicFormula) -> bool:
        """检查两个文字是否可以归结"""
        # 简化：假设一个是正文字，一个是负文字（通过谓词名判断）
        return lit1.predicate.name == lit2.predicate.name
    
    def _standardize_variables(self, clause: Clause, suffix: str) -> Clause:
        """标准化变量名"""
        var_mapping = {}
        new_literals = []
        
        for literal in clause.literals:
            new_args = []
            for arg in literal.predicate.args:
                new_arg = self._rename_variables_in_term(arg, suffix, var_mapping)
                new_args.append(new_arg)
            new_literals.append(AtomicFormula(Predicate(literal.predicate.name, new_args)))
        
        return Clause(new_literals)
    
    def _rename_variables_in_term(self, term: Term, suffix: str, var_mapping: Dict[str, str]) -> Term:
        """重命名项中的变量"""
        if term.is_variable():
            if term.name not in var_mapping:
                var_mapping[term.name] = f"{term.name}_{suffix}"
            return Term(var_mapping[term.name])
        elif term.is_function():
            new_args = [self._rename_variables_in_term(arg, suffix, var_mapping) for arg in term.args]
            return Term(term.name, new_args)
        else:
            return term
    
    def prove_by_contradiction(self, query: AtomicFormula) -> bool:
        """使用反证法证明查询"""
        print(f"\n尝试证明: {query}")
        
        # 添加查询的否定
        negated_query = self._negate_formula(query)
        self.add_clause(Clause([negated_query]))
        
        print(f"添加否定: {negated_query}")
        
        # 归结循环
        new_clauses = []
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n第{iteration}轮归结:")
            
            n = len(self.clauses)
            found_new = False
            
            # 尝试归结所有子句对
            for i in range(n):
                for j in range(i + 1, n):
                    resolvents = self.resolve(self.clauses[i], self.clauses[j])
                    
                    for resolvent in resolvents:
                        print(f"  {self.clauses[i]} ⊗ {self.clauses[j]} = {resolvent}")
                        
                        if resolvent.is_empty():
                            print("  得到空子句 □ - 证明成功！")
                            return True
                        
                        # 检查是否为新子句
                        if not any(str(resolvent) == str(existing) for existing in self.clauses + new_clauses):
                            new_clauses.append(resolvent)
                            found_new = True
            
            if not found_new:
                print("  无法生成新子句 - 证明失败")
                return False
            
            # 添加新子句到知识库
            self.clauses.extend(new_clauses)
            new_clauses = []
            
            # 防止无限循环
            if iteration > 10:
                print("  达到最大迭代次数")
                return False
    
    def _negate_formula(self, formula: AtomicFormula) -> AtomicFormula:
        """否定公式（简化实现）"""
        # 这里简化为在谓词名前加¬
        neg_name = f"¬{formula.predicate.name}"
        return AtomicFormula(Predicate(neg_name, formula.predicate.args))

class ForwardChaining:
    """前向链接推理"""
    
    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Tuple[List[str], str]] = []
    
    def add_fact(self, fact: str):
        """添加事实"""
        self.facts.add(fact)
    
    def add_rule(self, premises: List[str], conclusion: str):
        """添加规则"""
        self.rules.append((premises, conclusion))
    
    def infer(self, max_iterations: int = 100) -> Set[str]:
        """前向链接推理"""
        print("\n前向链接推理过程:")
        print(f"初始事实: {self.facts}")
        
        new_facts = set()
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n第{iteration}轮推理:")
            
            added_facts = False
            
            for premises, conclusion in self.rules:
                # 检查所有前提是否都满足
                if all(premise in self.facts or premise in new_facts for premise in premises):
                    if conclusion not in self.facts and conclusion not in new_facts:
                        new_facts.add(conclusion)
                        print(f"  应用规则: {' ∧ '.join(premises)} → {conclusion}")
                        added_facts = True
            
            if not added_facts:
                print("  无新事实可推导")
                break
            
            # 将新事实加入事实库
            self.facts.update(new_facts)
            new_facts.clear()
        
        print(f"\n最终事实库: {self.facts}")
        return self.facts
    
    def query(self, goal: str) -> bool:
        """查询目标是否可推导"""
        self.infer()
        return goal in self.facts

class BackwardChaining:
    """后向链接推理"""
    
    def __init__(self):
        self.facts: Set[str] = set()
        self.rules: List[Tuple[List[str], str]] = []
    
    def add_fact(self, fact: str):
        """添加事实"""
        self.facts.add(fact)
    
    def add_rule(self, premises: List[str], conclusion: str):
        """添加规则"""
        self.rules.append((premises, conclusion))
    
    def prove(self, goal: str, depth: int = 0) -> bool:
        """后向链接证明"""
        indent = "  " * depth
        print(f"{indent}尝试证明: {goal}")
        
        # 检查是否为已知事实
        if goal in self.facts:
            print(f"{indent}✓ {goal} 是已知事实")
            return True
        
        # 寻找能推导出目标的规则
        for premises, conclusion in self.rules:
            if conclusion == goal:
                print(f"{indent}找到规则: {' ∧ '.join(premises)} → {conclusion}")
                
                # 递归证明所有前提
                all_premises_proved = True
                for premise in premises:
                    if not self.prove(premise, depth + 1):
                        all_premises_proved = False
                        break
                
                if all_premises_proved:
                    print(f"{indent}✓ 成功证明 {goal}")
                    return True
                else:
                    print(f"{indent}✗ 无法证明所有前提")
        
        print(f"{indent}✗ 无法证明 {goal}")
        return False

def demo_unification():
    """演示合一算法"""
    print("\n" + "="*50)
    print("合一算法演示")
    print("="*50)
    
    # 测试用例
    test_cases = [
        (Term("x"), Term("John")),
        (Term("f", [Term("x")]), Term("f", [Term("a")])),
        (Term("f", [Term("x"), Term("y")]), Term("f", [Term("a"), Term("b")])),
        (Term("x"), Term("f", [Term("x")])),  # 发生检查失败
    ]
    
    for term1, term2 in test_cases:
        print(f"\n合一 {term1} 和 {term2}:")
        result = UnificationEngine.unify(term1, term2)
        if result:
            print(f"  成功: {result}")
        else:
            print(f"  失败")

def demo_resolution():
    """演示归结推理"""
    print("\n" + "="*50)
    print("归结推理演示")
    print("="*50)
    
    engine = ResolutionEngine()
    
    # 添加知识库
    print("\n知识库:")
    
    # Human(John)
    human_john = AtomicFormula(Predicate("Human", [Term("John")]))
    engine.add_clause(Clause([human_john]))
    print(f"  {human_john}")
    
    # ¬Human(x) ∨ Mortal(x)  (即 Human(x) → Mortal(x))
    not_human_x = AtomicFormula(Predicate("¬Human", [Term("x")]))
    mortal_x = AtomicFormula(Predicate("Mortal", [Term("x")]))
    engine.add_clause(Clause([not_human_x, mortal_x]))
    print(f"  {not_human_x} ∨ {mortal_x}")
    
    # 查询: Mortal(John)
    query = AtomicFormula(Predicate("Mortal", [Term("John")]))
    result = engine.prove_by_contradiction(query)
    
    print(f"\n查询结果: {result}")

def demo_forward_chaining():
    """演示前向链接"""
    print("\n" + "="*50)
    print("前向链接推理演示")
    print("="*50)
    
    fc = ForwardChaining()
    
    # 添加事实
    fc.add_fact("Human(John)")
    fc.add_fact("Human(Mary)")
    
    # 添加规则
    fc.add_rule(["Human(x)"], "Mortal(x)")
    fc.add_rule(["Mortal(x)"], "CanDie(x)")
    fc.add_rule(["Human(x)", "Human(y)"], "SameSpecies(x,y)")
    
    # 查询
    result = fc.query("CanDie(John)")
    print(f"\n查询 CanDie(John): {result}")

def demo_backward_chaining():
    """演示后向链接"""
    print("\n" + "="*50)
    print("后向链接推理演示")
    print("="*50)
    
    bc = BackwardChaining()
    
    # 添加事实
    bc.add_fact("Human(John)")
    bc.add_fact("Human(Mary)")
    
    # 添加规则
    bc.add_rule(["Human(x)"], "Mortal(x)")
    bc.add_rule(["Mortal(x)"], "CanDie(x)")
    
    # 查询
    print("\n后向链接证明过程:")
    result = bc.prove("CanDie(John)")
    print(f"\n查询结果: {result}")

def run_comprehensive_demo():
    """运行完整演示"""
    print("🧠 第9章：一阶逻辑推理 - 完整演示")
    print("="*60)
    
    # 运行各个演示
    demo_unification()
    demo_resolution()
    demo_forward_chaining()
    demo_backward_chaining()
    
    print("\n" + "="*60)
    print("一阶逻辑推理演示完成！")
    print("="*60)
    print("\n📚 学习要点:")
    print("• 合一算法是一阶逻辑推理的基础")
    print("• 归结推理通过反证法进行定理证明")
    print("• 前向链接从事实推导结论")
    print("• 后向链接从目标倒推前提")
    print("• 不同推理策略适用于不同的问题类型")

if __name__ == "__main__":
    run_comprehensive_demo() 