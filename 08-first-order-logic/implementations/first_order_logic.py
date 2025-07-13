#!/usr/bin/env python3
"""
第8章：一阶逻辑 (First-Order Logic)

本模块实现了一阶逻辑的核心概念：
- 语法和语义
- 量词处理
- 知识工程
- 逻辑表达式处理
"""

import re
from typing import Dict, List, Set, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

class LogicalConnective(Enum):
    """逻辑连接词"""
    AND = "∧"
    OR = "∨"
    NOT = "¬"
    IMPLIES = "→"
    BICONDITIONAL = "↔"

class Quantifier(Enum):
    """量词"""
    UNIVERSAL = "∀"
    EXISTENTIAL = "∃"

@dataclass
class Term:
    """项 (Term) - 常数、变量或函数"""
    name: str
    args: List['Term'] = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = []
    
    def is_constant(self) -> bool:
        """判断是否为常数"""
        return len(self.args) == 0 and self.name[0].isupper()
    
    def is_variable(self) -> bool:
        """判断是否为变量"""
        return len(self.args) == 0 and self.name[0].islower()
    
    def is_function(self) -> bool:
        """判断是否为函数"""
        return len(self.args) > 0
    
    def get_variables(self) -> Set[str]:
        """获取所有变量"""
        if self.is_variable():
            return {self.name}
        variables = set()
        for arg in self.args:
            variables.update(arg.get_variables())
        return variables
    
    def __str__(self):
        if len(self.args) == 0:
            return self.name
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"

@dataclass 
class Predicate:
    """谓词"""
    name: str
    args: List[Term]
    
    def get_variables(self) -> Set[str]:
        """获取所有变量"""
        variables = set()
        for arg in self.args:
            variables.update(arg.get_variables())
        return variables
    
    def __str__(self):
        if len(self.args) == 0:
            return self.name
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.name}({args_str})"

class Formula:
    """逻辑公式基类"""
    def get_variables(self) -> Set[str]:
        """获取自由变量"""
        raise NotImplementedError
    
    def get_predicates(self) -> Set[str]:
        """获取谓词"""
        raise NotImplementedError
    
    def substitute(self, substitution: Dict[str, Term]) -> 'Formula':
        """变量替换"""
        raise NotImplementedError

class AtomicFormula(Formula):
    """原子公式"""
    def __init__(self, predicate: Predicate):
        self.predicate = predicate
    
    def get_variables(self) -> Set[str]:
        return self.predicate.get_variables()
    
    def get_predicates(self) -> Set[str]:
        return {self.predicate.name}
    
    def substitute(self, substitution: Dict[str, Term]) -> 'AtomicFormula':
        new_args = []
        for arg in self.predicate.args:
            if arg.is_variable() and arg.name in substitution:
                new_args.append(substitution[arg.name])
            else:
                new_args.append(arg)
        return AtomicFormula(Predicate(self.predicate.name, new_args))
    
    def __str__(self):
        return str(self.predicate)

class CompoundFormula(Formula):
    """复合公式"""
    def __init__(self, connective: LogicalConnective, *formulas: Formula):
        self.connective = connective
        self.formulas = list(formulas)
    
    def get_variables(self) -> Set[str]:
        variables = set()
        for formula in self.formulas:
            variables.update(formula.get_variables())
        return variables
    
    def get_predicates(self) -> Set[str]:
        predicates = set()
        for formula in self.formulas:
            predicates.update(formula.get_predicates())
        return predicates
    
    def substitute(self, substitution: Dict[str, Term]) -> 'CompoundFormula':
        new_formulas = [f.substitute(substitution) for f in self.formulas]
        return CompoundFormula(self.connective, *new_formulas)
    
    def __str__(self):
        if self.connective == LogicalConnective.NOT:
            return f"¬{self.formulas[0]}"
        elif len(self.formulas) == 2:
            return f"({self.formulas[0]} {self.connective.value} {self.formulas[1]})"
        else:
            formula_strs = [str(f) for f in self.formulas]
            return f"({f' {self.connective.value} '.join(formula_strs)})"

class QuantifiedFormula(Formula):
    """量化公式"""
    def __init__(self, quantifier: Quantifier, variable: str, formula: Formula):
        self.quantifier = quantifier
        self.variable = variable
        self.formula = formula
    
    def get_variables(self) -> Set[str]:
        """获取自由变量（排除被量化的变量）"""
        variables = self.formula.get_variables()
        variables.discard(self.variable)
        return variables
    
    def get_predicates(self) -> Set[str]:
        return self.formula.get_predicates()
    
    def substitute(self, substitution: Dict[str, Term]) -> 'QuantifiedFormula':
        # 避免变量捕获
        new_substitution = {k: v for k, v in substitution.items() if k != self.variable}
        new_formula = self.formula.substitute(new_substitution)
        return QuantifiedFormula(self.quantifier, self.variable, new_formula)
    
    def __str__(self):
        return f"{self.quantifier.value}{self.variable} {self.formula}"

class KnowledgeBase:
    """知识库"""
    def __init__(self):
        self.formulas: List[Formula] = []
        self.constants: Set[str] = set()
        self.predicates: Set[str] = set()
    
    def tell(self, formula: Formula):
        """向知识库添加知识"""
        self.formulas.append(formula)
        self.predicates.update(formula.get_predicates())
    
    def ask(self, query: Formula) -> bool:
        """查询知识库（简单实现）"""
        # 这里只是一个简化的实现
        for formula in self.formulas:
            if str(formula) == str(query):
                return True
        return False
    
    def get_all_predicates(self) -> Set[str]:
        """获取所有谓词"""
        return self.predicates.copy()
    
    def get_formulas_with_predicate(self, predicate_name: str) -> List[Formula]:
        """获取包含特定谓词的公式"""
        result = []
        for formula in self.formulas:
            if predicate_name in formula.get_predicates():
                result.append(formula)
        return result

class FirstOrderLogicParser:
    """一阶逻辑解析器"""
    
    @staticmethod
    def parse_term(term_str: str) -> Term:
        """解析项"""
        term_str = term_str.strip()
        
        # 检查是否为函数调用
        if '(' in term_str:
            name = term_str[:term_str.index('(')]
            args_str = term_str[term_str.index('(')+1:term_str.rindex(')')]
            args = []
            if args_str.strip():
                for arg_str in args_str.split(','):
                    args.append(FirstOrderLogicParser.parse_term(arg_str.strip()))
            return Term(name, args)
        else:
            return Term(term_str)
    
    @staticmethod
    def parse_predicate(pred_str: str) -> Predicate:
        """解析谓词"""
        pred_str = pred_str.strip()
        
        if '(' in pred_str:
            name = pred_str[:pred_str.index('(')]
            args_str = pred_str[pred_str.index('(')+1:pred_str.rindex(')')]
            args = []
            if args_str.strip():
                for arg_str in args_str.split(','):
                    args.append(FirstOrderLogicParser.parse_term(arg_str.strip()))
            return Predicate(name, args)
        else:
            return Predicate(pred_str, [])

class FirstOrderLogicDemo:
    """一阶逻辑演示"""
    
    def __init__(self):
        self.kb = KnowledgeBase()
        self.setup_examples()
    
    def setup_examples(self):
        """设置示例知识"""
        print("设置示例知识库...")
        
        # 创建一些项
        john = Term("John")
        mary = Term("Mary")
        mother_john = Term("Mother", [john])
        
        # 创建一些谓词
        human_john = Predicate("Human", [john])
        human_mary = Predicate("Human", [mary])
        parent_relation = Predicate("Parent", [mother_john, john])
        
        # 创建公式
        formula1 = AtomicFormula(human_john)
        formula2 = AtomicFormula(human_mary)
        formula3 = AtomicFormula(parent_relation)
        
        # 添加到知识库
        self.kb.tell(formula1)
        self.kb.tell(formula2)
        self.kb.tell(formula3)
        
        # 创建量化公式：∀x (Human(x) → Mortal(x))
        x = Term("x")
        human_x = AtomicFormula(Predicate("Human", [x]))
        mortal_x = AtomicFormula(Predicate("Mortal", [x]))
        implies_formula = CompoundFormula(LogicalConnective.IMPLIES, human_x, mortal_x)
        universal_formula = QuantifiedFormula(Quantifier.UNIVERSAL, "x", implies_formula)
        
        self.kb.tell(universal_formula)
        
        print("知识库设置完成！")
    
    def demonstrate_syntax(self):
        """演示语法"""
        print("\n" + "="*50)
        print("一阶逻辑语法演示")
        print("="*50)
        
        # 项的示例
        print("\n1. 项 (Terms):")
        const = Term("John")
        var = Term("x")
        func = Term("Father", [Term("John")])
        
        print(f"   常数: {const} (是常数: {const.is_constant()})")
        print(f"   变量: {var} (是变量: {var.is_variable()})")
        print(f"   函数: {func} (是函数: {func.is_function()})")
        
        # 谓词的示例
        print("\n2. 谓词 (Predicates):")
        pred1 = Predicate("Human", [Term("John")])
        pred2 = Predicate("Loves", [Term("John"), Term("Mary")])
        
        print(f"   一元谓词: {pred1}")
        print(f"   二元谓词: {pred2}")
        
        # 公式的示例
        print("\n3. 公式 (Formulas):")
        atomic = AtomicFormula(pred1)
        compound = CompoundFormula(LogicalConnective.AND, 
                                 AtomicFormula(pred1), 
                                 AtomicFormula(pred2))
        quantified = QuantifiedFormula(Quantifier.UNIVERSAL, "x", 
                                     AtomicFormula(Predicate("Human", [Term("x")])))
        
        print(f"   原子公式: {atomic}")
        print(f"   复合公式: {compound}")
        print(f"   量化公式: {quantified}")
    
    def demonstrate_semantics(self):
        """演示语义"""
        print("\n" + "="*50)
        print("一阶逻辑语义演示")
        print("="*50)
        
        # 领域和解释
        print("\n1. 领域和解释:")
        print("   领域 D = {john, mary, ann}")
        print("   解释:")
        print("     John → john")
        print("     Mary → mary")
        print("     Human(x) → {john, mary}")
        print("     Loves(x,y) → {(john,mary), (mary,john)}")
        
        # 真值评估
        print("\n2. 真值评估:")
        formulas = [
            "Human(John)",
            "Human(Mary)",
            "Loves(John, Mary)",
            "Loves(Mary, Ann)",
            "∀x Human(x)",
            "∃x Loves(x, Mary)"
        ]
        
        truth_values = [True, True, True, False, False, True]
        
        for formula, truth in zip(formulas, truth_values):
            status = "真" if truth else "假"
            print(f"   {formula:<20} : {status}")
    
    def demonstrate_unification(self):
        """演示合一算法"""
        print("\n" + "="*50)
        print("合一算法演示")
        print("="*50)
        
        print("\n合一 (Unification) 是一阶逻辑推理的核心算法")
        print("目标：找到使两个表达式相同的替换")
        
        # 示例合一
        examples = [
            ("P(x)", "P(John)", {"x": "John"}),
            ("P(x, f(y))", "P(a, f(b))", {"x": "a", "y": "b"}),
            ("P(x, x)", "P(a, b)", "失败 - 变量x不能同时匹配a和b"),
            ("Q(f(x), y)", "Q(f(a), b)", {"x": "a", "y": "b"})
        ]
        
        print("\n合一示例:")
        for expr1, expr2, result in examples:
            if isinstance(result, dict):
                subst_str = ", ".join(f"{k}→{v}" for k, v in result.items())
                print(f"   {expr1:<15} 与 {expr2:<15} : {{{subst_str}}}")
            else:
                print(f"   {expr1:<15} 与 {expr2:<15} : {result}")
    
    def demonstrate_inference(self):
        """演示推理"""
        print("\n" + "="*50)
        print("一阶逻辑推理演示")
        print("="*50)
        
        print("\n当前知识库内容:")
        for i, formula in enumerate(self.kb.formulas, 1):
            print(f"   {i}. {formula}")
        
        print("\n推理示例:")
        print("   知识: Human(John), ∀x (Human(x) → Mortal(x))")
        print("   结论: Mortal(John)")
        print("   推理方法: 全称实例化 + 肯定前件")
        
        # 简单的推理链
        print("\n推理步骤:")
        print("   1. Human(John)                    [已知]")
        print("   2. ∀x (Human(x) → Mortal(x))      [已知]")
        print("   3. Human(John) → Mortal(John)     [全称实例化, 2]")
        print("   4. Mortal(John)                   [肯定前件, 1,3]")

def demo_family_relationships():
    """演示家庭关系知识表示"""
    print("\n" + "="*60)
    print("家庭关系知识表示示例")
    print("="*60)
    
    kb = KnowledgeBase()
    
    # 添加家庭关系知识
    print("\n添加家庭关系知识:")
    
    # 事实
    facts = [
        "Parent(John, Mary)",
        "Parent(John, Tom)", 
        "Parent(Mary, Ann)",
        "Male(John)",
        "Male(Tom)",
        "Female(Mary)",
        "Female(Ann)"
    ]
    
    for fact in facts:
        print(f"   {fact}")
        # 这里简化为字符串，实际应该解析为Formula对象
    
    # 规则
    print("\n添加规则:")
    rules = [
        "∀x,y (Parent(x,y) ∧ Male(x) → Father(x,y))",
        "∀x,y (Parent(x,y) ∧ Female(x) → Mother(x,y))",
        "∀x,y (Parent(x,y) → Child(y,x))",
        "∀x,y,z (Parent(x,y) ∧ Parent(y,z) → Grandparent(x,z))"
    ]
    
    for rule in rules:
        print(f"   {rule}")
    
    # 查询示例
    print("\n查询示例:")
    queries = [
        "Father(John, Mary)",
        "Mother(Mary, Ann)",
        "Grandparent(John, Ann)",
        "∃x Father(x, Tom)"
    ]
    
    for query in queries:
        print(f"   查询: {query}")
        print(f"   结果: 真 (基于推理)")

def run_comprehensive_demo():
    """运行完整演示"""
    print("🧠 第8章：一阶逻辑 - 完整演示")
    print("="*60)
    
    demo = FirstOrderLogicDemo()
    
    # 运行各个演示
    demo.demonstrate_syntax()
    demo.demonstrate_semantics()
    demo.demonstrate_unification()
    demo.demonstrate_inference()
    demo_family_relationships()
    
    print("\n" + "="*60)
    print("一阶逻辑演示完成！")
    print("="*60)
    print("\n📚 学习要点:")
    print("• 一阶逻辑扩展了命题逻辑，支持对象、属性和关系")
    print("• 量词（∀, ∃）允许表达关于所有或某些对象的陈述")
    print("• 合一算法是一阶逻辑推理的核心")
    print("• 一阶逻辑为知识表示提供了强大的框架")

if __name__ == "__main__":
    run_comprehensive_demo() 