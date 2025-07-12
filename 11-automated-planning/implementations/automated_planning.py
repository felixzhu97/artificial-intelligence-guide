"""
第11章：自动规划
实现了AIMA第11章中的自动规划算法：STRIPS、状态空间规划、部分排序规划等
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Set, Optional, Any, Callable, Union
import copy
from collections import deque, defaultdict
from dataclasses import dataclass
import itertools
from enum import Enum


class Predicate:
    """谓词类：表示规划中的原子命题"""
    
    def __init__(self, name: str, args: List[str] = None):
        self.name = name
        self.args = args or []
    
    def __str__(self):
        if self.args:
            return f"{self.name}({', '.join(self.args)})"
        return self.name
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return isinstance(other, Predicate) and self.name == other.name and self.args == other.args
    
    def __hash__(self):
        return hash((self.name, tuple(self.args)))


class State:
    """状态类：表示世界的一个状态"""
    
    def __init__(self, predicates: Set[Predicate] = None):
        self.predicates = predicates or set()
    
    def add(self, predicate: Predicate):
        """添加谓词到状态"""
        self.predicates.add(predicate)
    
    def remove(self, predicate: Predicate):
        """从状态中删除谓词"""
        self.predicates.discard(predicate)
    
    def contains(self, predicate: Predicate) -> bool:
        """检查状态是否包含谓词"""
        return predicate in self.predicates
    
    def satisfies(self, conditions: Set[Predicate]) -> bool:
        """检查状态是否满足条件集合"""
        return conditions.issubset(self.predicates)
    
    def copy(self) -> 'State':
        """复制状态"""
        return State(self.predicates.copy())
    
    def __str__(self):
        return "{" + ", ".join(str(p) for p in sorted(self.predicates, key=str)) + "}"
    
    def __repr__(self):
        return f"State({self.predicates})"
    
    def __eq__(self, other):
        return isinstance(other, State) and self.predicates == other.predicates
    
    def __hash__(self):
        return hash(frozenset(self.predicates))


class Action:
    """动作类：STRIPS风格的动作定义"""
    
    def __init__(self, name: str, parameters: List[str] = None,
                 preconditions: Set[Predicate] = None,
                 add_effects: Set[Predicate] = None,
                 delete_effects: Set[Predicate] = None):
        self.name = name
        self.parameters = parameters or []
        self.preconditions = preconditions or set()
        self.add_effects = add_effects or set()
        self.delete_effects = delete_effects or set()
    
    def is_applicable(self, state: State) -> bool:
        """检查动作是否可以在给定状态执行"""
        return state.satisfies(self.preconditions)
    
    def apply(self, state: State) -> State:
        """在状态上执行动作，返回新状态"""
        if not self.is_applicable(state):
            raise ValueError(f"Action {self.name} is not applicable in state {state}")
        
        new_state = state.copy()
        
        # 删除效果
        for pred in self.delete_effects:
            new_state.remove(pred)
        
        # 添加效果
        for pred in self.add_effects:
            new_state.add(pred)
        
        return new_state
    
    def __str__(self):
        return f"{self.name}({', '.join(self.parameters)})"
    
    def __repr__(self):
        return f"Action({self.name}, pre={self.preconditions}, add={self.add_effects}, del={self.delete_effects})"


class Plan:
    """规划解：动作序列"""
    
    def __init__(self, actions: List[Action] = None):
        self.actions = actions or []
    
    def add_action(self, action: Action):
        """添加动作到规划"""
        self.actions.append(action)
    
    def execute(self, initial_state: State) -> State:
        """执行规划，返回最终状态"""
        current_state = initial_state.copy()
        for action in self.actions:
            current_state = action.apply(current_state)
        return current_state
    
    def is_valid(self, initial_state: State, goal: Set[Predicate]) -> bool:
        """检查规划是否有效"""
        try:
            final_state = self.execute(initial_state)
            return final_state.satisfies(goal)
        except ValueError:
            return False
    
    def __len__(self):
        return len(self.actions)
    
    def __str__(self):
        return " -> ".join(str(action) for action in self.actions)
    
    def __repr__(self):
        return f"Plan({self.actions})"


class PlanningProblem:
    """规划问题定义"""
    
    def __init__(self, initial_state: State, goal: Set[Predicate], actions: List[Action]):
        self.initial_state = initial_state
        self.goal = goal
        self.actions = actions
    
    def is_goal(self, state: State) -> bool:
        """检查状态是否满足目标"""
        return state.satisfies(self.goal)
    
    def applicable_actions(self, state: State) -> List[Action]:
        """返回在给定状态下可执行的动作"""
        return [action for action in self.actions if action.is_applicable(state)]


class ForwardSearchPlanner:
    """前向搜索规划器：从初始状态向目标搜索"""
    
    def plan(self, problem: PlanningProblem) -> Optional[Plan]:
        """使用广度优先搜索进行规划"""
        frontier = deque([(problem.initial_state, Plan())])
        explored = set()
        
        while frontier:
            current_state, current_plan = frontier.popleft()
            
            if problem.is_goal(current_state):
                return current_plan
            
            state_hash = hash(current_state)
            if state_hash in explored:
                continue
            explored.add(state_hash)
            
            for action in problem.applicable_actions(current_state):
                try:
                    new_state = action.apply(current_state)
                    new_plan = Plan(current_plan.actions + [action])
                    frontier.append((new_state, new_plan))
                except ValueError:
                    continue
        
        return None


class BackwardSearchPlanner:
    """后向搜索规划器：从目标向初始状态搜索"""
    
    def plan(self, problem: PlanningProblem) -> Optional[Plan]:
        """使用后向搜索进行规划"""
        # 简化实现：使用目标回归
        frontier = deque([problem.goal])
        plan_actions = []
        
        current_goals = problem.goal.copy()
        
        max_depth = 10  # 限制搜索深度
        depth = 0
        
        while current_goals and depth < max_depth:
            depth += 1
            
            # 找到一个能够实现当前目标的动作
            applicable_action = None
            for action in problem.actions:
                if current_goals.intersection(action.add_effects):
                    applicable_action = action
                    break
            
            if applicable_action is None:
                break
            
            plan_actions.insert(0, applicable_action)
            
            # 更新目标：移除已实现的，添加前提条件
            current_goals = current_goals.difference(applicable_action.add_effects)
            current_goals = current_goals.union(applicable_action.preconditions)
        
        # 检查初始状态是否满足最终的前提条件
        if problem.initial_state.satisfies(current_goals):
            return Plan(plan_actions)
        
        return None


@dataclass
class PartialOrder:
    """部分排序约束"""
    before: Action
    after: Action


class PartialOrderPlan:
    """部分排序规划"""
    
    def __init__(self):
        self.actions = set()
        self.orderings = set()  # 排序约束
        self.causal_links = set()  # 因果链接
        self.open_conditions = set()  # 开放条件
    
    def add_action(self, action: Action):
        """添加动作"""
        self.actions.add(action)
        self.open_conditions.update(action.preconditions)
    
    def add_ordering(self, before: Action, after: Action):
        """添加排序约束"""
        self.orderings.add(PartialOrder(before, after))
    
    def is_consistent(self) -> bool:
        """检查规划是否一致（无循环）"""
        # 简化实现：检查是否有直接的循环冲突
        for order in self.orderings:
            reverse_order = PartialOrder(order.after, order.before)
            if reverse_order in self.orderings:
                return False
        return True
    
    def linearize(self) -> List[Action]:
        """将部分排序规划线性化"""
        # 拓扑排序
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        for action in self.actions:
            in_degree[action] = 0
        
        for order in self.orderings:
            graph[order.before].append(order.after)
            in_degree[order.after] += 1
        
        queue = deque([action for action in self.actions if in_degree[action] == 0])
        result = []
        
        while queue:
            action = queue.popleft()
            result.append(action)
            
            for neighbor in graph[action]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == len(self.actions) else []


class BlocksWorldProblem:
    """积木世界问题：经典的规划测试域"""
    
    @staticmethod
    def create_problem(initial_config: List[List[str]], goal_config: List[List[str]]) -> PlanningProblem:
        """创建积木世界规划问题"""
        
        # 创建初始状态
        initial_state = State()
        for stack in initial_config:
            for i, block in enumerate(stack):
                if i == 0:
                    initial_state.add(Predicate("on_table", [block]))
                else:
                    initial_state.add(Predicate("on", [block, stack[i-1]]))
                initial_state.add(Predicate("clear", [block]))
        
        # 移除顶部积木的clear标记
        for stack in initial_config:
            if len(stack) > 1:
                for i in range(len(stack) - 1):
                    initial_state.remove(Predicate("clear", [stack[i]]))
        
        # 创建目标状态
        goal = set()
        for stack in goal_config:
            for i, block in enumerate(stack):
                if i == 0:
                    goal.add(Predicate("on_table", [block]))
                else:
                    goal.add(Predicate("on", [block, stack[i-1]]))
        
        # 创建动作
        blocks = set()
        for stack in initial_config + goal_config:
            blocks.update(stack)
        
        actions = []
        
        # 移动积木到另一个积木上
        for block1 in blocks:
            for block2 in blocks:
                if block1 != block2:
                    action = Action(
                        name=f"move_{block1}_to_{block2}",
                        parameters=[block1, block2],
                        preconditions={
                            Predicate("clear", [block1]),
                            Predicate("clear", [block2])
                        },
                        add_effects={
                            Predicate("on", [block1, block2]),
                            Predicate("clear", [block1])
                        },
                        delete_effects={
                            Predicate("clear", [block2])
                        }
                    )
                    actions.append(action)
        
        # 移动积木到桌子上
        for block in blocks:
            action = Action(
                name=f"move_{block}_to_table",
                parameters=[block],
                preconditions={Predicate("clear", [block])},
                add_effects={
                    Predicate("on_table", [block]),
                    Predicate("clear", [block])
                },
                delete_effects=set()
            )
            actions.append(action)
        
        return PlanningProblem(initial_state, goal, actions)


class GraphPlan:
    """GraphPlan算法实现"""
    
    def __init__(self, problem: PlanningProblem):
        self.problem = problem
        self.planning_graph = []
    
    def build_graph(self, max_levels: int = 10):
        """构建规划图"""
        self.planning_graph = []
        
        # 第0层：初始状态
        current_facts = self.problem.initial_state.predicates.copy()
        self.planning_graph.append({'facts': current_facts, 'actions': set()})
        
        for level in range(max_levels):
            # 动作层
            applicable_actions = set()
            for action in self.problem.actions:
                if action.preconditions.issubset(current_facts):
                    applicable_actions.add(action)
            
            # 命题层
            new_facts = current_facts.copy()
            for action in applicable_actions:
                new_facts.update(action.add_effects)
                new_facts.difference_update(action.delete_effects)
            
            self.planning_graph.append({
                'facts': new_facts,
                'actions': applicable_actions
            })
            
            # 检查是否达到目标
            if self.problem.goal.issubset(new_facts):
                return True
            
            # 如果图稳定，停止扩展
            if new_facts == current_facts:
                break
            
            current_facts = new_facts
        
        return False
    
    def extract_plan(self) -> Optional[Plan]:
        """从规划图提取规划"""
        if not self.build_graph():
            return None
        
        # 简化实现：后向搜索提取规划
        current_goals = self.problem.goal.copy()
        plan_actions = []
        
        for level in range(len(self.planning_graph) - 1, 0, -1):
            layer = self.planning_graph[level]
            
            # 找到能实现当前目标的动作
            for action in layer['actions']:
                if current_goals.intersection(action.add_effects):
                    plan_actions.insert(0, action)
                    current_goals = current_goals.difference(action.add_effects)
                    current_goals = current_goals.union(action.preconditions)
                    break
        
        return Plan(plan_actions) if not current_goals else None


def demonstrate_blocks_world():
    """演示积木世界规划"""
    print("=== 积木世界规划演示 ===")
    
    # 初始配置：A在B上，C在桌子上
    initial = [["B", "A"], ["C"]]
    # 目标配置：C在A上，A在B上
    goal = [["B", "A", "C"]]
    
    problem = BlocksWorldProblem.create_problem(initial, goal)
    
    print("初始状态:", problem.initial_state)
    print("目标:", problem.goal)
    
    # 使用前向搜索规划器
    planner = ForwardSearchPlanner()
    plan = planner.plan(problem)
    
    if plan:
        print(f"找到规划（{len(plan)}步）:")
        for i, action in enumerate(plan.actions):
            print(f"  {i+1}. {action}")
        
        # 验证规划
        final_state = plan.execute(problem.initial_state)
        print(f"规划有效: {plan.is_valid(problem.initial_state, problem.goal)}")
        print("最终状态:", final_state)
    else:
        print("未找到规划")


def demonstrate_graph_plan():
    """演示GraphPlan算法"""
    print("\n=== GraphPlan算法演示 ===")
    
    # 简单的积木世界问题
    initial = [["A"], ["B"]]
    goal = [["A", "B"]]
    
    problem = BlocksWorldProblem.create_problem(initial, goal)
    
    graph_planner = GraphPlan(problem)
    plan = graph_planner.extract_plan()
    
    if plan:
        print(f"GraphPlan找到规划（{len(plan)}步）:")
        for i, action in enumerate(plan.actions):
            print(f"  {i+1}. {action}")
    else:
        print("GraphPlan未找到规划")


def demonstrate_planning_comparison():
    """演示不同规划算法的比较"""
    print("\n=== 规划算法比较 ===")
    
    initial = [["A"], ["B"], ["C"]]
    goal = [["A", "B", "C"]]
    
    problem = BlocksWorldProblem.create_problem(initial, goal)
    
    algorithms = [
        ("前向搜索", ForwardSearchPlanner()),
        ("后向搜索", BackwardSearchPlanner()),
    ]
    
    for name, planner in algorithms:
        print(f"\n{name}规划器:")
        plan = planner.plan(problem)
        if plan:
            print(f"  找到规划（{len(plan)}步）")
            print(f"  规划: {plan}")
        else:
            print("  未找到规划")


def main():
    """主演示函数"""
    print("第11章：自动规划")
    print("实现了STRIPS、状态空间搜索、GraphPlan等规划算法")
    
    demonstrate_blocks_world()
    demonstrate_graph_plan()
    demonstrate_planning_comparison()
    
    print("\n=== 规划算法总结 ===")
    print("1. 前向搜索：从初始状态向目标搜索")
    print("2. 后向搜索：从目标向初始状态搜索") 
    print("3. GraphPlan：基于规划图的高效算法")
    print("4. 部分排序规划：处理动作间的依赖关系")
    print("5. 分层任务网络：处理复杂的任务分解")


if __name__ == "__main__":
    main() 