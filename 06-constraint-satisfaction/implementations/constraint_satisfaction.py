"""
约束满足问题实现

包含回溯搜索、约束传播、弧一致性等算法
"""

import random
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
import copy


class Variable:
    """变量类"""
    
    def __init__(self, name: str, domain: List[Any]):
        self.name = name
        self.domain = domain[:]  # 创建副本
        self.initial_domain = domain[:]
        self.value = None
        self.assigned = False
    
    def assign(self, value: Any):
        """分配值"""
        if value not in self.domain:
            raise ValueError(f"Value {value} not in domain {self.domain}")
        self.value = value
        self.assigned = True
    
    def unassign(self):
        """取消分配"""
        self.value = None
        self.assigned = False
    
    def reset_domain(self):
        """重置域"""
        self.domain = self.initial_domain[:]
    
    def __str__(self):
        return f"Variable({self.name}, domain={self.domain}, value={self.value})"
    
    def __repr__(self):
        return self.__str__()


class Constraint:
    """约束类"""
    
    def __init__(self, variables: List[Variable], constraint_func: Callable):
        self.variables = variables
        self.constraint_func = constraint_func
    
    def is_satisfied(self, assignment: Dict[Variable, Any]) -> bool:
        """检查约束是否满足"""
        # 只检查已分配变量
        relevant_assignment = {var: assignment[var] for var in self.variables 
                              if var in assignment and assignment[var] is not None}
        
        if len(relevant_assignment) < len(self.variables):
            return True  # 未完全分配时认为满足
        
        return self.constraint_func(relevant_assignment)
    
    def get_unassigned_variables(self, assignment: Dict[Variable, Any]) -> List[Variable]:
        """获取未分配的变量"""
        return [var for var in self.variables 
                if var not in assignment or assignment[var] is None]


class CSP:
    """约束满足问题"""
    
    def __init__(self, variables: List[Variable], constraints: List[Constraint]):
        self.variables = variables
        self.constraints = constraints
        self.var_to_constraints = defaultdict(list)
        
        # 构建变量到约束的映射
        for constraint in constraints:
            for var in constraint.variables:
                self.var_to_constraints[var].append(constraint)
    
    def is_consistent(self, assignment: Dict[Variable, Any], var: Variable, value: Any) -> bool:
        """检查赋值是否与约束一致"""
        temp_assignment = assignment.copy()
        temp_assignment[var] = value
        
        for constraint in self.var_to_constraints[var]:
            if not constraint.is_satisfied(temp_assignment):
                return False
        
        return True
    
    def get_neighbors(self, var: Variable) -> Set[Variable]:
        """获取变量的邻居"""
        neighbors = set()
        for constraint in self.var_to_constraints[var]:
            neighbors.update(constraint.variables)
        neighbors.discard(var)
        return neighbors
    
    def get_unassigned_variables(self, assignment: Dict[Variable, Any]) -> List[Variable]:
        """获取未分配的变量"""
        return [var for var in self.variables 
                if var not in assignment or assignment[var] is None]


class BacktrackingSearch:
    """回溯搜索算法"""
    
    def __init__(self, csp: CSP, 
                 var_heuristic: str = 'mrv',
                 value_heuristic: str = 'lcv',
                 inference: str = 'ac3'):
        self.csp = csp
        self.var_heuristic = var_heuristic
        self.value_heuristic = value_heuristic
        self.inference = inference
        self.assignments_tried = 0
        self.backtrack_count = 0
    
    def solve(self) -> Optional[Dict[Variable, Any]]:
        """求解CSP"""
        self.assignments_tried = 0
        self.backtrack_count = 0
        
        # 初始化分配
        assignment = {}
        
        # 应用弧一致性作为预处理
        if self.inference == 'ac3':
            if not self.ac3():
                return None
        
        result = self.backtrack(assignment)
        return result
    
    def backtrack(self, assignment: Dict[Variable, Any]) -> Optional[Dict[Variable, Any]]:
        """回溯搜索"""
        # 检查是否完成
        if len(assignment) == len(self.csp.variables):
            return assignment
        
        # 选择变量
        var = self.select_unassigned_variable(assignment)
        
        # 尝试每个值
        for value in self.order_domain_values(var, assignment):
            self.assignments_tried += 1
            
            if self.csp.is_consistent(assignment, var, value):
                # 添加赋值
                assignment[var] = value
                
                # 应用推理
                inferences = self.inference_step(var, value, assignment)
                
                if inferences is not None:
                    # 递归搜索
                    result = self.backtrack(assignment)
                    if result is not None:
                        return result
                
                # 撤销推理
                if inferences is not None:
                    self.restore_domains(inferences)
                
                # 撤销赋值
                del assignment[var]
                self.backtrack_count += 1
        
        return None
    
    def select_unassigned_variable(self, assignment: Dict[Variable, Any]) -> Variable:
        """选择未分配的变量"""
        unassigned = self.csp.get_unassigned_variables(assignment)
        
        if self.var_heuristic == 'mrv':
            # 最小剩余值启发式
            return min(unassigned, key=lambda var: len(var.domain))
        elif self.var_heuristic == 'degree':
            # 度启发式
            return max(unassigned, key=lambda var: len(self.csp.get_neighbors(var)))
        else:
            # 随机选择
            return random.choice(unassigned)
    
    def order_domain_values(self, var: Variable, assignment: Dict[Variable, Any]) -> List[Any]:
        """对域值排序"""
        if self.value_heuristic == 'lcv':
            # 最少约束值启发式
            def constraint_count(value):
                count = 0
                for neighbor in self.csp.get_neighbors(var):
                    if neighbor not in assignment:
                        if value in neighbor.domain:
                            count += 1
                return count
            
            return sorted(var.domain, key=constraint_count)
        else:
            # 随机顺序
            domain_copy = var.domain[:]
            random.shuffle(domain_copy)
            return domain_copy
    
    def inference_step(self, var: Variable, value: Any, 
                      assignment: Dict[Variable, Any]) -> Optional[List[Tuple[Variable, Any]]]:
        """推理步骤"""
        if self.inference == 'fc':
            return self.forward_checking(var, value, assignment)
        elif self.inference == 'ac3':
            return self.ac3_inference(var, value, assignment)
        else:
            return []
    
    def forward_checking(self, var: Variable, value: Any, 
                        assignment: Dict[Variable, Any]) -> Optional[List[Tuple[Variable, Any]]]:
        """前向检查"""
        inferences = []
        
        for neighbor in self.csp.get_neighbors(var):
            if neighbor not in assignment:
                values_to_remove = []
                
                for neighbor_value in neighbor.domain:
                    temp_assignment = assignment.copy()
                    temp_assignment[var] = value
                    temp_assignment[neighbor] = neighbor_value
                    
                    # 检查是否违反约束
                    consistent = True
                    for constraint in self.csp.var_to_constraints[neighbor]:
                        if not constraint.is_satisfied(temp_assignment):
                            consistent = False
                            break
                    
                    if not consistent:
                        values_to_remove.append(neighbor_value)
                
                # 移除不一致的值
                for val in values_to_remove:
                    if val in neighbor.domain:
                        neighbor.domain.remove(val)
                        inferences.append((neighbor, val))
                
                # 检查域是否为空
                if not neighbor.domain:
                    # 恢复域并返回失败
                    self.restore_domains(inferences)
                    return None
        
        return inferences
    
    def ac3_inference(self, var: Variable, value: Any, 
                     assignment: Dict[Variable, Any]) -> Optional[List[Tuple[Variable, Any]]]:
        """AC-3推理"""
        inferences = []
        queue = deque()
        
        # 添加所有相关的弧
        for neighbor in self.csp.get_neighbors(var):
            if neighbor not in assignment:
                queue.append((neighbor, var))
        
        while queue:
            xi, xj = queue.popleft()
            
            if self.revise(xi, xj, assignment):
                inferences.extend([(xi, val) for val in xi.initial_domain if val not in xi.domain])
                
                if not xi.domain:
                    self.restore_domains(inferences)
                    return None
                
                # 添加新的弧
                for neighbor in self.csp.get_neighbors(xi):
                    if neighbor != xj and neighbor not in assignment:
                        queue.append((neighbor, xi))
        
        return inferences
    
    def revise(self, xi: Variable, xj: Variable, assignment: Dict[Variable, Any]) -> bool:
        """修正弧(xi, xj)"""
        revised = False
        values_to_remove = []
        
        for x in xi.domain:
            # 检查是否存在y使得约束满足
            consistent = False
            
            if xj in assignment:
                # xj已分配
                y = assignment[xj]
                temp_assignment = assignment.copy()
                temp_assignment[xi] = x
                temp_assignment[xj] = y
                
                for constraint in self.csp.var_to_constraints[xi]:
                    if xj in constraint.variables:
                        if constraint.is_satisfied(temp_assignment):
                            consistent = True
                            break
            else:
                # xj未分配，检查所有可能的值
                for y in xj.domain:
                    temp_assignment = assignment.copy()
                    temp_assignment[xi] = x
                    temp_assignment[xj] = y
                    
                    constraint_satisfied = True
                    for constraint in self.csp.var_to_constraints[xi]:
                        if xj in constraint.variables:
                            if not constraint.is_satisfied(temp_assignment):
                                constraint_satisfied = False
                                break
                    
                    if constraint_satisfied:
                        consistent = True
                        break
            
            if not consistent:
                values_to_remove.append(x)
                revised = True
        
        # 移除不一致的值
        for val in values_to_remove:
            xi.domain.remove(val)
        
        return revised
    
    def ac3(self) -> bool:
        """AC-3算法"""
        queue = deque()
        
        # 初始化队列
        for constraint in self.csp.constraints:
            for i, var1 in enumerate(constraint.variables):
                for j, var2 in enumerate(constraint.variables):
                    if i != j:
                        queue.append((var1, var2))
        
        while queue:
            xi, xj = queue.popleft()
            
            if self.revise(xi, xj, {}):
                if not xi.domain:
                    return False
                
                for neighbor in self.csp.get_neighbors(xi):
                    if neighbor != xj:
                        queue.append((neighbor, xi))
        
        return True
    
    def restore_domains(self, inferences: List[Tuple[Variable, Any]]):
        """恢复域"""
        for var, value in inferences:
            var.domain.append(value)


class NQueensProblem:
    """N皇后问题"""
    
    def __init__(self, n: int):
        self.n = n
        self.variables = []
        self.constraints = []
        
        # 创建变量（每行一个皇后）
        for i in range(n):
            var = Variable(f"Q{i}", list(range(n)))
            self.variables.append(var)
        
        # 创建约束
        for i in range(n):
            for j in range(i + 1, n):
                constraint = Constraint(
                    [self.variables[i], self.variables[j]],
                    lambda assignment, row1=i, row2=j: self.no_conflict(assignment, row1, row2)
                )
                self.constraints.append(constraint)
    
    def no_conflict(self, assignment: Dict[Variable, Any], row1: int, row2: int) -> bool:
        """检查两个皇后是否冲突"""
        var1 = self.variables[row1]
        var2 = self.variables[row2]
        
        col1 = assignment[var1]
        col2 = assignment[var2]
        
        # 检查列冲突
        if col1 == col2:
            return False
        
        # 检查对角线冲突
        if abs(row1 - row2) == abs(col1 - col2):
            return False
        
        return True
    
    def create_csp(self) -> CSP:
        """创建CSP"""
        return CSP(self.variables, self.constraints)


class SudokuProblem:
    """数独问题"""
    
    def __init__(self, puzzle: List[List[int]]):
        self.puzzle = puzzle
        self.size = 9
        self.box_size = 3
        self.variables = []
        self.constraints = []
        
        # 创建变量
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if puzzle[i][j] == 0:
                    var = Variable(f"Cell({i},{j})", list(range(1, 10)))
                else:
                    var = Variable(f"Cell({i},{j})", [puzzle[i][j]])
                    var.assign(puzzle[i][j])
                row.append(var)
            self.variables.append(row)
        
        # 创建约束
        self.create_constraints()
    
    def create_constraints(self):
        """创建约束"""
        # 行约束
        for i in range(self.size):
            for j in range(self.size):
                for k in range(j + 1, self.size):
                    constraint = Constraint(
                        [self.variables[i][j], self.variables[i][k]],
                        lambda assignment: len(set(assignment.values())) == len(assignment)
                    )
                    self.constraints.append(constraint)
        
        # 列约束
        for j in range(self.size):
            for i in range(self.size):
                for k in range(i + 1, self.size):
                    constraint = Constraint(
                        [self.variables[i][j], self.variables[k][j]],
                        lambda assignment: len(set(assignment.values())) == len(assignment)
                    )
                    self.constraints.append(constraint)
        
        # 九宫格约束
        for box_row in range(self.box_size):
            for box_col in range(self.box_size):
                box_vars = []
                for i in range(box_row * self.box_size, (box_row + 1) * self.box_size):
                    for j in range(box_col * self.box_size, (box_col + 1) * self.box_size):
                        box_vars.append(self.variables[i][j])
                
                # 为每对变量创建约束
                for i in range(len(box_vars)):
                    for j in range(i + 1, len(box_vars)):
                        constraint = Constraint(
                            [box_vars[i], box_vars[j]],
                            lambda assignment: len(set(assignment.values())) == len(assignment)
                        )
                        self.constraints.append(constraint)
    
    def create_csp(self) -> CSP:
        """创建CSP"""
        all_vars = [var for row in self.variables for var in row]
        return CSP(all_vars, self.constraints)
    
    def print_solution(self, assignment: Dict[Variable, Any]):
        """打印解决方案"""
        for i in range(self.size):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            
            row = ""
            for j in range(self.size):
                if j % 3 == 0 and j != 0:
                    row += "| "
                
                var = self.variables[i][j]
                if var in assignment:
                    row += str(assignment[var]) + " "
                else:
                    row += str(var.value) + " "
            
            print(row)


class GraphColoringProblem:
    """图着色问题"""
    
    def __init__(self, graph: Dict[str, List[str]], colors: List[str]):
        self.graph = graph
        self.colors = colors
        self.variables = []
        self.constraints = []
        
        # 创建变量
        for node in graph:
            var = Variable(node, colors[:])
            self.variables.append(var)
        
        # 创建约束
        self.create_constraints()
    
    def create_constraints(self):
        """创建约束"""
        var_dict = {var.name: var for var in self.variables}
        
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                if neighbor in var_dict:
                    constraint = Constraint(
                        [var_dict[node], var_dict[neighbor]],
                        lambda assignment: len(set(assignment.values())) == len(assignment)
                    )
                    self.constraints.append(constraint)
    
    def create_csp(self) -> CSP:
        """创建CSP"""
        return CSP(self.variables, self.constraints)


def solve_n_queens(n: int = 8):
    """求解N皇后问题"""
    print(f"求解{n}皇后问题")
    print("=" * 30)
    
    problem = NQueensProblem(n)
    csp = problem.create_csp()
    
    # 尝试不同的启发式
    heuristics = [
        ('mrv', 'lcv', 'fc'),
        ('degree', 'random', 'ac3'),
        ('random', 'random', 'none')
    ]
    
    for var_h, val_h, inf_h in heuristics:
        print(f"\n启发式: {var_h}, {val_h}, {inf_h}")
        
        # 重置变量域
        for var in csp.variables:
            var.reset_domain()
            var.unassign()
        
        solver = BacktrackingSearch(csp, var_h, val_h, inf_h)
        solution = solver.solve()
        
        if solution:
            print(f"找到解! 尝试分配次数: {solver.assignments_tried}")
            print(f"回溯次数: {solver.backtrack_count}")
            
            # 打印解
            board = [['.' for _ in range(n)] for _ in range(n)]
            for i, var in enumerate(csp.variables):
                col = solution[var]
                board[i][col] = 'Q'
            
            for row in board:
                print(' '.join(row))
            break
        else:
            print("无解")


def solve_sudoku():
    """求解数独问题"""
    print("求解数独问题")
    print("=" * 30)
    
    # 简单的数独谜题
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    problem = SudokuProblem(puzzle)
    csp = problem.create_csp()
    
    print("原始谜题:")
    problem.print_solution({})
    
    solver = BacktrackingSearch(csp, 'mrv', 'lcv', 'ac3')
    solution = solver.solve()
    
    if solution:
        print(f"\n找到解! 尝试分配次数: {solver.assignments_tried}")
        print(f"回溯次数: {solver.backtrack_count}")
        
        print("\n解:")
        problem.print_solution(solution)
    else:
        print("无解")


def solve_graph_coloring():
    """求解图着色问题"""
    print("求解图着色问题")
    print("=" * 30)
    
    # 澳大利亚地图着色问题
    australia_map = {
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'SA', 'Q'],
        'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
        'Q': ['NT', 'SA', 'NSW'],
        'NSW': ['Q', 'SA', 'V'],
        'V': ['SA', 'NSW'],
        'T': []
    }
    
    colors = ['红', '绿', '蓝']
    
    problem = GraphColoringProblem(australia_map, colors)
    csp = problem.create_csp()
    
    solver = BacktrackingSearch(csp, 'mrv', 'lcv', 'ac3')
    solution = solver.solve()
    
    if solution:
        print(f"找到解! 尝试分配次数: {solver.assignments_tried}")
        print(f"回溯次数: {solver.backtrack_count}")
        
        print("\n着色方案:")
        for var in csp.variables:
            print(f"{var.name}: {solution[var]}")
    else:
        print("无解")


if __name__ == "__main__":
    # 演示不同的约束满足问题
    solve_n_queens(8)
    print("\n" + "="*50)
    solve_sudoku()
    print("\n" + "="*50)
    solve_graph_coloring()
    
    print("\n✅ 约束满足问题演示完成！") 