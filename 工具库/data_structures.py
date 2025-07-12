"""
人工智能算法中常用的数据结构

包含队列、栈、优先队列、图等数据结构的实现。
"""

import heapq
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import deque


class Node:
    """通用节点类"""
    
    def __init__(self, state: Any, parent: Optional['Node'] = None, action: Any = None, cost: float = 0):
        self.state = state  # 节点状态
        self.parent = parent  # 父节点
        self.action = action  # 到达此节点的动作
        self.cost = cost  # 从起始节点到当前节点的代价
        self.depth = 0 if parent is None else parent.depth + 1
    
    def path(self) -> List['Node']:
        """返回从根节点到当前节点的路径"""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return path[::-1]
    
    def solution(self) -> List[Any]:
        """返回从根节点到当前节点的动作序列"""
        return [node.action for node in self.path()[1:]]
    
    def __repr__(self):
        return f"Node({self.state})"
    
    def __lt__(self, other):
        return self.cost < other.cost


class Queue:
    """先进先出队列"""
    
    def __init__(self):
        self.items = deque()
    
    def push(self, item: Any):
        """添加元素"""
        self.items.append(item)
    
    def pop(self) -> Any:
        """弹出元素"""
        return self.items.popleft()
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return len(self.items) == 0
    
    def size(self) -> int:
        """返回队列大小"""
        return len(self.items)


class Stack:
    """后进先出栈"""
    
    def __init__(self):
        self.items = []
    
    def push(self, item: Any):
        """添加元素"""
        self.items.append(item)
    
    def pop(self) -> Any:
        """弹出元素"""
        return self.items.pop()
    
    def is_empty(self) -> bool:
        """检查栈是否为空"""
        return len(self.items) == 0
    
    def size(self) -> int:
        """返回栈大小"""
        return len(self.items)


class PriorityQueue:
    """优先队列"""
    
    def __init__(self):
        self.items = []
        self.index = 0
    
    def push(self, item: Any, priority: float):
        """添加元素，优先级越小越先出队"""
        heapq.heappush(self.items, (priority, self.index, item))
        self.index += 1
    
    def pop(self) -> Any:
        """弹出优先级最高的元素"""
        if self.items:
            return heapq.heappop(self.items)[2]
        return None
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return len(self.items) == 0
    
    def size(self) -> int:
        """返回队列大小"""
        return len(self.items)


class Graph:
    """图数据结构"""
    
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.vertices = set()
        self.edges = {}
    
    def add_vertex(self, vertex: Any):
        """添加顶点"""
        self.vertices.add(vertex)
        if vertex not in self.edges:
            self.edges[vertex] = []
    
    def add_edge(self, from_vertex: Any, to_vertex: Any, weight: float = 1.0):
        """添加边"""
        self.add_vertex(from_vertex)
        self.add_vertex(to_vertex)
        
        self.edges[from_vertex].append((to_vertex, weight))
        if not self.directed:
            self.edges[to_vertex].append((from_vertex, weight))
    
    def get_neighbors(self, vertex: Any) -> List[Tuple[Any, float]]:
        """获取顶点的邻居"""
        return self.edges.get(vertex, [])
    
    def get_vertices(self) -> Set[Any]:
        """获取所有顶点"""
        return self.vertices
    
    def has_vertex(self, vertex: Any) -> bool:
        """检查顶点是否存在"""
        return vertex in self.vertices
    
    def has_edge(self, from_vertex: Any, to_vertex: Any) -> bool:
        """检查边是否存在"""
        if from_vertex not in self.edges:
            return False
        return any(neighbor[0] == to_vertex for neighbor in self.edges[from_vertex])
    
    def remove_vertex(self, vertex: Any):
        """移除顶点"""
        if vertex in self.vertices:
            self.vertices.remove(vertex)
            del self.edges[vertex]
            
            # 移除指向该顶点的边
            for v in self.edges:
                self.edges[v] = [(neighbor, weight) for neighbor, weight in self.edges[v] 
                               if neighbor != vertex]
    
    def remove_edge(self, from_vertex: Any, to_vertex: Any):
        """移除边"""
        if from_vertex in self.edges:
            self.edges[from_vertex] = [(neighbor, weight) for neighbor, weight in self.edges[from_vertex] 
                                     if neighbor != to_vertex]
        
        if not self.directed and to_vertex in self.edges:
            self.edges[to_vertex] = [(neighbor, weight) for neighbor, weight in self.edges[to_vertex] 
                                   if neighbor != from_vertex]
    
    def __repr__(self):
        return f"Graph(vertices={len(self.vertices)}, edges={sum(len(neighbors) for neighbors in self.edges.values())})"


class Tree:
    """树数据结构"""
    
    def __init__(self, data: Any):
        self.data = data
        self.children = []
        self.parent = None
    
    def add_child(self, child: 'Tree'):
        """添加子节点"""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: 'Tree'):
        """移除子节点"""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
    
    def is_leaf(self) -> bool:
        """检查是否为叶子节点"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """检查是否为根节点"""
        return self.parent is None
    
    def depth(self) -> int:
        """返回节点深度"""
        if self.parent is None:
            return 0
        return self.parent.depth() + 1
    
    def height(self) -> int:
        """返回树的高度"""
        if self.is_leaf():
            return 0
        return 1 + max(child.height() for child in self.children)
    
    def size(self) -> int:
        """返回树的大小（节点数）"""
        return 1 + sum(child.size() for child in self.children)
    
    def find(self, data: Any) -> Optional['Tree']:
        """查找数据对应的节点"""
        if self.data == data:
            return self
        
        for child in self.children:
            result = child.find(data)
            if result:
                return result
        
        return None
    
    def traverse_preorder(self) -> List[Any]:
        """前序遍历"""
        result = [self.data]
        for child in self.children:
            result.extend(child.traverse_preorder())
        return result
    
    def traverse_postorder(self) -> List[Any]:
        """后序遍历"""
        result = []
        for child in self.children:
            result.extend(child.traverse_postorder())
        result.append(self.data)
        return result
    
    def traverse_levelorder(self) -> List[Any]:
        """层序遍历"""
        result = []
        queue = [self]
        
        while queue:
            node = queue.pop(0)
            result.append(node.data)
            queue.extend(node.children)
        
        return result
    
    def __repr__(self):
        return f"Tree({self.data})"


class UnionFind:
    """并查集数据结构"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x: int) -> int:
        """查找元素所属集合的根"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    
    def union(self, x: int, y: int):
        """合并两个集合"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            # 按秩合并
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            
            self.components -= 1
    
    def connected(self, x: int, y: int) -> bool:
        """检查两个元素是否在同一集合"""
        return self.find(x) == self.find(y)
    
    def count_components(self) -> int:
        """返回连通分量数"""
        return self.components


class Trie:
    """字典树（前缀树）"""
    
    def __init__(self):
        self.root = {}
        self.end_symbol = '#'
    
    def insert(self, word: str):
        """插入单词"""
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[self.end_symbol] = True
    
    def search(self, word: str) -> bool:
        """搜索单词"""
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return self.end_symbol in node
    
    def starts_with(self, prefix: str) -> bool:
        """检查是否存在以prefix开头的单词"""
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
    
    def get_all_words_with_prefix(self, prefix: str) -> List[str]:
        """获取所有以prefix开头的单词"""
        node = self.root
        for char in prefix:
            if char not in node:
                return []
            node = node[char]
        
        words = []
        self._collect_words(node, prefix, words)
        return words
    
    def _collect_words(self, node: dict, prefix: str, words: List[str]):
        """收集以prefix开头的所有单词"""
        if self.end_symbol in node:
            words.append(prefix)
        
        for char, child_node in node.items():
            if char != self.end_symbol:
                self._collect_words(child_node, prefix + char, words)


class LRUCache:
    """最近最少使用缓存"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()
    
    def get(self, key: Any) -> Any:
        """获取值"""
        if key in self.cache:
            # 更新顺序
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: Any, value: Any):
        """设置值"""
        if key in self.cache:
            # 更新顺序
            self.order.remove(key)
            self.order.append(key)
            self.cache[key] = value
        else:
            if len(self.cache) >= self.capacity:
                # 移除最久未使用的
                oldest = self.order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = value
            self.order.append(key)
    
    def size(self) -> int:
        """返回缓存大小"""
        return len(self.cache) 