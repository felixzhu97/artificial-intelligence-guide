"""
算法工具库

包含项目中使用的核心算法实现。
"""

import numpy as np
import heapq
from typing import List, Dict, Tuple, Optional, Any, Callable, Set
from collections import defaultdict, deque
import random
import math


def euclidean_distance(point1: Tuple[float, ...], point2: Tuple[float, ...]) -> float:
    """计算欧几里得距离"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def manhattan_distance(point1: Tuple[float, ...], point2: Tuple[float, ...]) -> float:
    """计算曼哈顿距离"""
    return sum(abs(a - b) for a, b in zip(point1, point2))


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def k_means_clustering(data: List[List[float]], k: int, max_iterations: int = 100) -> Tuple[List[List[float]], List[int]]:
    """
    K-means聚类算法
    
    Args:
        data: 数据点列表
        k: 聚类数量
        max_iterations: 最大迭代次数
    
    Returns:
        (centroids, labels): 聚类中心和标签
    """
    if not data or k <= 0:
        return [], []
    
    # 初始化聚类中心
    centroids = random.sample(data, k)
    
    for _ in range(max_iterations):
        # 分配数据点到最近的聚类中心
        clusters = [[] for _ in range(k)]
        labels = []
        
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(point)
            labels.append(closest_centroid)
        
        # 更新聚类中心
        new_centroids = []
        for cluster in clusters:
            if cluster:
                # 计算平均值
                centroid = [sum(coords) / len(cluster) for coords in zip(*cluster)]
                new_centroids.append(centroid)
            else:
                # 如果聚类为空，保持原来的中心
                new_centroids.append(centroids[len(new_centroids)])
        
        # 检查收敛
        if all(euclidean_distance(old, new) < 1e-6 for old, new in zip(centroids, new_centroids)):
            break
        
        centroids = new_centroids
    
    return centroids, labels


def hierarchical_clustering(data: List[List[float]], linkage: str = 'single') -> List[Tuple[int, int, float]]:
    """
    层次聚类算法
    
    Args:
        data: 数据点列表
        linkage: 链接方法 ('single', 'complete', 'average')
    
    Returns:
        聚类树（合并历史）
    """
    n = len(data)
    if n <= 1:
        return []
    
    # 计算所有点对之间的距离
    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            distances[(i, j)] = euclidean_distance(data[i], data[j])
    
    # 初始化每个点作为一个聚类
    clusters = {i: [i] for i in range(n)}
    merges = []
    
    while len(clusters) > 1:
        # 找到最近的两个聚类
        min_dist = float('inf')
        merge_pair = None
        
        cluster_ids = list(clusters.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                c1, c2 = cluster_ids[i], cluster_ids[j]
                
                # 计算聚类间距离
                if linkage == 'single':
                    dist = min(distances.get((min(p1, p2), max(p1, p2)), float('inf'))
                              for p1 in clusters[c1] for p2 in clusters[c2])
                elif linkage == 'complete':
                    dist = max(distances.get((min(p1, p2), max(p1, p2)), 0)
                              for p1 in clusters[c1] for p2 in clusters[c2])
                elif linkage == 'average':
                    dists = [distances.get((min(p1, p2), max(p1, p2)), 0)
                            for p1 in clusters[c1] for p2 in clusters[c2]]
                    dist = sum(dists) / len(dists)
                
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (c1, c2)
        
        # 合并最近的两个聚类
        if merge_pair:
            c1, c2 = merge_pair
            merges.append((c1, c2, min_dist))
            
            # 创建新聚类
            new_cluster_id = max(clusters.keys()) + 1
            clusters[new_cluster_id] = clusters[c1] + clusters[c2]
            
            # 删除旧聚类
            del clusters[c1]
            del clusters[c2]
    
    return merges


def dijkstra_shortest_path(graph: Dict[Any, List[Tuple[Any, float]]], start: Any, end: Any) -> Tuple[List[Any], float]:
    """
    Dijkstra最短路径算法
    
    Args:
        graph: 图的邻接表表示 {node: [(neighbor, weight), ...]}
        start: 起始节点
        end: 结束节点
    
    Returns:
        (path, distance): 最短路径和距离
    """
    if start not in graph or end not in graph:
        return [], float('inf')
    
    # 初始化距离和前驱节点
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    predecessors = {}
    
    # 优先队列：(距离, 节点)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # 如果到达目标节点
        if current_node == end:
            break
        
        # 更新邻居节点的距离
        for neighbor, weight in graph[current_node]:
            if neighbor not in visited:
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))
    
    # 重构路径
    path = []
    current = end
    while current in predecessors:
        path.append(current)
        current = predecessors[current]
    path.append(start)
    path.reverse()
    
    # 如果无法到达目标节点
    if distances[end] == float('inf'):
        return [], float('inf')
    
    return path, distances[end]


def a_star_search(graph: Dict[Any, List[Tuple[Any, float]]], start: Any, end: Any, 
                  heuristic: Callable[[Any, Any], float]) -> Tuple[List[Any], float]:
    """
    A*搜索算法
    
    Args:
        graph: 图的邻接表表示
        start: 起始节点
        end: 结束节点
        heuristic: 启发式函数
    
    Returns:
        (path, distance): 最短路径和距离
    """
    if start not in graph or end not in graph:
        return [], float('inf')
    
    # 初始化
    open_set = [(heuristic(start, end), 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed_set = set()
    
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        if current == end:
            # 重构路径
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[end]
        
        for neighbor, weight in graph[current]:
            if neighbor in closed_set:
                continue
            
            tentative_g = g_score[current] + weight
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))
    
    return [], float('inf')


def breadth_first_search(graph: Dict[Any, List[Any]], start: Any, end: Any) -> List[Any]:
    """
    广度优先搜索
    
    Args:
        graph: 图的邻接表表示
        start: 起始节点
        end: 结束节点
    
    Returns:
        路径列表
    """
    if start not in graph or end not in graph:
        return []
    
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        node, path = queue.popleft()
        
        if node in visited:
            continue
        
        visited.add(node)
        
        if node == end:
            return path
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    
    return []


def depth_first_search(graph: Dict[Any, List[Any]], start: Any, end: Any) -> List[Any]:
    """
    深度优先搜索
    
    Args:
        graph: 图的邻接表表示
        start: 起始节点
        end: 结束节点
    
    Returns:
        路径列表
    """
    if start not in graph or end not in graph:
        return []
    
    stack = [(start, [start])]
    visited = set()
    
    while stack:
        node, path = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        
        if node == end:
            return path
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    
    return []


def topological_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    """
    拓扑排序
    
    Args:
        graph: 有向图的邻接表表示
    
    Returns:
        拓扑排序结果
    """
    # 计算入度
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            if neighbor not in in_degree:
                in_degree[neighbor] = 0
            in_degree[neighbor] += 1
    
    # 找到所有入度为0的节点
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # 更新邻居节点的入度
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # 检查是否有环
    if len(result) != len(in_degree):
        return []  # 有环，无法进行拓扑排序
    
    return result


def binary_search(arr: List[Any], target: Any) -> int:
    """
    二分搜索
    
    Args:
        arr: 已排序的数组
        target: 目标值
    
    Returns:
        目标值的索引，如果不存在返回-1
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


def quick_sort(arr: List[Any]) -> List[Any]:
    """
    快速排序
    
    Args:
        arr: 待排序数组
    
    Returns:
        排序后的数组
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)


def merge_sort(arr: List[Any]) -> List[Any]:
    """
    归并排序
    
    Args:
        arr: 待排序数组
    
    Returns:
        排序后的数组
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left: List[Any], right: List[Any]) -> List[Any]:
    """归并两个已排序的数组"""
    result = []
    i, j = 0, 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


def heap_sort(arr: List[Any]) -> List[Any]:
    """
    堆排序
    
    Args:
        arr: 待排序数组
    
    Returns:
        排序后的数组
    """
    def heapify(arr: List[Any], n: int, i: int):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and arr[left] > arr[largest]:
            largest = left
        
        if right < n and arr[right] > arr[largest]:
            largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    
    arr = arr.copy()
    n = len(arr)
    
    # 构建最大堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # 一个个从堆顶取出元素
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    
    return arr


def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0-1背包问题动态规划解法
    
    Args:
        weights: 物品重量列表
        values: 物品价值列表
        capacity: 背包容量
    
    Returns:
        最大价值
    """
    n = len(weights)
    if n == 0 or capacity == 0:
        return 0
    
    # dp[i][w] 表示前i个物品在容量为w时的最大价值
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],  # 不选择当前物品
                    dp[i-1][w-weights[i-1]] + values[i-1]  # 选择当前物品
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]


def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    最长公共子序列
    
    Args:
        text1: 第一个字符串
        text2: 第二个字符串
    
    Returns:
        最长公共子序列的长度
    """
    m, n = len(text1), len(text2)
    
    # dp[i][j] 表示text1[0:i]和text2[0:j]的LCS长度
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def edit_distance(word1: str, word2: str) -> int:
    """
    编辑距离（Levenshtein距离）
    
    Args:
        word1: 第一个字符串
        word2: 第二个字符串
    
    Returns:
        编辑距离
    """
    m, n = len(word1), len(word2)
    
    # dp[i][j] 表示word1[0:i]转换为word2[0:j]的最小操作数
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # 删除
                    dp[i][j-1],    # 插入
                    dp[i-1][j-1]   # 替换
                )
    
    return dp[m][n]


def maximum_subarray_sum(arr: List[int]) -> int:
    """
    最大子数组和（Kadane算法）
    
    Args:
        arr: 整数数组
    
    Returns:
        最大子数组和
    """
    if not arr:
        return 0
    
    max_sum = current_sum = arr[0]
    
    for i in range(1, len(arr)):
        current_sum = max(arr[i], current_sum + arr[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum


def find_connected_components(graph: Dict[Any, List[Any]]) -> List[List[Any]]:
    """
    找到图的所有连通分量
    
    Args:
        graph: 图的邻接表表示
    
    Returns:
        连通分量列表
    """
    visited = set()
    components = []
    
    def dfs(node: Any, component: List[Any]):
        visited.add(node)
        component.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
    
    return components


def is_bipartite(graph: Dict[Any, List[Any]]) -> bool:
    """
    判断图是否为二分图
    
    Args:
        graph: 图的邻接表表示
    
    Returns:
        是否为二分图
    """
    color = {}
    
    def dfs(node: Any, c: int) -> bool:
        color[node] = c
        
        for neighbor in graph.get(node, []):
            if neighbor in color:
                if color[neighbor] == c:
                    return False
            else:
                if not dfs(neighbor, 1 - c):
                    return False
        
        return True
    
    for node in graph:
        if node not in color:
            if not dfs(node, 0):
                return False
    
    return True 