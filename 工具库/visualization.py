"""
可视化工具库

包含项目中使用的常用可视化和绘图函数。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_learning_curve(train_scores: List[float], val_scores: List[float] = None,
                        title: str = "学习曲线", xlabel: str = "训练轮数", 
                        ylabel: str = "分数", save_path: str = None):
    """
    绘制学习曲线
    
    Args:
        train_scores: 训练分数列表
        val_scores: 验证分数列表（可选）
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'b-', label='训练分数', linewidth=2)
    
    if val_scores:
        plt.plot(epochs, val_scores, 'r-', label='验证分数', linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 添加最佳点标记
    if val_scores:
        best_epoch = np.argmax(val_scores) + 1
        best_score = max(val_scores)
        plt.scatter(best_epoch, best_score, color='red', s=100, zorder=5)
        plt.annotate(f'最佳: ({best_epoch}, {best_score:.3f})', 
                    xy=(best_epoch, best_score), 
                    xytext=(best_epoch + len(epochs) * 0.1, best_score),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         classes: List[str] = None, title: str = "混淆矩阵",
                         cmap: str = 'Blues', save_path: str = None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称列表
        title: 图表标题
        cmap: 颜色映射
        save_path: 保存路径（可选）
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    if classes is None:
        classes = [f'类别 {i}' for i in range(len(cm))]
    
    plt.figure(figsize=(8, 6))
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': '样本数量'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    
    # 添加准确率信息
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(len(classes) * 0.5, len(classes) + 0.5, 
             f'准确率: {accuracy:.3f}', 
             horizontalalignment='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=classes))


def plot_feature_importance(importance: Dict[str, float], title: str = "特征重要性",
                           top_n: int = 10, save_path: str = None):
    """
    绘制特征重要性
    
    Args:
        importance: 特征重要性字典
        title: 图表标题
        top_n: 显示前N个特征
        save_path: 保存路径（可选）
    """
    # 排序并取前N个
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    features = [item[0] for item in sorted_features]
    values = [item[1] for item in sorted_features]
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.barh(features, values, color='skyblue', alpha=0.8)
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center', fontsize=10)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('重要性', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.gca().invert_yaxis()  # 反转y轴，使最重要的在顶部
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_algorithm_comparison(results: Dict[str, Dict[str, float]], 
                             metrics: List[str] = None,
                             title: str = "算法性能比较", 
                             save_path: str = None):
    """
    绘制算法性能比较图
    
    Args:
        results: 算法结果字典，格式：{算法名: {指标名: 值}}
        metrics: 要比较的指标列表
        title: 图表标题
        save_path: 保存路径（可选）
    """
    if metrics is None:
        # 自动获取所有指标
        metrics = list(next(iter(results.values())).keys())
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    algorithms = list(results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    
    for i, metric in enumerate(metrics):
        values = [results[alg][metric] for alg in algorithms]
        
        bars = axes[i].bar(algorithms, values, color=colors, alpha=0.8)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        axes[i].set_title(f'{metric}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_decision_boundary_2d(X: np.ndarray, y: np.ndarray, model: Any,
                             title: str = "决策边界", resolution: int = 100,
                             save_path: str = None):
    """
    绘制2D决策边界
    
    Args:
        X: 特征数据 (n_samples, 2)
        y: 标签数据
        model: 训练好的模型（需要有predict方法）
        title: 图表标题
        resolution: 网格分辨率
        save_path: 保存路径（可选）
    """
    if X.shape[1] != 2:
        raise ValueError("此函数仅支持2D特征数据")
    
    plt.figure(figsize=(10, 8))
    
    # 创建网格
    h = (X[:, 0].max() - X[:, 0].min()) / resolution
    xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, h),
                         np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, h))
    
    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    try:
        Z = model.predict(grid_points)
    except:
        # 如果模型有特殊的预测方法
        if hasattr(model, 'predict_classes'):
            Z = model.predict_classes(grid_points)
        else:
            Z = model.forward(grid_points)
            if Z.ndim > 1:
                Z = np.argmax(Z, axis=1)
    
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # 绘制数据点
    unique_labels = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                   label=f'类别 {label}', alpha=0.8, s=50, edgecolors='black')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('特征 1', fontsize=12)
    plt.ylabel('特征 2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_recommendation_analysis(user_item_matrix: np.ndarray, 
                                title: str = "推荐系统分析",
                                save_path: str = None):
    """
    绘制推荐系统分析图
    
    Args:
        user_item_matrix: 用户-物品评分矩阵
        title: 图表标题
        save_path: 保存路径（可选）
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 评分分布
    ratings = user_item_matrix[user_item_matrix > 0]
    ax1.hist(ratings, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('评分分布', fontsize=14, fontweight='bold')
    ax1.set_xlabel('评分')
    ax1.set_ylabel('频次')
    ax1.grid(True, alpha=0.3)
    
    # 用户活跃度
    user_activity = np.sum(user_item_matrix > 0, axis=1)
    ax2.hist(user_activity, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
    ax2.set_title('用户活跃度分布', fontsize=14, fontweight='bold')
    ax2.set_xlabel('评分数量')
    ax2.set_ylabel('用户数')
    ax2.grid(True, alpha=0.3)
    
    # 物品流行度
    item_popularity = np.sum(user_item_matrix > 0, axis=0)
    ax3.hist(item_popularity, bins=20, color='salmon', alpha=0.7, edgecolor='black')
    ax3.set_title('物品流行度分布', fontsize=14, fontweight='bold')
    ax3.set_xlabel('被评分次数')
    ax3.set_ylabel('物品数')
    ax3.grid(True, alpha=0.3)
    
    # 稀疏度热力图
    sample_size = min(50, user_item_matrix.shape[0])
    sample_matrix = user_item_matrix[:sample_size, :min(50, user_item_matrix.shape[1])]
    
    im = ax4.imshow(sample_matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_title('评分矩阵热力图（样本）', fontsize=14, fontweight='bold')
    ax4.set_xlabel('物品')
    ax4.set_ylabel('用户')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('评分')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_neural_network_architecture(layer_sizes: List[int], 
                                    title: str = "神经网络架构",
                                    save_path: str = None):
    """
    绘制神经网络架构图
    
    Args:
        layer_sizes: 每层的神经元数量列表
        title: 图表标题
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置参数
    max_nodes = max(layer_sizes)
    layer_spacing = 3
    node_spacing = 1
    
    # 绘制每一层
    for i, layer_size in enumerate(layer_sizes):
        x = i * layer_spacing
        
        # 计算y位置，使节点居中
        y_start = (max_nodes - layer_size) * node_spacing / 2
        
        # 绘制节点
        for j in range(layer_size):
            y = y_start + j * node_spacing
            
            # 选择颜色
            if i == 0:
                color = 'lightblue'  # 输入层
            elif i == len(layer_sizes) - 1:
                color = 'lightcoral'  # 输出层
            else:
                color = 'lightgreen'  # 隐藏层
            
            circle = plt.Circle((x, y), 0.2, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # 添加连接线到下一层
            if i < len(layer_sizes) - 1:
                next_layer_size = layer_sizes[i + 1]
                next_y_start = (max_nodes - next_layer_size) * node_spacing / 2
                
                for k in range(next_layer_size):
                    next_y = next_y_start + k * node_spacing
                    ax.plot([x + 0.2, (i + 1) * layer_spacing - 0.2], 
                           [y, next_y], 'gray', alpha=0.5, linewidth=0.5)
    
    # 添加层标签
    layer_labels = ['输入层'] + [f'隐藏层{i}' for i in range(1, len(layer_sizes) - 1)] + ['输出层']
    
    for i, (layer_size, label) in enumerate(zip(layer_sizes, layer_labels)):
        x = i * layer_spacing
        y = max_nodes * node_spacing / 2 + 0.8
        ax.text(x, y, f'{label}\n({layer_size})', ha='center', va='center', 
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlim(-0.5, (len(layer_sizes) - 1) * layer_spacing + 0.5)
    ax.set_ylim(-0.5, max_nodes * node_spacing)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_search_tree(tree_data: Dict, title: str = "搜索树", save_path: str = None):
    """
    绘制搜索树
    
    Args:
        tree_data: 树数据，格式：{'root': 'A', 'children': {'A': ['B', 'C'], ...}}
        title: 图表标题
        save_path: 保存路径（可选）
    """
    try:
        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点和边
        for parent, children in tree_data.get('children', {}).items():
            for child in children:
                G.add_edge(parent, child)
        
        plt.figure(figsize=(12, 8))
        
        # 计算布局
        try:
            pos = graphviz_layout(G, prog='dot')
        except:
            pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, alpha=0.7)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=800, alpha=0.8)
        
        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("需要安装networkx和pygraphviz来绘制搜索树")
        print("pip install networkx pygraphviz")


def create_interactive_plot(data: Dict[str, List], plot_type: str = 'line',
                          title: str = "交互式图表", save_path: str = None):
    """
    创建交互式图表
    
    Args:
        data: 数据字典
        plot_type: 图表类型 ('line', 'bar', 'scatter')
        title: 图表标题
        save_path: 保存路径（可选）
    """
    try:
        if plot_type == 'line':
            fig = go.Figure()
            
            for key, values in data.items():
                fig.add_trace(go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    name=key,
                    mode='lines+markers'
                ))
            
        elif plot_type == 'bar':
            fig = px.bar(x=list(data.keys()), y=list(data.values()))
            
        elif plot_type == 'scatter':
            if len(data) >= 2:
                keys = list(data.keys())
                fig = px.scatter(x=data[keys[0]], y=data[keys[1]])
            else:
                raise ValueError("散点图需要至少2个数据系列")
        
        fig.update_layout(
            title=title,
            xaxis_title="X轴",
            yaxis_title="Y轴",
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
    except ImportError:
        print("需要安装plotly来创建交互式图表")
        print("pip install plotly")


def save_all_plots(save_dir: str = "plots"):
    """
    保存所有生成的图表
    
    Args:
        save_dir: 保存目录
    """
    import os
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"所有图表将保存到: {save_dir}")
    
    # 返回保存目录路径，供其他函数使用
    return save_dir


# 设置默认样式
def set_plot_style(style: str = 'seaborn'):
    """
    设置绘图样式
    
    Args:
        style: 样式名称
    """
    available_styles = plt.style.available
    
    if style in available_styles:
        plt.style.use(style)
        print(f"已设置绘图样式: {style}")
    else:
        print(f"样式 '{style}' 不可用")
        print(f"可用样式: {available_styles}")


# 颜色调色板
def get_color_palette(n_colors: int = 10, palette: str = 'husl'):
    """
    获取颜色调色板
    
    Args:
        n_colors: 颜色数量
        palette: 调色板名称
    
    Returns:
        颜色列表
    """
    return sns.color_palette(palette, n_colors) 