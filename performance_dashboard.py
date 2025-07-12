"""
性能仪表板 - 算法性能监控和比较
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import psutil
import threading
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """性能指标"""
    algorithm_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    accuracy: float
    throughput: float
    error_rate: float
    timestamp: datetime

class PerformanceDashboard:
    """性能仪表板"""
    
    def __init__(self):
        self.metrics_history = []
        self.real_time_data = []
        self.benchmark_results = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """开始实时监控"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_system, 
            args=(interval,)
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """停止实时监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitor_system(self, interval: float):
        """系统监控线程"""
        while self.monitoring_active:
            # 获取系统指标
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            # 记录实时数据
            self.real_time_data.append({
                'timestamp': datetime.now(),
                'cpu_usage': cpu_percent,
                'memory_usage': memory_info.percent,
                'memory_available': memory_info.available / (1024**3),  # GB
                'memory_total': memory_info.total / (1024**3)  # GB
            })
            
            # 保持最近1000个数据点
            if len(self.real_time_data) > 1000:
                self.real_time_data.pop(0)
            
            time.sleep(interval)
    
    def add_metrics(self, metrics: PerformanceMetrics):
        """添加性能指标"""
        self.metrics_history.append(metrics)
    
    def run_benchmark(self, algorithms: Dict[str, Any], 
                     test_data: Dict[str, Any],
                     iterations: int = 10) -> Dict[str, Any]:
        """运行基准测试"""
        print("🏁 开始基准测试...")
        
        results = {}
        
        for alg_name, algorithm in algorithms.items():
            print(f"  测试 {alg_name}...")
            
            alg_results = {
                'execution_times': [],
                'memory_usage': [],
                'cpu_usage': [],
                'accuracy': [],
                'throughput': []
            }
            
            for i in range(iterations):
                # 清理内存
                import gc
                gc.collect()
                
                # 记录初始状态
                initial_memory = psutil.virtual_memory().used / (1024**2)  # MB
                initial_cpu = psutil.cpu_percent(interval=0.1)
                
                # 运行算法
                start_time = time.time()
                result = self._run_algorithm(algorithm, test_data)
                end_time = time.time()
                
                # 记录结果
                execution_time = end_time - start_time
                final_memory = psutil.virtual_memory().used / (1024**2)  # MB
                memory_used = final_memory - initial_memory
                final_cpu = psutil.cpu_percent(interval=0.1)
                
                alg_results['execution_times'].append(execution_time)
                alg_results['memory_usage'].append(memory_used)
                alg_results['cpu_usage'].append(final_cpu - initial_cpu)
                
                if 'accuracy' in result:
                    alg_results['accuracy'].append(result['accuracy'])
                
                if execution_time > 0:
                    throughput = len(test_data.get('X', [])) / execution_time
                    alg_results['throughput'].append(throughput)
            
            # 计算统计信息
            results[alg_name] = {
                'mean_execution_time': np.mean(alg_results['execution_times']),
                'std_execution_time': np.std(alg_results['execution_times']),
                'mean_memory_usage': np.mean(alg_results['memory_usage']),
                'std_memory_usage': np.std(alg_results['memory_usage']),
                'mean_cpu_usage': np.mean(alg_results['cpu_usage']),
                'std_cpu_usage': np.std(alg_results['cpu_usage']),
                'mean_accuracy': np.mean(alg_results['accuracy']) if alg_results['accuracy'] else 0,
                'std_accuracy': np.std(alg_results['accuracy']) if alg_results['accuracy'] else 0,
                'mean_throughput': np.mean(alg_results['throughput']) if alg_results['throughput'] else 0,
                'std_throughput': np.std(alg_results['throughput']) if alg_results['throughput'] else 0,
                'raw_data': alg_results
            }
        
        self.benchmark_results = results
        return results
    
    def _run_algorithm(self, algorithm: Any, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行算法"""
        # 这里应该根据算法类型调用相应的方法
        # 为了演示，我们模拟一个简单的执行
        
        if hasattr(algorithm, 'fit') and hasattr(algorithm, 'predict'):
            # 机器学习算法
            if 'X' in test_data and 'y' in test_data:
                algorithm.fit(test_data['X'], test_data['y'])
                predictions = algorithm.predict(test_data['X'])
                accuracy = np.mean(predictions == test_data['y'])
                return {'accuracy': accuracy}
        
        elif hasattr(algorithm, 'run'):
            # 搜索算法
            if 'problem' in test_data:
                result = algorithm.run(test_data['problem'])
                return result
        
        # 默认情况
        time.sleep(0.001)  # 模拟计算
        return {'accuracy': np.random.uniform(0.7, 0.95)}
    
    def create_performance_comparison_chart(self) -> go.Figure:
        """创建性能比较图表"""
        if not self.benchmark_results:
            return go.Figure()
        
        algorithms = list(self.benchmark_results.keys())
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('执行时间', '内存使用', '准确率', '吞吐量'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 执行时间
        execution_times = [self.benchmark_results[alg]['mean_execution_time'] for alg in algorithms]
        execution_stds = [self.benchmark_results[alg]['std_execution_time'] for alg in algorithms]
        
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=execution_times,
                error_y=dict(type='data', array=execution_stds),
                name='执行时间',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 内存使用
        memory_usage = [self.benchmark_results[alg]['mean_memory_usage'] for alg in algorithms]
        memory_stds = [self.benchmark_results[alg]['std_memory_usage'] for alg in algorithms]
        
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=memory_usage,
                error_y=dict(type='data', array=memory_stds),
                name='内存使用',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 准确率
        accuracy = [self.benchmark_results[alg]['mean_accuracy'] for alg in algorithms]
        accuracy_stds = [self.benchmark_results[alg]['std_accuracy'] for alg in algorithms]
        
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=accuracy,
                error_y=dict(type='data', array=accuracy_stds),
                name='准确率',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 吞吐量
        throughput = [self.benchmark_results[alg]['mean_throughput'] for alg in algorithms]
        throughput_stds = [self.benchmark_results[alg]['std_throughput'] for alg in algorithms]
        
        fig.add_trace(
            go.Bar(
                x=algorithms,
                y=throughput,
                error_y=dict(type='data', array=throughput_stds),
                name='吞吐量',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="算法性能比较",
            height=600,
            showlegend=False
        )
        
        # 更新轴标签
        fig.update_xaxes(title_text="算法", row=1, col=1)
        fig.update_xaxes(title_text="算法", row=1, col=2)
        fig.update_xaxes(title_text="算法", row=2, col=1)
        fig.update_xaxes(title_text="算法", row=2, col=2)
        
        fig.update_yaxes(title_text="时间(秒)", row=1, col=1)
        fig.update_yaxes(title_text="内存(MB)", row=1, col=2)
        fig.update_yaxes(title_text="准确率", row=2, col=1)
        fig.update_yaxes(title_text="样本/秒", row=2, col=2)
        
        return fig
    
    def create_real_time_monitoring_chart(self) -> go.Figure:
        """创建实时监控图表"""
        if not self.real_time_data:
            return go.Figure()
        
        df = pd.DataFrame(self.real_time_data)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU使用率', '内存使用'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # CPU使用率
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cpu_usage'],
                mode='lines',
                name='CPU使用率',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 内存使用
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['memory_usage'],
                mode='lines',
                name='内存使用率',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title_text="实时系统监控",
            height=500,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="时间", row=2, col=1)
        fig.update_yaxes(title_text="CPU使用率 (%)", row=1, col=1)
        fig.update_yaxes(title_text="内存使用率 (%)", row=2, col=1)
        
        return fig
    
    def create_performance_trend_chart(self) -> go.Figure:
        """创建性能趋势图表"""
        if not self.metrics_history:
            return go.Figure()
        
        df = pd.DataFrame([{
            'timestamp': m.timestamp,
            'algorithm': m.algorithm_name,
            'execution_time': m.execution_time,
            'memory_usage': m.memory_usage,
            'accuracy': m.accuracy,
            'throughput': m.throughput
        } for m in self.metrics_history])
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('执行时间趋势', '内存使用趋势', '准确率趋势', '吞吐量趋势'),
            shared_xaxes=True
        )
        
        algorithms = df['algorithm'].unique()
        colors = px.colors.qualitative.Set1[:len(algorithms)]
        
        for i, alg in enumerate(algorithms):
            alg_data = df[df['algorithm'] == alg]
            
            # 执行时间
            fig.add_trace(
                go.Scatter(
                    x=alg_data['timestamp'],
                    y=alg_data['execution_time'],
                    mode='lines+markers',
                    name=alg,
                    line=dict(color=colors[i]),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # 内存使用
            fig.add_trace(
                go.Scatter(
                    x=alg_data['timestamp'],
                    y=alg_data['memory_usage'],
                    mode='lines+markers',
                    name=alg,
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # 准确率
            fig.add_trace(
                go.Scatter(
                    x=alg_data['timestamp'],
                    y=alg_data['accuracy'],
                    mode='lines+markers',
                    name=alg,
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # 吞吐量
            fig.add_trace(
                go.Scatter(
                    x=alg_data['timestamp'],
                    y=alg_data['throughput'],
                    mode='lines+markers',
                    name=alg,
                    line=dict(color=colors[i]),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="算法性能趋势",
            height=600
        )
        
        fig.update_yaxes(title_text="时间(秒)", row=1, col=1)
        fig.update_yaxes(title_text="内存(MB)", row=1, col=2)
        fig.update_yaxes(title_text="准确率", row=2, col=1)
        fig.update_yaxes(title_text="样本/秒", row=2, col=2)
        
        return fig
    
    def create_algorithm_ranking_chart(self) -> go.Figure:
        """创建算法排名图表"""
        if not self.benchmark_results:
            return go.Figure()
        
        # 计算综合评分
        algorithms = list(self.benchmark_results.keys())
        scores = []
        
        for alg in algorithms:
            result = self.benchmark_results[alg]
            
            # 归一化各指标（越小越好的指标需要取倒数）
            time_score = 1 / (result['mean_execution_time'] + 1e-6)
            memory_score = 1 / (result['mean_memory_usage'] + 1e-6)
            accuracy_score = result['mean_accuracy']
            throughput_score = result['mean_throughput']
            
            # 加权综合评分
            total_score = (time_score * 0.3 + memory_score * 0.2 + 
                          accuracy_score * 0.3 + throughput_score * 0.2)
            
            scores.append({
                'algorithm': alg,
                'time_score': time_score,
                'memory_score': memory_score,
                'accuracy_score': accuracy_score,
                'throughput_score': throughput_score,
                'total_score': total_score
            })
        
        # 排序
        scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        df = pd.DataFrame(scores)
        
        fig = go.Figure()
        
        # 堆叠条形图
        fig.add_trace(go.Bar(
            x=df['algorithm'],
            y=df['time_score'],
            name='时间效率',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=df['algorithm'],
            y=df['memory_score'],
            name='内存效率',
            marker_color='lightgreen'
        ))
        
        fig.add_trace(go.Bar(
            x=df['algorithm'],
            y=df['accuracy_score'],
            name='准确率',
            marker_color='lightcoral'
        ))
        
        fig.add_trace(go.Bar(
            x=df['algorithm'],
            y=df['throughput_score'],
            name='吞吐量',
            marker_color='lightyellow'
        ))
        
        fig.update_layout(
            title='算法综合性能排名',
            xaxis_title='算法',
            yaxis_title='评分',
            barmode='stack',
            height=500
        )
        
        return fig
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        if not self.benchmark_results:
            return "没有可用的基准测试结果"
        
        report = []
        report.append("# 算法性能分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 算法概览
        report.append("## 算法概览")
        report.append(f"测试算法数量: {len(self.benchmark_results)}")
        report.append(f"算法列表: {', '.join(self.benchmark_results.keys())}")
        report.append("")
        
        # 性能排名
        algorithms = list(self.benchmark_results.keys())
        
        # 按执行时间排序
        time_ranking = sorted(algorithms, 
                             key=lambda x: self.benchmark_results[x]['mean_execution_time'])
        
        report.append("## 性能排名")
        report.append("")
        report.append("### 执行时间排名 (从快到慢)")
        for i, alg in enumerate(time_ranking, 1):
            time_val = self.benchmark_results[alg]['mean_execution_time']
            report.append(f"{i}. {alg}: {time_val:.4f} 秒")
        
        report.append("")
        
        # 内存使用排名
        memory_ranking = sorted(algorithms, 
                               key=lambda x: self.benchmark_results[x]['mean_memory_usage'])
        
        report.append("### 内存使用排名 (从低到高)")
        for i, alg in enumerate(memory_ranking, 1):
            memory_val = self.benchmark_results[alg]['mean_memory_usage']
            report.append(f"{i}. {alg}: {memory_val:.2f} MB")
        
        report.append("")
        
        # 准确率排名
        accuracy_ranking = sorted(algorithms, 
                                 key=lambda x: self.benchmark_results[x]['mean_accuracy'], 
                                 reverse=True)
        
        report.append("### 准确率排名 (从高到低)")
        for i, alg in enumerate(accuracy_ranking, 1):
            accuracy_val = self.benchmark_results[alg]['mean_accuracy']
            report.append(f"{i}. {alg}: {accuracy_val:.3f}")
        
        report.append("")
        
        # 详细统计
        report.append("## 详细统计")
        report.append("")
        
        for alg in algorithms:
            result = self.benchmark_results[alg]
            report.append(f"### {alg}")
            report.append(f"- 平均执行时间: {result['mean_execution_time']:.4f} ± {result['std_execution_time']:.4f} 秒")
            report.append(f"- 平均内存使用: {result['mean_memory_usage']:.2f} ± {result['std_memory_usage']:.2f} MB")
            report.append(f"- 平均准确率: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
            report.append(f"- 平均吞吐量: {result['mean_throughput']:.2f} ± {result['std_throughput']:.2f} 样本/秒")
            report.append("")
        
        # 建议
        report.append("## 性能建议")
        report.append("")
        
        fastest_alg = time_ranking[0]
        most_accurate_alg = accuracy_ranking[0]
        least_memory_alg = memory_ranking[0]
        
        report.append(f"- **最快算法**: {fastest_alg}")
        report.append(f"- **最准确算法**: {most_accurate_alg}")
        report.append(f"- **最省内存算法**: {least_memory_alg}")
        
        if fastest_alg == most_accurate_alg:
            report.append(f"- **推荐算法**: {fastest_alg} (速度和准确率均最优)")
        else:
            report.append(f"- **速度优先**: 选择 {fastest_alg}")
            report.append(f"- **准确率优先**: 选择 {most_accurate_alg}")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "performance_results.json"):
        """保存结果到文件"""
        data = {
            'benchmark_results': self.benchmark_results,
            'metrics_history': [
                {
                    'algorithm_name': m.algorithm_name,
                    'execution_time': m.execution_time,
                    'memory_usage': m.memory_usage,
                    'cpu_usage': m.cpu_usage,
                    'accuracy': m.accuracy,
                    'throughput': m.throughput,
                    'error_rate': m.error_rate,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self.metrics_history
            ],
            'real_time_data': [
                {
                    'timestamp': d['timestamp'].isoformat(),
                    'cpu_usage': d['cpu_usage'],
                    'memory_usage': d['memory_usage'],
                    'memory_available': d['memory_available'],
                    'memory_total': d['memory_total']
                }
                for d in self.real_time_data
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到 {filename}")
    
    def load_results(self, filename: str = "performance_results.json"):
        """从文件加载结果"""
        if not Path(filename).exists():
            print(f"文件 {filename} 不存在")
            return
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.benchmark_results = data.get('benchmark_results', {})
        
        # 加载指标历史
        self.metrics_history = []
        for m in data.get('metrics_history', []):
            self.metrics_history.append(PerformanceMetrics(
                algorithm_name=m['algorithm_name'],
                execution_time=m['execution_time'],
                memory_usage=m['memory_usage'],
                cpu_usage=m['cpu_usage'],
                accuracy=m['accuracy'],
                throughput=m['throughput'],
                error_rate=m['error_rate'],
                timestamp=datetime.fromisoformat(m['timestamp'])
            ))
        
        # 加载实时数据
        self.real_time_data = []
        for d in data.get('real_time_data', []):
            self.real_time_data.append({
                'timestamp': datetime.fromisoformat(d['timestamp']),
                'cpu_usage': d['cpu_usage'],
                'memory_usage': d['memory_usage'],
                'memory_available': d['memory_available'],
                'memory_total': d['memory_total']
            })
        
        print(f"结果已从 {filename} 加载")

# 模拟算法类
class MockAlgorithm:
    """模拟算法类"""
    
    def __init__(self, name: str, base_time: float = 1.0, 
                 base_memory: float = 10.0, base_accuracy: float = 0.85):
        self.name = name
        self.base_time = base_time
        self.base_memory = base_memory
        self.base_accuracy = base_accuracy
    
    def fit(self, X, y):
        """训练"""
        time.sleep(self.base_time * np.random.uniform(0.8, 1.2))
    
    def predict(self, X):
        """预测"""
        # 模拟预测结果
        return np.random.randint(0, 2, len(X))

# 演示函数
def demo_performance_dashboard():
    """演示性能仪表板"""
    print("📊 性能仪表板演示")
    print("="*50)
    
    # 创建仪表板
    dashboard = PerformanceDashboard()
    
    # 创建模拟算法
    algorithms = {
        'FastAlgorithm': MockAlgorithm('FastAlgorithm', base_time=0.5, base_accuracy=0.80),
        'AccurateAlgorithm': MockAlgorithm('AccurateAlgorithm', base_time=2.0, base_accuracy=0.95),
        'BalancedAlgorithm': MockAlgorithm('BalancedAlgorithm', base_time=1.0, base_accuracy=0.88),
        'MemoryEfficientAlgorithm': MockAlgorithm('MemoryEfficientAlgorithm', 
                                                  base_time=1.5, base_memory=5.0, base_accuracy=0.85)
    }
    
    # 生成测试数据
    np.random.seed(42)
    test_data = {
        'X': np.random.randn(1000, 10),
        'y': np.random.randint(0, 2, 1000)
    }
    
    # 启动实时监控
    print("🔍 启动实时监控...")
    dashboard.start_monitoring(interval=0.5)
    
    # 运行基准测试
    print("🏁 运行基准测试...")
    benchmark_results = dashboard.run_benchmark(algorithms, test_data, iterations=5)
    
    # 停止监控
    dashboard.stop_monitoring()
    
    # 显示结果
    print("\n📈 基准测试结果:")
    for alg_name, result in benchmark_results.items():
        print(f"\n{alg_name}:")
        print(f"  平均执行时间: {result['mean_execution_time']:.4f} ± {result['std_execution_time']:.4f} 秒")
        print(f"  平均内存使用: {result['mean_memory_usage']:.2f} ± {result['std_memory_usage']:.2f} MB")
        print(f"  平均准确率: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
        print(f"  平均吞吐量: {result['mean_throughput']:.2f} ± {result['std_throughput']:.2f} 样本/秒")
    
    # 生成报告
    print("\n📝 生成性能报告...")
    report = dashboard.generate_performance_report()
    
    # 保存报告
    with open("performance_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("报告已保存到 performance_report.md")
    
    # 保存结果
    dashboard.save_results("performance_results.json")
    
    print("\n✅ 演示完成！")

if __name__ == "__main__":
    demo_performance_dashboard() 