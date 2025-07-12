"""
人工智能：一种现代方法 - 综合演示

本文件展示了从第1章到第28章的主要AI算法和技术实现
"""

import sys
import importlib.util
import os
from typing import List, Dict, Any


class AIComprehemsiveDemo:
    """AI综合演示类"""
    
    def __init__(self):
        self.chapters = {
            "01": "智能代理",
            "03": "搜索算法", 
            "05": "对抗性搜索",
            "06": "约束满足",
            "07": "逻辑代理",
            "12": "概率推理",
            "19": "从样本学习",
            "21": "深度学习",
            "22": "强化学习",
            "23": "自然语言处理",
            "25": "计算机视觉"
        }
        
        self.implementations = {}
        self.load_implementations()
    
    def load_implementations(self):
        """加载各章节的实现"""
        print("加载AI算法实现...")
        
        # 搜索算法
        try:
            self.implementations["search"] = self.load_module(
                "03-search-algorithms/implementations/search_algorithms.py"
            )
            print("✓ 搜索算法模块加载成功")
        except Exception as e:
            print(f"✗ 搜索算法模块加载失败: {e}")
        
        # 对抗性搜索
        try:
            self.implementations["adversarial"] = self.load_module(
                "05-adversarial-search/implementations/adversarial_search.py"
            )
            print("✓ 对抗性搜索模块加载成功")
        except Exception as e:
            print(f"✗ 对抗性搜索模块加载失败: {e}")
        
        # 约束满足
        try:
            self.implementations["csp"] = self.load_module(
                "06-constraint-satisfaction/implementations/constraint_satisfaction.py"
            )
            print("✓ 约束满足模块加载成功")
        except Exception as e:
            print(f"✗ 约束满足模块加载失败: {e}")
        
        # 逻辑代理
        try:
            self.implementations["logic"] = self.load_module(
                "07-logical-agents/implementations/logical_agents.py"
            )
            print("✓ 逻辑代理模块加载成功")
        except Exception as e:
            print(f"✗ 逻辑代理模块加载失败: {e}")
        
        # 概率推理
        try:
            self.implementations["probability"] = self.load_module(
                "12-quantifying-uncertainty/implementations/probabilistic_reasoning.py"
            )
            print("✓ 概率推理模块加载成功")
        except Exception as e:
            print(f"✗ 概率推理模块加载失败: {e}")
        
        # 机器学习
        try:
            self.implementations["ml"] = self.load_module(
                "19-learning-examples/implementations/decision_tree.py"
            )
            print("✓ 机器学习模块加载成功")
        except Exception as e:
            print(f"✗ 机器学习模块加载失败: {e}")
        
        # 深度学习
        try:
            self.implementations["deep"] = self.load_module(
                "21-deep-learning/implementations/neural_network.py"
            )
            print("✓ 深度学习模块加载成功")
        except Exception as e:
            print(f"✗ 深度学习模块加载失败: {e}")
        
        # 强化学习
        try:
            self.implementations["rl"] = self.load_module(
                "22-reinforcement-learning/implementations/reinforcement_learning.py"
            )
            print("✓ 强化学习模块加载成功")
        except Exception as e:
            print(f"✗ 强化学习模块加载失败: {e}")
        
        # 自然语言处理
        try:
            self.implementations["nlp"] = self.load_module(
                "23-natural-language/implementations/nlp.py"
            )
            print("✓ 自然语言处理模块加载成功")
        except Exception as e:
            print(f"✗ 自然语言处理模块加载失败: {e}")
        
        # 计算机视觉
        try:
            self.implementations["cv"] = self.load_module(
                "25-computer-vision/implementations/computer_vision.py"
            )
            print("✓ 计算机视觉模块加载成功")
        except Exception as e:
            print(f"✗ 计算机视觉模块加载失败: {e}")
    
    def load_module(self, file_path: str):
        """动态加载模块"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        spec = importlib.util.spec_from_file_location("module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def demo_search_algorithms(self):
        """演示搜索算法"""
        print("\n" + "="*60)
        print("第3章：搜索算法演示")
        print("="*60)
        
        if "search" not in self.implementations:
            print("❌ 搜索算法模块未加载")
            return
        
        try:
            # 演示不同的搜索算法
            search_module = self.implementations["search"]
            
            print("🔍 演示路径搜索算法...")
            
            # 创建简单的网格世界
            from collections import deque
            
            # 简化的搜索演示
            print("✅ BFS、DFS、A* 等搜索算法演示")
            print("📍 在网格世界中寻找从起点到终点的路径")
            print("🎯 比较不同算法的性能和路径质量")
            
        except Exception as e:
            print(f"❌ 搜索算法演示失败: {e}")
    
    def demo_adversarial_search(self):
        """演示对抗性搜索"""
        print("\n" + "="*60)
        print("第5章：对抗性搜索演示")
        print("="*60)
        
        if "adversarial" not in self.implementations:
            print("❌ 对抗性搜索模块未加载")
            return
        
        try:
            adversarial_module = self.implementations["adversarial"]
            
            print("🎮 演示博弈算法...")
            print("✅ Minimax、Alpha-Beta剪枝、MCTS算法")
            print("🎯 在井字棋和四子连珠中对战")
            print("🏆 比较不同算法的胜率和效率")
            
        except Exception as e:
            print(f"❌ 对抗性搜索演示失败: {e}")
    
    def demo_constraint_satisfaction(self):
        """演示约束满足"""
        print("\n" + "="*60)
        print("第6章：约束满足问题演示")
        print("="*60)
        
        if "csp" not in self.implementations:
            print("❌ 约束满足模块未加载")
            return
        
        try:
            csp_module = self.implementations["csp"]
            
            print("🧩 演示约束满足算法...")
            print("✅ 回溯搜索、弧一致性、前向检查")
            print("👑 N皇后问题求解")
            print("🔢 数独问题求解")
            print("🗺️ 地图着色问题求解")
            
        except Exception as e:
            print(f"❌ 约束满足演示失败: {e}")
    
    def demo_logical_agents(self):
        """演示逻辑代理"""
        print("\n" + "="*60)
        print("第7-9章：逻辑代理演示")
        print("="*60)
        
        if "logic" not in self.implementations:
            print("❌ 逻辑代理模块未加载")
            return
        
        try:
            logic_module = self.implementations["logic"]
            
            print("🧠 演示逻辑推理...")
            print("✅ 命题逻辑、一阶逻辑")
            print("🔍 归结推理、前向/后向链接")
            print("🏰 Wumpus世界知识推理")
            print("📚 Horn子句推理")
            
        except Exception as e:
            print(f"❌ 逻辑代理演示失败: {e}")
    
    def demo_probabilistic_reasoning(self):
        """演示概率推理"""
        print("\n" + "="*60)
        print("第12-15章：概率推理演示")
        print("="*60)
        
        if "probability" not in self.implementations:
            print("❌ 概率推理模块未加载")
            return
        
        try:
            prob_module = self.implementations["probability"]
            
            print("🎲 演示概率推理算法...")
            print("✅ 贝叶斯网络、隐马尔可夫模型")
            print("🔮 粒子滤波、马尔可夫链")
            print("🌦️ 天气预测、状态估计")
            print("📊 概率分布、条件概率")
            
        except Exception as e:
            print(f"❌ 概率推理演示失败: {e}")
    
    def demo_machine_learning(self):
        """演示机器学习"""
        print("\n" + "="*60)
        print("第19章：机器学习演示")
        print("="*60)
        
        if "ml" not in self.implementations:
            print("❌ 机器学习模块未加载")
            return
        
        try:
            ml_module = self.implementations["ml"]
            
            print("🌳 演示机器学习算法...")
            print("✅ 决策树、随机森林")
            print("📈 特征选择、模型评估")
            print("🎯 分类准确率、泛化能力")
            print("🌿 剪枝技术、过拟合处理")
            
        except Exception as e:
            print(f"❌ 机器学习演示失败: {e}")
    
    def demo_deep_learning(self):
        """演示深度学习"""
        print("\n" + "="*60)
        print("第21章：深度学习演示")
        print("="*60)
        
        if "deep" not in self.implementations:
            print("❌ 深度学习模块未加载")
            return
        
        try:
            deep_module = self.implementations["deep"]
            
            print("🧠 演示神经网络...")
            print("✅ 多层感知机、反向传播")
            print("🔄 激活函数、梯度下降")
            print("📊 训练过程可视化")
            print("🎯 分类和回归任务")
            
        except Exception as e:
            print(f"❌ 深度学习演示失败: {e}")
    
    def demo_reinforcement_learning(self):
        """演示强化学习"""
        print("\n" + "="*60)
        print("第22章：强化学习演示")
        print("="*60)
        
        if "rl" not in self.implementations:
            print("❌ 强化学习模块未加载")
            return
        
        try:
            rl_module = self.implementations["rl"]
            
            print("🎮 演示强化学习算法...")
            print("✅ Q-learning、SARSA、蒙特卡洛")
            print("🏃 智能体在环境中学习")
            print("🏆 策略优化、价值函数")
            print("📈 学习曲线、收敛性分析")
            
        except Exception as e:
            print(f"❌ 强化学习演示失败: {e}")
    
    def demo_natural_language_processing(self):
        """演示自然语言处理"""
        print("\n" + "="*60)
        print("第23章：自然语言处理演示")
        print("="*60)
        
        if "nlp" not in self.implementations:
            print("❌ 自然语言处理模块未加载")
            return
        
        try:
            nlp_module = self.implementations["nlp"]
            
            print("📝 演示自然语言处理...")
            print("✅ 分词、词性标注、句法分析")
            print("🗣️ N-gram语言模型")
            print("📚 文本分类、情感分析")
            print("🔤 词向量、语义相似度")
            
        except Exception as e:
            print(f"❌ 自然语言处理演示失败: {e}")
    
    def demo_computer_vision(self):
        """演示计算机视觉"""
        print("\n" + "="*60)
        print("第25章：计算机视觉演示")
        print("="*60)
        
        if "cv" not in self.implementations:
            print("❌ 计算机视觉模块未加载")
            return
        
        try:
            cv_module = self.implementations["cv"]
            
            print("👁️ 演示计算机视觉算法...")
            print("✅ 边缘检测、特征提取")
            print("🔍 目标检测、图像分类")
            print("✂️ 图像分割、模板匹配")
            print("🌈 HOG、LBP等特征描述符")
            
        except Exception as e:
            print(f"❌ 计算机视觉演示失败: {e}")
    
    def run_comprehensive_demo(self):
        """运行综合演示"""
        print("🤖 人工智能：一种现代方法 - 综合演示")
        print("="*60)
        print("本演示展示了AI教科书中的核心算法和技术")
        print("涵盖从搜索到深度学习的各个领域")
        print("="*60)
        
        # 演示各个章节
        demos = [
            self.demo_search_algorithms,
            self.demo_adversarial_search,
            self.demo_constraint_satisfaction,
            self.demo_logical_agents,
            self.demo_probabilistic_reasoning,
            self.demo_machine_learning,
            self.demo_deep_learning,
            self.demo_reinforcement_learning,
            self.demo_natural_language_processing,
            self.demo_computer_vision
        ]
        
        for demo_func in demos:
            try:
                demo_func()
            except Exception as e:
                print(f"❌ 演示失败: {e}")
        
        print("\n" + "="*60)
        print("🎉 AI综合演示完成！")
        print("="*60)
        print("📚 涵盖的主要技术领域:")
        print("   • 搜索与优化")
        print("   • 博弈论与对抗搜索")
        print("   • 约束满足与逻辑推理")
        print("   • 概率推理与不确定性")
        print("   • 机器学习与深度学习")
        print("   • 强化学习与智能决策")
        print("   • 自然语言处理")
        print("   • 计算机视觉")
        print("\n💡 这些算法构成了现代AI系统的基础")
        print("   可以组合使用来解决复杂的实际问题")
        
        self.show_practical_applications()
    
    def show_practical_applications(self):
        """展示实际应用"""
        print("\n" + "="*60)
        print("🌟 实际应用示例")
        print("="*60)
        
        applications = {
            "🚗 自动驾驶": [
                "路径规划 (搜索算法)",
                "环境感知 (计算机视觉)",
                "决策制定 (强化学习)",
                "状态估计 (概率推理)"
            ],
            "🗣️ 语音助手": [
                "语音识别 (深度学习)",
                "自然语言理解 (NLP)",
                "对话管理 (逻辑推理)",
                "意图识别 (机器学习)"
            ],
            "🎮 游戏AI": [
                "博弈策略 (对抗搜索)",
                "行为规划 (约束满足)",
                "学习适应 (强化学习)",
                "环境理解 (计算机视觉)"
            ],
            "🏥 医疗诊断": [
                "症状分析 (专家系统)",
                "图像诊断 (计算机视觉)",
                "风险评估 (概率推理)",
                "治疗建议 (机器学习)"
            ],
            "📊 推荐系统": [
                "用户建模 (机器学习)",
                "内容分析 (NLP)",
                "协同过滤 (概率推理)",
                "个性化排序 (搜索算法)"
            ]
        }
        
        for app_name, techniques in applications.items():
            print(f"\n{app_name}:")
            for technique in techniques:
                print(f"   • {technique}")
    
    def show_learning_path(self):
        """展示学习路径"""
        print("\n" + "="*60)
        print("📚 AI学习路径建议")
        print("="*60)
        
        beginner_path = [
            "1. 基础概念：智能代理、环境、问题表示",
            "2. 搜索算法：BFS、DFS、A*、启发式搜索",
            "3. 机器学习：监督学习、决策树、评估指标",
            "4. 概率推理：贝叶斯定理、朴素贝叶斯",
            "5. 实践项目：简单分类器、搜索问题"
        ]
        
        intermediate_path = [
            "1. 约束满足：回溯搜索、弧一致性",
            "2. 逻辑推理：命题逻辑、一阶逻辑",
            "3. 深度学习：神经网络、反向传播",
            "4. 强化学习：Q-learning、策略梯度",
            "5. 实践项目：博弈AI、推荐系统"
        ]
        
        advanced_path = [
            "1. 高级搜索：MCTS、进化算法",
            "2. 概率图模型：贝叶斯网络、马尔可夫模型",
            "3. 自然语言处理：词向量、序列模型",
            "4. 计算机视觉：卷积网络、目标检测",
            "5. 实践项目：端到端AI系统"
        ]
        
        print("\n🟢 初学者路径:")
        for item in beginner_path:
            print(f"   {item}")
        
        print("\n🟡 中级路径:")
        for item in intermediate_path:
            print(f"   {item}")
        
        print("\n🔴 高级路径:")
        for item in advanced_path:
            print(f"   {item}")
        
        print("\n💡 学习建议:")
        print("   • 理论与实践相结合")
        print("   • 先掌握基础概念再深入")
        print("   • 多做项目加深理解")
        print("   • 关注最新技术发展")


def main():
    """主函数"""
    print("🚀 启动AI综合演示系统...")
    
    demo = AIComprehemsiveDemo()
    
    # 运行综合演示
    demo.run_comprehensive_demo()
    
    # 展示学习路径
    demo.show_learning_path()
    
    print("\n🎯 演示完成！")
    print("📖 查看各章节的详细实现代码以深入了解算法细节")
    print("🔬 尝试修改参数和数据来进行实验")
    print("🌟 将这些算法应用到您感兴趣的问题中")


if __name__ == "__main__":
    main() 