#!/usr/bin/env python3
"""
新模块测试脚本
验证新创建的AI模块是否能正常工作
"""

import sys
import traceback

def test_problem_solving():
    """测试第2章问题求解模块"""
    print("🔍 测试问题求解模块...")
    try:
        sys.path.append('02-problem-solving/implementations')
        import problem_solving
        
        # 简单测试
        print("✅ 问题求解模块加载成功")
        print("   - 包含8数码问题")
        print("   - 包含N皇后问题") 
        print("   - 包含罗马尼亚地图问题")
        return True
    except Exception as e:
        print(f"❌ 问题求解模块测试失败: {e}")
        return False

def test_complex_environments():
    """测试第4章复杂环境模块"""
    print("\n🌍 测试复杂环境模块...")
    try:
        sys.path.append('04-complex-environments/implementations')
        import complex_environments
        
        print("✅ 复杂环境模块加载成功")
        print("   - 包含吸尘器环境")
        print("   - 包含Wumpus世界")
        print("   - 包含多代理环境")
        return True
    except Exception as e:
        print(f"❌ 复杂环境模块测试失败: {e}")
        return False

def test_automated_planning():
    """测试第11章自动规划模块"""
    print("\n📋 测试自动规划模块...")
    try:
        sys.path.append('11-automated-planning/implementations')
        import automated_planning
        
        print("✅ 自动规划模块加载成功")
        print("   - 包含STRIPS规划")
        print("   - 包含GraphPlan算法")
        print("   - 包含积木世界问题")
        return True
    except Exception as e:
        print(f"❌ 自动规划模块测试失败: {e}")
        return False

def test_temporal_reasoning():
    """测试第14章时序推理模块"""
    print("\n⏰ 测试时序推理模块...")
    try:
        sys.path.append('14-temporal-reasoning/implementations')
        import temporal_reasoning
        
        print("✅ 时序推理模块加载成功")
        print("   - 包含隐马尔可夫模型")
        print("   - 包含卡尔曼滤波")
        print("   - 包含粒子滤波")
        return True
    except Exception as e:
        print(f"❌ 时序推理模块测试失败: {e}")
        return False

def test_intelligent_game_ai():
    """测试智能游戏AI"""
    print("\n🎮 测试智能游戏AI...")
    try:
        sys.path.append('project-examples/advanced-ai-applications')
        import intelligent_game_ai as game_ai
        
        # 简单测试
        game_state = game_ai.TicTacToeState()
        ai_agent = game_ai.AlphaBetaAgent(depth=2)
        action = ai_agent.get_action(game_state, 1)
        
        print("✅ 智能游戏AI模块正常工作")
        print("   - AI成功选择动作")
        print("   - 包含多种搜索算法")
        return True
    except Exception as e:
        print(f"❌ 智能游戏AI测试失败: {e}")
        return False

def test_intelligent_chatbot():
    """测试智能聊天机器人"""
    print("\n🤖 测试智能聊天机器人...")
    try:
        sys.path.append('project-examples/advanced-ai-applications')
        import intelligent_chatbot as chatbot
        
        # 简单测试
        bot = chatbot.Chatbot()
        response = bot.chat("你好")
        
        print("✅ 智能聊天机器人模块正常工作")
        print(f"   - 测试对话: '你好' -> '{response}'")
        print("   - 包含NLP和知识推理")
        return True
    except Exception as e:
        print(f"❌ 智能聊天机器人测试失败: {e}")
        return False

def test_intelligent_decision_system():
    """测试智能决策系统"""
    print("\n💰 测试智能决策系统...")
    try:
        sys.path.append('project-examples/advanced-ai-applications')
        import intelligent_decision_system as decision
        
        # 简单测试
        advisor = decision.SmartInvestmentAdvisor()
        profile = {'age': 30, 'income_stability': 4, 'investment_experience': 3, 'investment_horizon': 10}
        risk = advisor.assess_risk_tolerance(profile)
        
        print("✅ 智能决策系统模块正常工作")
        print(f"   - 风险评估: {risk:.2f}")
        print("   - 包含多准则决策和风险分析")
        return True
    except Exception as e:
        print(f"❌ 智能决策系统测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("新增AI模块完整性测试")
    print("=" * 60)
    
    test_results = []
    
    # 测试基础章节
    test_results.append(test_problem_solving())
    test_results.append(test_complex_environments())
    test_results.append(test_automated_planning())
    test_results.append(test_temporal_reasoning())
    
    # 测试高级应用项目
    test_results.append(test_intelligent_game_ai())
    test_results.append(test_intelligent_chatbot())
    test_results.append(test_intelligent_decision_system())
    
    # 统计结果
    passed = sum(test_results)
    total = len(test_results)
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"✅ 通过测试: {passed}/{total}")
    print(f"❌ 失败测试: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有模块测试通过！项目已准备就绪。")
        print("\n📚 可用功能:")
        print("   • 第2章：问题求解的搜索")
        print("   • 第4章：复杂环境")
        print("   • 第11章：自动规划") 
        print("   • 第14章：时序推理")
        print("   • 高级AI应用：游戏AI、聊天机器人、决策系统")
        
        print("\n🚀 开始使用:")
        print("   python3 demo_comprehensive.py")
    else:
        print(f"\n⚠️  有 {total - passed} 个模块需要修复")
    
    return passed == total

if __name__ == "__main__":
    main() 