#!/usr/bin/env python3
"""
人工智能综合平台启动器
整合所有功能模块的统一入口
"""

import sys
import os
import subprocess
import time
import threading
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

class AIplatformLauncher:
    """AI平台启动器"""
    
    def __init__(self):
        self.platform_name = "人工智能综合演示平台"
        self.version = "2.0.0"
        self.modules = {
            'web': '交互式Web界面',
            'playground': '算法游乐场',
            'dashboard': '性能仪表板',
            'tutorial': '教育教程',
            'demo': '综合演示',
            'test': '模块测试'
        }
        self.process_pool = []
    
    def print_banner(self):
        """打印启动横幅"""
        print("="*80)
        print(f"🤖 {self.platform_name} v{self.version}")
        print("="*80)
        print("基于《Artificial Intelligence: A Modern Approach》教材")
        print("包含完整的AI算法实现、交互式演示和教育资源")
        print("="*80)
        print()
    
    def print_menu(self):
        """打印主菜单"""
        print("📋 功能模块菜单:")
        print()
        print("1. 🌐 启动Web界面 - 浏览器中访问所有功能")
        print("2. 🎮 算法游乐场 - 交互式算法参数调整")
        print("3. 📊 性能仪表板 - 算法性能监控和比较")
        print("4. 🎓 教育教程 - 交互式AI算法学习")
        print("5. 🚀 综合演示 - 运行所有算法演示")
        print("6. 🔍 模块测试 - 验证所有模块功能")
        print("7. 🎯 快速体验 - 自动启动推荐功能")
        print("8. ⚙️  系统信息 - 查看系统状态")
        print("9. 📖 帮助文档 - 查看使用说明")
        print("0. 🚪 退出程序")
        print()
    
    def check_dependencies(self) -> bool:
        """检查依赖"""
        print("🔍 检查依赖...")
        
        required_packages = [
            'numpy', 'pandas', 'matplotlib', 'plotly', 'streamlit'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} (缺失)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
            print("请运行以下命令安装:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ 所有依赖检查通过")
        return True
    
    def launch_web_interface(self):
        """启动Web界面"""
        print("🌐 启动Web界面...")
        
        if not Path("web_interface.py").exists():
            print("❌ web_interface.py 文件不存在")
            return
        
        try:
            # 启动Streamlit应用
            cmd = [sys.executable, "start_web_interface.py"]
            process = subprocess.Popen(cmd)
            self.process_pool.append(process)
            
            print("✅ Web界面启动成功")
            print("📍 访问地址: http://localhost:8501")
            print("🌐 浏览器将自动打开")
            
        except Exception as e:
            print(f"❌ Web界面启动失败: {e}")
    
    def launch_algorithm_playground(self):
        """启动算法游乐场"""
        print("🎮 启动算法游乐场...")
        
        try:
            from algorithm_playground import demo_search_playground, demo_ml_playground
            
            print("选择游乐场类型:")
            print("1. 搜索算法游乐场")
            print("2. 机器学习游乐场")
            print("3. 全部演示")
            
            choice = input("请选择 (1-3): ").strip()
            
            if choice == '1':
                demo_search_playground()
            elif choice == '2':
                demo_ml_playground()
            elif choice == '3':
                demo_search_playground()
                demo_ml_playground()
            else:
                print("无效选择")
                
        except ImportError as e:
            print(f"❌ 算法游乐场模块加载失败: {e}")
        except Exception as e:
            print(f"❌ 算法游乐场启动失败: {e}")
    
    def launch_performance_dashboard(self):
        """启动性能仪表板"""
        print("📊 启动性能仪表板...")
        
        try:
            from performance_dashboard import demo_performance_dashboard
            
            print("开始性能基准测试...")
            demo_performance_dashboard()
            
        except ImportError as e:
            print(f"❌ 性能仪表板模块加载失败: {e}")
        except Exception as e:
            print(f"❌ 性能仪表板启动失败: {e}")
    
    def launch_educational_tutorials(self):
        """启动教育教程"""
        print("🎓 启动教育教程...")
        
        try:
            from educational_tutorials import demo_educational_platform
            
            print("开始交互式教学...")
            demo_educational_platform()
            
        except ImportError as e:
            print(f"❌ 教育教程模块加载失败: {e}")
        except Exception as e:
            print(f"❌ 教育教程启动失败: {e}")
    
    def launch_comprehensive_demo(self):
        """启动综合演示"""
        print("🚀 启动综合演示...")
        
        try:
            # 运行综合演示
            cmd = [sys.executable, "demo_comprehensive.py"]
            subprocess.run(cmd)
            
        except Exception as e:
            print(f"❌ 综合演示启动失败: {e}")
    
    def run_module_tests(self):
        """运行模块测试"""
        print("🔍 运行模块测试...")
        
        try:
            # 运行测试脚本
            cmd = [sys.executable, "test_new_modules.py"]
            subprocess.run(cmd)
            
        except Exception as e:
            print(f"❌ 模块测试失败: {e}")
    
    def quick_experience(self):
        """快速体验"""
        print("🎯 快速体验模式...")
        print("推荐体验流程：")
        print("1. 首先运行模块测试")
        print("2. 然后启动Web界面")
        print("3. 最后体验算法游乐场")
        print()
        
        if input("是否开始快速体验? (y/n): ").lower() == 'y':
            print("\n步骤1: 运行模块测试...")
            self.run_module_tests()
            
            print("\n步骤2: 启动Web界面...")
            self.launch_web_interface()
            
            print("\n步骤3: 等待5秒后启动算法游乐场...")
            time.sleep(5)
            self.launch_algorithm_playground()
    
    def show_system_info(self):
        """显示系统信息"""
        print("⚙️  系统信息:")
        print(f"  Python版本: {sys.version}")
        print(f"  平台: {sys.platform}")
        print(f"  工作目录: {os.getcwd()}")
        print(f"  可用模块: {len(self.modules)}")
        
        # 检查关键文件
        key_files = [
            "web_interface.py",
            "algorithm_playground.py", 
            "performance_dashboard.py",
            "educational_tutorials.py",
            "demo_comprehensive.py"
        ]
        
        print("\n  关键文件状态:")
        for file in key_files:
            status = "✅" if Path(file).exists() else "❌"
            print(f"    {status} {file}")
    
    def show_help(self):
        """显示帮助信息"""
        print("📖 帮助文档")
        print("="*50)
        print()
        print("🌐 Web界面:")
        print("  - 在浏览器中访问所有功能")
        print("  - 包含交互式算法演示")
        print("  - 支持实时参数调整")
        print()
        print("🎮 算法游乐场:")
        print("  - 交互式算法参数调整")
        print("  - 实时可视化结果")
        print("  - 支持多种算法比较")
        print()
        print("📊 性能仪表板:")
        print("  - 算法性能基准测试")
        print("  - 实时系统监控")
        print("  - 生成详细性能报告")
        print()
        print("🎓 教育教程:")
        print("  - 交互式AI算法学习")
        print("  - 包含理论讲解和代码示例")
        print("  - 支持在线测验")
        print()
        print("📁 项目结构:")
        print("  - 28个AI算法章节")
        print("  - 3个高级应用项目")
        print("  - 完整的测试和文档")
        print()
        print("🔗 相关链接:")
        print("  - 项目GitHub: https://github.com/your-repo")
        print("  - 教材官网: http://aima.cs.berkeley.edu/")
        print("  - 技术支持: your-email@example.com")
    
    def cleanup(self):
        """清理资源"""
        print("\n🧹 清理资源...")
        
        # 终止所有子进程
        for process in self.process_pool:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception:
                pass
        
        print("✅ 清理完成")
    
    def run(self):
        """运行主程序"""
        try:
            self.print_banner()
            
            # 检查依赖
            if not self.check_dependencies():
                return
            
            while True:
                self.print_menu()
                choice = input("请选择功能 (0-9): ").strip()
                
                if choice == '0':
                    print("👋 感谢使用人工智能综合平台！")
                    break
                elif choice == '1':
                    self.launch_web_interface()
                elif choice == '2':
                    self.launch_algorithm_playground()
                elif choice == '3':
                    self.launch_performance_dashboard()
                elif choice == '4':
                    self.launch_educational_tutorials()
                elif choice == '5':
                    self.launch_comprehensive_demo()
                elif choice == '6':
                    self.run_module_tests()
                elif choice == '7':
                    self.quick_experience()
                elif choice == '8':
                    self.show_system_info()
                elif choice == '9':
                    self.show_help()
                else:
                    print("❌ 无效选择，请重新输入")
                
                if choice != '0':
                    input("\n按Enter键返回主菜单...")
                    print("\n" + "="*80)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  程序被用户中断")
        except Exception as e:
            print(f"\n❌ 程序发生错误: {e}")
        finally:
            self.cleanup()

def main():
    """主函数"""
    launcher = AIplatformLauncher()
    launcher.run()

if __name__ == "__main__":
    main() 