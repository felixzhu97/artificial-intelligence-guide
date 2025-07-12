#!/usr/bin/env python3
"""
人工智能综合演示系统 - Web界面启动脚本
"""

import sys
import subprocess
import os
import webbrowser
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误：需要Python 3.8或更高版本")
        print(f"当前版本：{sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python版本检查通过：{sys.version}")

def install_dependencies():
    """安装依赖包"""
    print("📦 检查并安装依赖包...")
    
    # 基础依赖
    basic_deps = [
        "streamlit>=1.28.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "plotly>=5.0.0",
        "seaborn>=0.11.0"
    ]
    
    for dep in basic_deps:
        try:
            print(f"  正在安装 {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  ✅ {dep} 安装成功")
        except subprocess.CalledProcessError:
            print(f"  ❌ {dep} 安装失败")
            return False
    
    return True

def check_streamlit():
    """检查Streamlit是否可用"""
    try:
        import streamlit
        print(f"✅ Streamlit检查通过：版本 {streamlit.__version__}")
        return True
    except ImportError:
        print("❌ Streamlit未安装")
        return False

def start_web_interface():
    """启动Web界面"""
    print("🚀 启动Web界面...")
    
    # 检查web_interface.py是否存在
    if not Path("web_interface.py").exists():
        print("❌ 错误：web_interface.py文件不存在")
        sys.exit(1)
    
    # 设置Streamlit配置
    config_options = [
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--theme.base=light"
    ]
    
    try:
        # 构建命令
        cmd = [sys.executable, "-m", "streamlit", "run", "web_interface.py"] + config_options
        
        print("🌐 Web界面启动中...")
        print("📍 访问地址：http://localhost:8501")
        print("⏹️  按Ctrl+C停止服务")
        print("="*50)
        
        # 自动打开浏览器
        webbrowser.open("http://localhost:8501")
        
        # 启动Streamlit应用
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Web界面已停止")
    except Exception as e:
        print(f"❌ 启动失败：{e}")
        sys.exit(1)

def main():
    """主函数"""
    print("🤖 人工智能综合演示系统 - Web界面启动器")
    print("="*50)
    
    # 检查Python版本
    check_python_version()
    
    # 检查Streamlit
    if not check_streamlit():
        print("正在安装依赖...")
        if not install_dependencies():
            print("❌ 依赖安装失败，请手动安装：")
            print("pip install streamlit numpy pandas matplotlib plotly seaborn")
            sys.exit(1)
    
    # 启动Web界面
    start_web_interface()

if __name__ == "__main__":
    main() 