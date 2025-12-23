#!/usr/bin/env python3
"""
BlurMaster - 经典模糊图像分析与处理系统
主程序入口
"""

import sys
import tkinter as tk
from src.gui import BlurMasterGUI

def main():
    """主函数"""
    try:
        # 创建Tkinter根窗口
        root = tk.Tk()
        
        # 创建并运行GUI应用
        app = BlurMasterGUI(root)
        app.run()
        
    except KeyboardInterrupt:
        print("\n程序已退出")
        sys.exit(0)
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
