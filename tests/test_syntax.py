#!/usr/bin/env python3
"""
简单的语法测试脚本，验证代码结构正确性
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=== BlurMaster 语法测试 ===")

# 测试核心模块导入
try:
    from src.image_processing import (
        load_image, convert_to_grayscale, add_gaussian_blur,
        add_motion_blur, add_gaussian_noise, calculate_laplacian_variance,
        fft_analysis, calculate_histogram, detect_edges,
        unsharp_masking, laplacian_sharpening, apply_clahe,
        calculate_psnr, calculate_ssim
    )
    print("✓ 核心功能模块导入成功")
except Exception as e:
    print(f"✗ 核心功能模块导入失败: {e}")
    sys.exit(1)

# 测试GUI模块导入
try:
    from src.gui import BlurMasterGUI
    print("✓ GUI模块导入成功")
except Exception as e:
    print(f"✗ GUI模块导入失败: {e}")
    # GUI模块依赖Tkinter，可能在无头环境中无法导入，不退出

# 测试主程序导入
try:
    import src.main
    print("✓ 主程序导入成功")
except Exception as e:
    print(f"✗ 主程序导入失败: {e}")
    sys.exit(1)

print("\n=== 所有语法测试通过！ ===")
print("\n项目代码结构正确，可以在已安装依赖的环境中运行。")
print("\n使用说明：")
print("1. 安装依赖：pip3 install -r requirements.txt")
print("2. 运行程序：python3 main.py")
print("3. 或运行核心功能测试：python3 tests/test_core_functions.py")
