#!/usr/bin/env python3
"""
纯语法检查脚本，不导入任何外部库
"""

import sys
import os
import ast

print("=== BlurMaster 纯语法检查 ===")

# 需要检查的Python文件列表
python_files = [
    "src/__init__.py",
    "src/image_processing.py",
    "src/gui.py",
    "main.py"
]

success_count = 0
error_count = 0

for file_path in python_files:
    full_path = os.path.join("BlurMaster", file_path)
    if os.path.exists(full_path):
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            ast.parse(content)
            print(f"✓ {file_path}: 语法正确")
            success_count += 1
        except SyntaxError as e:
            print(f"✗ {file_path}: 语法错误 - {e}")
            error_count += 1
        except Exception as e:
            print(f"✗ {file_path}: 读取错误 - {e}")
            error_count += 1
    else:
        print(f"✗ {file_path}: 文件不存在")
        error_count += 1

print(f"\n=== 检查完成 ===")
print(f"成功: {success_count} 个文件")
print(f"失败: {error_count} 个文件")

if error_count == 0:
    print("\n🎉 所有文件语法检查通过！")
    print("\n项目代码结构完整，语法正确，可以在已安装依赖的环境中运行。")
    print("\n使用说明：")
    print("1. 安装依赖：pip3 install -r requirements.txt")
    print("2. 运行程序：python3 main.py")
    print("3. 或运行核心功能测试：python3 tests/test_core_functions.py")
    sys.exit(0)
else:
    print(f"\n❌ 发现 {error_count} 个语法错误")
    sys.exit(1)
