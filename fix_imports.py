#!/usr/bin/env python
"""
修复脚本：为所有Python脚本添加正确的导入路径
"""
import os
import sys

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 添加到Python路径
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"✓ 项目路径已添加: {PROJECT_ROOT}")
print(f"✓ sys.path: {sys.path[:3]}")  # 显示前3个路径
