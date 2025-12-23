# BlurMaster - 经典模糊图像分析与处理系统

BlurMaster是一个基于经典数字图像处理（Digital Image Processing, DIP）算法的完整项目，涵盖了从图像的生成（模拟）、分析（评估）、处理（去模糊/锐化）到评价的全过程。

## 🌟 项目特点

- **无需显卡训练**：基于经典数字图像处理算法，运行速度快
- **原理清晰**：所有算法基于数学原理（傅里叶变换、卷积、滤波等）
- **友好的GUI界面**：Tkinter实现，操作简单直观
- **完整的功能套件**：15个功能模块，涵盖图像处理全流程
- **易于扩展**：模块化设计，便于添加新算法
- **适合学习**：代码结构清晰，注释详细，适合作为课程设计或个人项目

## 📋 功能模块清单

### 第一类：图像模拟与预处理 (4个)

1. **加载与灰度化**：支持读取本地图片，并提供一键转灰度功能
2. **高斯模糊模拟**：用户调节滑块，人为给清晰图片添加高斯模糊（模拟对焦不准）
3. **运动模糊模拟**：用户指定角度和长度，模拟相机抖动产生的线性模糊
4. **添加高斯噪声**：用于测试算法的鲁棒性，经典去模糊算法对噪声很敏感

### 第二类：图像质量分析 (4个)

5. **拉普拉斯方差评分**：经典的“清晰度评分”，值越高图像越清晰
6. **FFT频谱分析**：将图像转换到频域，可视化频谱图，模糊图像高频部分缺失
7. **直方图统计**：展示图像的灰度直方图，分析模糊对像素分布的影响
8. **边缘检测预览**：Canny边缘检测，直观展示当前图像保留的边缘信息

### 第三类：去模糊与增强 (5个)

9. **非锐化掩模**：通过减去模糊版本来增强边缘，视觉上让图片变清晰
10. **拉普拉斯锐化**：利用二阶微分算子直接增强图像细节
11. **CLAHE直方图均衡化**：增强对比度，使模糊图像中的细节更易识别
12. **维纳滤波去模糊**：经典的频域复原算法，需要已知PSF和信噪比
13. **Richardson-Lucy迭代复原**：迭代算法，对于泊松噪声环境下的模糊恢复效果较好

### 第四类：结果评估与对比 (2个)

14. **PSNR计算**：峰值信噪比，衡量与原图的相似度
15. **SSIM计算**：结构相似性，更符合人眼视觉感知的图像质量评价指标

## 🛠️ 技术栈

- **语言**：Python 3.9+
- **核心库**：
  - OpenCV (cv2)：图像处理核心库
  - NumPy：数值计算
  - Matplotlib：可视化
  - SciPy：科学计算
  - Scikit-image：图像处理算法
- **GUI框架**：Tkinter (Python自带，简单易用)

## 📦 安装指南

### 1. 克隆或下载项目

```bash
git clone <项目地址>
cd BlurMaster
```

### 2. 安装依赖

#### 方法1：使用pip安装（推荐）

```bash
# 升级pip
pip3 install --upgrade pip

# 安装依赖
pip3 install -r requirements.txt
```

#### 方法2：使用清华大学镜像源加速安装

```bash
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

#### 方法3：使用conda安装（如果有conda环境）

```bash
conda create -n blurmaster python=3.9
conda activate blurmaster
conda install -c conda-forge opencv numpy matplotlib scipy scikit-image
pip install pyqt5
```

### 3. 验证安装

```bash
python3 -c "
import cv2
import numpy as np
import matplotlib
import scipy
import skimage
print('✓ OpenCV版本:', cv2.__version__)
print('✓ NumPy版本:', np.__version__)
print('✓ Matplotlib版本:', matplotlib.__version__)
print('✓ SciPy版本:', scipy.__version__)
print('✓ Scikit-image版本:', skimage.__version__)
"
```

## 🚀 运行程序

### 1. 运行GUI程序

```bash
python3 main.py
```

### 2. 运行核心功能测试

```bash
python3 tests/test_core_functions.py
```

### 3. 运行语法检查

```bash
python3 tests/test_pure_syntax.py
```

## 🖥️ GUI界面操作指南

### 1. 菜单栏

- **文件**：
  - 打开图像：选择本地图片加载
  - 保存当前图像：保存处理后的图像
  - 退出：关闭程序
- **帮助**：
  - 关于：查看程序信息

### 2. 图像显示区域

- **左侧**：显示原始图像
- **右侧**：显示处理后的图像

### 3. 控制面板（标签页）

#### 标签页1：图像模拟与预处理

- **转换为灰度图**：一键将彩色图像转为灰度图
- **高斯模糊**：通过滑块调节核大小，实时添加高斯模糊
- **运动模糊**：调节角度和长度滑块，模拟相机抖动产生的模糊
- **高斯噪声**：调节标准差滑块，添加不同强度的高斯噪声
- **重置为原始图像**：恢复到加载时的原始图像

#### 标签页2：图像质量分析

- **计算拉普拉斯方差评分**：显示当前图像的清晰度评分
- **显示FFT频谱**：展示图像的频域频谱图
- **显示直方图**：绘制灰度直方图，分析像素分布
- **Canny边缘检测**：通过低阈值和高阈值滑块，实时显示边缘检测结果

#### 标签页3：去模糊与增强

- **非锐化掩模**：调节强度滑块，增强图像边缘
- **应用拉普拉斯锐化**：直接增强图像细节
- **应用CLAHE直方图均衡化**：增强对比度
- **应用维纳滤波去模糊**：经典频域去模糊算法
- **Richardson-Lucy**：调节迭代次数，应用迭代复原有算法

#### 标签页4：结果评估与对比

- **计算PSNR**：显示处理后图像与原始图像的峰值信噪比
- **计算SSIM**：显示结构相似性
- **显示原始图像**：在右侧面板显示原始图像
- **显示处理后图像**：在右侧面板显示当前处理结果

### 4. 信息显示区域

底部显示操作结果、图像信息和算法输出。

## 📁 项目结构

```
BlurMaster/
├── src/                    # 源代码目录
│   ├── __init__.py         # 包初始化文件
│   ├── image_processing.py # 核心图像处理功能
│   ├── gui.py              # GUI界面实现
│   └── utils.py            # 辅助工具函数
├── tests/                  # 测试文件目录
│   ├── test_core_functions.py   # 核心功能测试
│   ├── test_pure_syntax.py      # 纯语法检查
│   └── test_syntax.py           # 模块导入测试
├── assets/                 # 资源文件目录
├── main.py                 # 主程序入口
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明文档
```

## 📊 核心功能测试结果

运行核心功能测试后，您将看到类似以下的输出：

```
=== BlurMaster 核心功能测试 ===

测试1: 图像加载与灰度化
✓ 彩色图像加载成功，尺寸: (100, 100, 3)
✓ 灰度化成功，尺寸: (100, 100)

测试2: 模糊功能
✓ 高斯模糊功能正常
✓ 运动模糊功能正常
✓ 高斯噪声添加功能正常

测试3: 图像质量分析
✓ 拉普拉斯方差计算正常 - 原图: 0.00, 模糊图: 0.00
✓ FFT频谱分析正常，结果尺寸: (100, 100)
✓ 直方图计算正常，结果尺寸: (256, 1)
✓ Canny边缘检测正常，结果尺寸: (100, 100)

测试4: 图像增强功能
✓ 非锐化掩模功能正常
✓ 拉普拉斯锐化功能正常
✓ CLAHE直方图均衡化功能正常

测试5: 图像评估功能
✓ PSNR计算正常: inf dB
✓ SSIM计算正常: 1.0000

=== 所有核心功能测试通过！ ===
```

## 🔧 常见问题与解决方案

### 1. 问题：无法导入scikit-image

**解决方案**：尝试重新安装scikit-image，指定兼容版本

```bash
pip3 install --force-reinstall scikit-image==0.24.0
```

### 2. 问题：GUI程序无法运行，报错`no display name and no $DISPLAY environment variable`

**解决方案**：

- 使用VNC远程桌面访问
- 在有图形界面的环境中运行
- 只运行核心功能测试

### 3. 问题：安装依赖时下载速度慢

**解决方案**：使用国内镜像源

```bash
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 4. 问题：Python版本不兼容

**解决方案**：确保使用Python 3.9+版本

```bash
python3 --version
```

## 📚 学习资源

- [OpenCV官方文档](https://docs.opencv.org/)
- [Scikit-image官方文档](https://scikit-image.org/docs/stable/)
- [数字图像处理（第三版）](https://www.pearson.com/store/p/digital-image-processing/P100000194035/9780131687288)
- [Python数字图像处理](https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/)

## 🤝 贡献

欢迎贡献代码、报告bug或提出建议！

## 📄 许可证

本项目采用MIT许可证。

## 🙏 致谢

感谢所有为经典数字图像处理算法做出贡献的研究者和开发者！

## 📞 联系方式

如有任何问题，请随时联系项目作者。

---

**BluerMaster** - 让经典图像处理算法触手可及！
