#!/usr/bin/env python3
"""
测试BlurMaster的核心功能
"""

import sys
import os
import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.image_processing import (
    load_image, convert_to_grayscale, add_gaussian_blur,
    add_motion_blur, add_gaussian_noise, calculate_laplacian_variance,
    fft_analysis, calculate_histogram, detect_edges,
    unsharp_masking, laplacian_sharpening, apply_clahe,
    calculate_psnr, calculate_ssim
)

def test_load_image():
    """测试图像加载功能"""
    print("\n测试1: 图像加载与灰度化")
    
    # 创建一个简单的测试图像
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    cv2.imwrite("test_image.jpg", test_img)
    
    try:
        img = load_image("test_image.jpg")
        print(f"✓ 彩色图像加载成功，尺寸: {img.shape}")
        
        gray_img = convert_to_grayscale(img)
        print(f"✓ 灰度化成功，尺寸: {gray_img.shape}")
        
        return True, img, gray_img
    except Exception as e:
        print(f"✗ 图像加载失败: {e}")
        return False, None, None

def test_blur_functions(original_img):
    """测试模糊功能"""
    print("\n测试2: 模糊功能")
    
    try:
        # 测试高斯模糊
        gaussian_blur = add_gaussian_blur(original_img, kernel_size=5, sigma=1.0)
        print("✓ 高斯模糊功能正常")
        
        # 测试运动模糊
        motion_blur = add_motion_blur(original_img, angle=45, length=10)
        print("✓ 运动模糊功能正常")
        
        # 测试添加高斯噪声
        noisy_img = add_gaussian_noise(original_img, mean=0, sigma=10)
        print("✓ 高斯噪声添加功能正常")
        
        return True, gaussian_blur
    except Exception as e:
        print(f"✗ 模糊功能失败: {e}")
        return False, None

def test_quality_analysis(original_img, blurred_img):
    """测试图像质量分析功能"""
    print("\n测试3: 图像质量分析")
    
    try:
        # 测试拉普拉斯方差
        original_var = calculate_laplacian_variance(original_img)
        blurred_var = calculate_laplacian_variance(blurred_img)
        print(f"✓ 拉普拉斯方差计算正常 - 原图: {original_var:.2f}, 模糊图: {blurred_var:.2f}")
        
        # 测试FFT分析
        fft_result = fft_analysis(original_img)
        print(f"✓ FFT频谱分析正常，结果尺寸: {fft_result.shape}")
        
        # 测试直方图
        hist = calculate_histogram(original_img)
        print(f"✓ 直方图计算正常，结果尺寸: {hist.shape}")
        
        # 测试Canny边缘检测
        edges = detect_edges(original_img, low_threshold=50, high_threshold=150)
        print(f"✓ Canny边缘检测正常，结果尺寸: {edges.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 图像质量分析失败: {e}")
        return False

def test_enhancement_functions(blurred_img):
    """测试图像增强功能"""
    print("\n测试4: 图像增强功能")
    
    try:
        # 测试非锐化掩模
        unsharp = unsharp_masking(blurred_img, strength=1.5)
        print("✓ 非锐化掩模功能正常")
        
        # 测试拉普拉斯锐化
        laplacian_sharp = laplacian_sharpening(blurred_img)
        print("✓ 拉普拉斯锐化功能正常")
        
        # 测试CLAHE
        clahe_img = apply_clahe(blurred_img)
        print("✓ CLAHE直方图均衡化功能正常")
        
        return True, unsharp
    except Exception as e:
        print(f"✗ 图像增强功能失败: {e}")
        return False, None

def test_evaluation_functions(original_img, processed_img):
    """测试图像评估功能"""
    print("\n测试5: 图像评估功能")
    
    try:
        # 测试PSNR
        psnr = calculate_psnr(original_img, processed_img)
        print(f"✓ PSNR计算正常: {psnr:.2f} dB")
        
        # 测试SSIM
        ssim = calculate_ssim(original_img, processed_img)
        print(f"✓ SSIM计算正常: {ssim:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 图像评估功能失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== BlurMaster 核心功能测试 ===")
    
    # 测试1: 图像加载
    success, original_img, gray_img = test_load_image()
    if not success:
        return 1
    
    # 测试2: 模糊功能
    success, blurred_img = test_blur_functions(original_img)
    if not success:
        return 1
    
    # 测试3: 图像质量分析
    success = test_quality_analysis(original_img, blurred_img)
    if not success:
        return 1
    
    # 测试4: 图像增强功能
    success, enhanced_img = test_enhancement_functions(blurred_img)
    if not success:
        return 1
    
    # 测试5: 图像评估功能
    success = test_evaluation_functions(original_img, enhanced_img)
    if not success:
        return 1
    
    # 清理测试文件
    if os.path.exists("test_image.jpg"):
        os.remove("test_image.jpg")
    
    print("\n=== 所有核心功能测试通过！ ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
