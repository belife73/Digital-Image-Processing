import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal
from skimage import restoration, exposure, metrics
from skimage.filters import laplace

# 第一类：图像模拟与预处理

def load_image(image_path, grayscale=False):
    """加载图像，支持灰度化"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def convert_to_grayscale(img):
    """将彩色图像转换为灰度图"""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def add_gaussian_blur(img, kernel_size=5, sigma=1.0):
    """添加高斯模糊"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def add_motion_blur(img, angle=0, length=10):
    """添加运动模糊"""
    # 创建运动模糊核
    kernel = np.zeros((length, length))
    center = length // 2
    # 计算终点坐标
    angle_rad = np.radians(angle)
    x_end = int(center + length * np.cos(angle_rad))
    y_end = int(center + length * np.sin(angle_rad))
    # 绘制直线
    cv2.line(kernel, (center, center), (x_end, y_end), (1,), 1)
    # 归一化
    kernel = kernel / np.sum(kernel)
    # 应用卷积
    return cv2.filter2D(img, -1, kernel)

def add_gaussian_noise(img, mean=0, sigma=10):
    """添加高斯噪声"""
    if len(img.shape) == 3:
        noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
    else:
        noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

# 第二类：图像质量分析

def calculate_laplacian_variance(img):
    """计算拉普拉斯方差评分"""
    gray = convert_to_grayscale(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def fft_analysis(img):
    """FFT频谱分析"""
    gray = convert_to_grayscale(img)
    # 傅里叶变换
    f = fftpack.fft2(gray)
    fshift = fftpack.fftshift(f)
    # 计算幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum

def calculate_histogram(img):
    """计算直方图"""
    gray = convert_to_grayscale(img)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist

def detect_edges(img, low_threshold=50, high_threshold=150):
    """Canny边缘检测"""
    gray = convert_to_grayscale(img)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

# 第三类：去模糊与增强

def wiener_deconvolution(img, psf, snr=100):
    """维纳滤波去模糊"""
    gray = convert_to_grayscale(img)
    # 计算PSF的傅里叶变换
    psf_fft = fftpack.fft2(psf, s=gray.shape)
    # 计算图像的傅里叶变换
    img_fft = fftpack.fft2(gray)
    # 计算维纳滤波
    wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft)**2 + 1/snr)
    # 应用滤波
    result_fft = img_fft * wiener_filter
    # 逆傅里叶变换
    result = np.abs(fftpack.ifft2(result_fft))
    return result.astype(np.uint8)

def richardson_lucy_deconvolution(img, psf, iterations=30):
    """Richardson-Lucy迭代复原"""
    gray = convert_to_grayscale(img)
    result = restoration.richardson_lucy(gray, psf, iterations=iterations)
    # 归一化到0-255
    result = (result * 255).astype(np.uint8)
    return result

def unsharp_masking(img, sigma=1.0, strength=1.5):
    """非锐化掩模"""
    if len(img.shape) == 3:
        # 对彩色图像分通道处理
        channels = cv2.split(img)
        sharpened = []
        for channel in channels:
            blurred = cv2.GaussianBlur(channel, (0, 0), sigma)
            sharp = cv2.addWeighted(channel, 1 + strength, blurred, -strength, 0)
            sharpened.append(sharp)
        return cv2.merge(sharpened)
    else:
        # 灰度图像处理
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        return cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)

def laplacian_sharpening(img):
    """拉普拉斯锐化"""
    if len(img.shape) == 3:
        # 对彩色图像分通道处理
        channels = cv2.split(img)
        sharpened = []
        for channel in channels:
            lap = cv2.Laplacian(channel, cv2.CV_64F)
            sharp = channel - lap
            sharpened.append(np.clip(sharp, 0, 255).astype(np.uint8))
        return cv2.merge(sharpened)
    else:
        # 灰度图像处理
        lap = cv2.Laplacian(img, cv2.CV_64F)
        sharp = img - lap
        return np.clip(sharp, 0, 255).astype(np.uint8)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """应用CLAHE直方图均衡化"""
    if len(img.shape) == 3:
        # 对彩色图像转换到LAB空间处理
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # 灰度图像处理
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

# 第四类：结果评估与对比

def calculate_psnr(original, processed):
    """计算PSNR"""
    original_gray = convert_to_grayscale(original)
    processed_gray = convert_to_grayscale(processed)
    return metrics.peak_signal_noise_ratio(original_gray, processed_gray)

def calculate_ssim(original, processed):
    """计算SSIM"""
    original_gray = convert_to_grayscale(original)
    processed_gray = convert_to_grayscale(processed)
    return metrics.structural_similarity(original_gray, processed_gray)

# 辅助函数：生成PSF（点扩散函数）
def create_gaussian_psf(kernel_size=5, sigma=1.0):
    """创建高斯PSF"""
    x, y = np.mgrid[-kernel_size//2+1:kernel_size//2+1, -kernel_size//2+1:kernel_size//2+1]
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return psf / np.sum(psf)

def create_motion_psf(angle=0, length=10):
    """创建运动模糊PSF"""
    psf = np.zeros((length, length))
    center = length // 2
    angle_rad = np.radians(angle)
    x_end = int(center + length * np.cos(angle_rad))
    y_end = int(center + length * np.sin(angle_rad))
    cv2.line(psf, (center, center), (x_end, y_end), (1,), 1)
    return psf / np.sum(psf)
