import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .image_processing import *

class BlurMasterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BlurMaster - 经典模糊图像分析与处理系统")
        self.root.geometry("1200x800")
        
        # 图像数据
        self.original_img = None
        self.current_img = None
        self.processed_img = None
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建菜单栏
        self.create_menu()
        
        # 创建图像显示区域
        self.create_image_display()
        
        # 创建控制面板
        self.create_control_panel()
        
        # 创建信息显示区域
        self.create_info_panel()
        
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开图像", command=self.load_image)
        file_menu.add_command(label="保存当前图像", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_image_display(self):
        """创建图像显示区域"""
        display_frame = ttk.LabelFrame(self.main_frame, text="图像显示")
        display_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建两个子框架用于显示原始图像和处理后的图像
        left_frame = ttk.Frame(display_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = ttk.Frame(display_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 原始图像标签
        self.original_label = ttk.Label(left_frame, text="原始图像")
        self.original_label.pack(anchor=tk.W, pady=2)
        
        self.original_canvas = tk.Canvas(left_frame, bg="#f0f0f0", relief=tk.SUNKEN)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 处理后图像标签
        self.processed_label = ttk.Label(right_frame, text="处理后图像")
        self.processed_label.pack(anchor=tk.W, pady=2)
        
        self.processed_canvas = tk.Canvas(right_frame, bg="#f0f0f0", relief=tk.SUNKEN)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
    
    def create_control_panel(self):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(self.main_frame, text="控制面板")
        control_frame.pack(fill=tk.X, pady=5)
        
        # 创建笔记本（标签页）
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 第一类：图像模拟与预处理
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="1. 图像模拟与预处理")
        self.create_tab1(tab1)
        
        # 第二类：图像质量分析
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="2. 图像质量分析")
        self.create_tab2(tab2)
        
        # 第三类：去模糊与增强
        tab3 = ttk.Frame(notebook)
        notebook.add(tab3, text="3. 去模糊与增强")
        self.create_tab3(tab3)
        
        # 第四类：结果评估与对比
        tab4 = ttk.Frame(notebook)
        notebook.add(tab4, text="4. 结果评估")
        self.create_tab4(tab4)
    
    def create_tab1(self, parent):
        """图像模拟与预处理标签页"""
        # 灰度化按钮
        ttk.Button(parent, text="转换为灰度图", command=self.convert_to_grayscale_gui).pack(pady=5, padx=5, fill=tk.X)
        
        # 高斯模糊
        ttk.Label(parent, text="高斯模糊：").pack(anchor=tk.W, padx=5, pady=2)
        gaussian_frame = ttk.Frame(parent)
        gaussian_frame.pack(fill=tk.X, padx=10)
        
        ttk.Label(gaussian_frame, text="核大小：").pack(side=tk.LEFT)
        self.gaussian_kernel = tk.IntVar(value=5)
        ttk.Scale(gaussian_frame, from_=3, to=21, variable=self.gaussian_kernel, orient=tk.HORIZONTAL, 
                 command=lambda x: self.update_gaussian_blur()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(gaussian_frame, textvariable=self.gaussian_kernel).pack(side=tk.RIGHT, width=30)
        
        # 运动模糊
        ttk.Label(parent, text="运动模糊：").pack(anchor=tk.W, padx=5, pady=5)
        motion_frame = ttk.Frame(parent)
        motion_frame.pack(fill=tk.X, padx=10)
        
        ttk.Label(motion_frame, text="角度：").pack(side=tk.LEFT)
        self.motion_angle = tk.IntVar(value=0)
        ttk.Scale(motion_frame, from_=0, to=360, variable=self.motion_angle, orient=tk.HORIZONTAL, 
                 command=lambda x: self.update_motion_blur()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(motion_frame, textvariable=self.motion_angle).pack(side=tk.RIGHT, width=30)
        
        motion_length_frame = ttk.Frame(parent)
        motion_length_frame.pack(fill=tk.X, padx=10)
        ttk.Label(motion_length_frame, text="长度：").pack(side=tk.LEFT)
        self.motion_length = tk.IntVar(value=10)
        ttk.Scale(motion_length_frame, from_=1, to=50, variable=self.motion_length, orient=tk.HORIZONTAL, 
                 command=lambda x: self.update_motion_blur()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(motion_length_frame, textvariable=self.motion_length).pack(side=tk.RIGHT, width=30)
        
        # 高斯噪声
        ttk.Label(parent, text="高斯噪声：").pack(anchor=tk.W, padx=5, pady=5)
        noise_frame = ttk.Frame(parent)
        noise_frame.pack(fill=tk.X, padx=10)
        
        ttk.Label(noise_frame, text="标准差：").pack(side=tk.LEFT)
        self.noise_sigma = tk.IntVar(value=10)
        ttk.Scale(noise_frame, from_=1, to=50, variable=self.noise_sigma, orient=tk.HORIZONTAL, 
                 command=lambda x: self.update_gaussian_noise()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(noise_frame, textvariable=self.noise_sigma).pack(side=tk.RIGHT, width=30)
        
        # 重置按钮
        ttk.Button(parent, text="重置为原始图像", command=self.reset_image).pack(pady=10, padx=5, fill=tk.X)
    
    def create_tab2(self, parent):
        """图像质量分析标签页"""
        # 拉普拉斯方差评分
        ttk.Button(parent, text="计算拉普拉斯方差评分", command=self.calculate_laplacian).pack(pady=5, padx=5, fill=tk.X)
        
        # FFT频谱分析
        ttk.Button(parent, text="显示FFT频谱", command=self.show_fft).pack(pady=5, padx=5, fill=tk.X)
        
        # 直方图
        ttk.Button(parent, text="显示直方图", command=self.show_histogram).pack(pady=5, padx=5, fill=tk.X)
        
        # Canny边缘检测
        ttk.Label(parent, text="Canny边缘检测：").pack(anchor=tk.W, padx=5, pady=5)
        canny_frame = ttk.Frame(parent)
        canny_frame.pack(fill=tk.X, padx=10)
        
        ttk.Label(canny_frame, text="低阈值：").pack(side=tk.LEFT)
        self.canny_low = tk.IntVar(value=50)
        ttk.Scale(canny_frame, from_=1, to=200, variable=self.canny_low, orient=tk.HORIZONTAL, 
                 command=lambda x: self.update_canny()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(canny_frame, textvariable=self.canny_low).pack(side=tk.RIGHT, width=30)
        
        canny_high_frame = ttk.Frame(parent)
        canny_high_frame.pack(fill=tk.X, padx=10)
        ttk.Label(canny_high_frame, text="高阈值：").pack(side=tk.LEFT)
        self.canny_high = tk.IntVar(value=150)
        ttk.Scale(canny_high_frame, from_=1, to=300, variable=self.canny_high, orient=tk.HORIZONTAL, 
                 command=lambda x: self.update_canny()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(canny_high_frame, textvariable=self.canny_high).pack(side=tk.RIGHT, width=30)
    
    def create_tab3(self, parent):
        """去模糊与增强标签页"""
        # 非锐化掩模
        ttk.Label(parent, text="非锐化掩模：").pack(anchor=tk.W, padx=5, pady=5)
        unsharp_frame = ttk.Frame(parent)
        unsharp_frame.pack(fill=tk.X, padx=10)
        
        ttk.Label(unsharp_frame, text="强度：").pack(side=tk.LEFT)
        self.unsharp_strength = tk.DoubleVar(value=1.5)
        ttk.Scale(unsharp_frame, from_=0.1, to=5.0, variable=self.unsharp_strength, orient=tk.HORIZONTAL, 
                 command=lambda x: self.update_unsharp()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(unsharp_frame, textvariable=self.unsharp_strength, width=5).pack(side=tk.RIGHT, width=50)
        
        # 拉普拉斯锐化
        ttk.Button(parent, text="应用拉普拉斯锐化", command=self.apply_laplacian).pack(pady=5, padx=5, fill=tk.X)
        
        # CLAHE
        ttk.Button(parent, text="应用CLAHE直方图均衡化", command=self.apply_clahe).pack(pady=5, padx=5, fill=tk.X)
        
        # 维纳滤波
        ttk.Button(parent, text="应用维纳滤波去模糊", command=self.apply_wiener).pack(pady=5, padx=5, fill=tk.X)
        
        # Richardson-Lucy
        ttk.Label(parent, text="Richardson-Lucy迭代次数：").pack(anchor=tk.W, padx=5, pady=5)
        rl_frame = ttk.Frame(parent)
        rl_frame.pack(fill=tk.X, padx=10)
        
        ttk.Label(rl_frame, text="迭代次数：").pack(side=tk.LEFT)
        self.rl_iterations = tk.IntVar(value=30)
        ttk.Scale(rl_frame, from_=5, to=100, variable=self.rl_iterations, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(rl_frame, textvariable=self.rl_iterations).pack(side=tk.RIGHT, width=30)
        
        ttk.Button(parent, text="应用Richardson-Lucy", command=self.apply_richardson_lucy).pack(pady=5, padx=5, fill=tk.X)
    
    def create_tab4(self, parent):
        """结果评估与对比标签页"""
        # PSNR计算
        ttk.Button(parent, text="计算PSNR", command=self.calculate_psnr_gui).pack(pady=5, padx=5, fill=tk.X)
        
        # SSIM计算
        ttk.Button(parent, text="计算SSIM", command=self.calculate_ssim_gui).pack(pady=5, padx=5, fill=tk.X)
        
        # 显示原图按钮
        ttk.Button(parent, text="显示原始图像", command=self.show_original).pack(pady=5, padx=5, fill=tk.X)
        
        # 显示处理后图像按钮
        ttk.Button(parent, text="显示处理后图像", command=self.show_processed).pack(pady=5, padx=5, fill=tk.X)
    
    def create_info_panel(self):
        """创建信息显示区域"""
        info_frame = ttk.LabelFrame(self.main_frame, text="图像信息")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_text = tk.Text(info_frame, height=5, width=80)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.info_text.insert(tk.END, "图像信息将显示在这里...\n")
        self.info_text.config(state=tk.DISABLED)
    
    def load_image(self):
        """加载图像"""
        file_path = filedialog.askopenfilename(filetypes=[
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("所有文件", "*.*")
        ])
        if file_path:
            try:
                self.original_img = load_image(file_path)
                self.current_img = self.original_img.copy()
                self.processed_img = self.original_img.copy()
                self.display_image(self.original_canvas, self.original_img)
                self.display_image(self.processed_canvas, self.processed_img)
                self.update_info(f"已加载图像：{file_path}\n尺寸：{self.original_img.shape[1]}x{self.original_img.shape[0]}\n")
            except Exception as e:
                messagebox.showerror("错误", f"加载图像失败：{str(e)}")
    
    def save_image(self):
        """保存图像"""
        if self.processed_img is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[
                ("JPEG文件", "*.jpg"),
                ("PNG文件", "*.png"),
                ("所有文件", "*.*")
            ])
            if file_path:
                try:
                    cv2.imwrite(file_path, self.processed_img)
                    messagebox.showinfo("成功", f"图像已保存：{file_path}")
                except Exception as e:
                    messagebox.showerror("错误", f"保存图像失败：{str(e)}")
        else:
            messagebox.showwarning("警告", "没有图像可保存")
    
    def display_image(self, canvas, img):
        """在Canvas上显示图像"""
        # 清空Canvas
        canvas.delete("all")
        
        # 获取Canvas尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width == 1 or canvas_height == 1:
            # Canvas尚未初始化，等待下一次更新
            return
        
        # 调整图像尺寸以适应Canvas
        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        if canvas_width / canvas_height > aspect_ratio:
            # Canvas更宽，以高度为基准
            new_h = canvas_height - 20
            new_w = int(new_h * aspect_ratio)
        else:
            # Canvas更高，以宽度为基准
            new_w = canvas_width - 20
            new_h = int(new_w / aspect_ratio)
        
        # 调整图像大小
        resized_img = cv2.resize(img, (new_w, new_h))
        
        # 如果是彩色图像，转换为RGB
        if len(resized_img.shape) == 3:
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # 创建PhotoImage
        from PIL import Image, ImageTk
        image = Image.fromarray(resized_img)
        photo = ImageTk.PhotoImage(image=image)
        
        # 保存引用，防止被垃圾回收
        canvas.photo = photo
        
        # 计算居中位置
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        
        # 显示图像
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
    
    def update_info(self, text):
        """更新信息显示"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, text)
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)
    
    def convert_to_grayscale_gui(self):
        """转换为灰度图"""
        if self.current_img is not None:
            self.processed_img = convert_to_grayscale(self.current_img)
            self.display_image(self.processed_canvas, self.processed_img)
            self.update_info("已转换为灰度图\n")
    
    def update_gaussian_blur(self):
        """更新高斯模糊"""
        if self.current_img is not None:
            kernel_size = self.gaussian_kernel.get()
            # 确保核大小为奇数
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            self.processed_img = add_gaussian_blur(self.current_img, kernel_size=kernel_size)
            self.display_image(self.processed_canvas, self.processed_img)
    
    def update_motion_blur(self):
        """更新运动模糊"""
        if self.current_img is not None:
            angle = self.motion_angle.get()
            length = self.motion_length.get()
            self.processed_img = add_motion_blur(self.current_img, angle=angle, length=length)
            self.display_image(self.processed_canvas, self.processed_img)
    
    def update_gaussian_noise(self):
        """更新高斯噪声"""
        if self.current_img is not None:
            sigma = self.noise_sigma.get()
            self.processed_img = add_gaussian_noise(self.current_img, sigma=sigma)
            self.display_image(self.processed_canvas, self.processed_img)
    
    def update_canny(self):
        """更新Canny边缘检测"""
        if self.current_img is not None:
            low = self.canny_low.get()
            high = self.canny_high.get()
            edges = detect_edges(self.current_img, low_threshold=low, high_threshold=high)
            self.display_image(self.processed_canvas, edges)
    
    def update_unsharp(self):
        """更新非锐化掩模"""
        if self.current_img is not None:
            strength = self.unsharp_strength.get()
            self.processed_img = unsharp_masking(self.current_img, strength=strength)
            self.display_image(self.processed_canvas, self.processed_img)
    
    def calculate_laplacian(self):
        """计算拉普拉斯方差评分"""
        if self.current_img is not None:
            score = calculate_laplacian_variance(self.current_img)
            self.update_info(f"拉普拉斯方差评分：{score:.2f} (值越高越清晰)\n")
    
    def show_fft(self):
        """显示FFT频谱"""
        if self.current_img is not None:
            fft_result = fft_analysis(self.current_img)
            # 将FFT结果转换为8位图像以便显示
            fft_normalized = cv2.normalize(fft_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.display_image(self.processed_canvas, fft_normalized)
            self.update_info("显示FFT频谱图\n")
    
    def show_histogram(self):
        """显示直方图"""
        if self.current_img is not None:
            hist = calculate_histogram(self.current_img)
            
            # 创建直方图图像
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(hist)
            ax.set_title("灰度直方图")
            ax.set_xlabel("灰度值")
            ax.set_ylabel("像素数")
            ax.set_xlim([0, 256])
            
            # 转换为图像以便显示
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            self.display_image(self.processed_canvas, buf)
            self.update_info("显示灰度直方图\n")
    
    def apply_laplacian(self):
        """应用拉普拉斯锐化"""
        if self.current_img is not None:
            self.processed_img = laplacian_sharpening(self.current_img)
            self.display_image(self.processed_canvas, self.processed_img)
            self.update_info("已应用拉普拉斯锐化\n")
    
    def apply_clahe(self):
        """应用CLAHE直方图均衡化"""
        if self.current_img is not None:
            self.processed_img = apply_clahe(self.current_img)
            self.display_image(self.processed_canvas, self.processed_img)
            self.update_info("已应用CLAHE直方图均衡化\n")
    
    def apply_wiener(self):
        """应用维纳滤波去模糊"""
        if self.current_img is not None:
            try:
                # 使用高斯PSF
                psf = create_gaussian_psf(kernel_size=15, sigma=3.0)
                self.processed_img = wiener_deconvolution(self.current_img, psf)
                self.display_image(self.processed_canvas, self.processed_img)
                self.update_info("已应用维纳滤波去模糊\n")
            except Exception as e:
                messagebox.showerror("错误", f"维纳滤波失败：{str(e)}")
    
    def apply_richardson_lucy(self):
        """应用Richardson-Lucy"""
        if self.current_img is not None:
            try:
                # 使用高斯PSF
                psf = create_gaussian_psf(kernel_size=15, sigma=3.0)
                iterations = self.rl_iterations.get()
                self.processed_img = richardson_lucy_deconvolution(self.current_img, psf, iterations=iterations)
                self.display_image(self.processed_canvas, self.processed_img)
                self.update_info(f"已应用Richardson-Lucy，迭代次数：{iterations}\n")
            except Exception as e:
                messagebox.showerror("错误", f"Richardson-Lucy失败：{str(e)}")
    
    def calculate_psnr_gui(self):
        """计算PSNR"""
        if self.original_img is not None and self.processed_img is not None:
            psnr = calculate_psnr(self.original_img, self.processed_img)
            self.update_info(f"PSNR：{psnr:.2f} dB (值越高越相似)\n")
    
    def calculate_ssim_gui(self):
        """计算SSIM"""
        if self.original_img is not None and self.processed_img is not None:
            ssim = calculate_ssim(self.original_img, self.processed_img)
            self.update_info(f"SSIM：{ssim:.4f} (值越高越相似)\n")
    
    def show_original(self):
        """显示原始图像"""
        if self.original_img is not None:
            self.display_image(self.processed_canvas, self.original_img)
    
    def show_processed(self):
        """显示处理后图像"""
        if self.processed_img is not None:
            self.display_image(self.processed_canvas, self.processed_img)
    
    def reset_image(self):
        """重置为原始图像"""
        if self.original_img is not None:
            self.processed_img = self.original_img.copy()
            self.display_image(self.processed_canvas, self.processed_img)
            self.update_info("已重置为原始图像\n")
    
    def update_canny(self):
        """更新Canny边缘检测"""
        if self.current_img is not None:
            low = self.canny_low.get()
            high = self.canny_high.get()
            edges = detect_edges(self.current_img, low_threshold=low, high_threshold=high)
            self.display_image(self.processed_canvas, edges)
    
    def show_about(self):
        """显示关于对话框"""
        about_text = "BlurMaster - 经典模糊图像分析与处理系统\n\n"
        about_text += "技术栈：Python 3.9+ + OpenCV + Scikit-image\n\n"
        about_text += "功能：\n"
        about_text += "1. 图像模拟与预处理\n"
        about_text += "2. 图像质量分析\n"
        about_text += "3. 去模糊与增强\n"
        about_text += "4. 结果评估与对比\n\n"
        about_text += "版本：1.0.0\n"
        about_text += "作者：BlurMaster团队"
        
        messagebox.showinfo("关于", about_text)
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()
