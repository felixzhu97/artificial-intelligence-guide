"""
计算机视觉实现

包含图像处理、特征提取、目标检测、图像分类等
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import math


class Image:
    """图像类"""
    
    def __init__(self, data: np.ndarray, channels: int = 1):
        self.data = data
        self.height, self.width = data.shape[:2]
        self.channels = channels
    
    def to_grayscale(self) -> 'Image':
        """转换为灰度图"""
        if len(self.data.shape) == 3:
            # RGB到灰度的转换
            gray = np.dot(self.data, [0.299, 0.587, 0.114])
            return Image(gray, 1)
        return self
    
    def resize(self, new_height: int, new_width: int) -> 'Image':
        """调整图像大小"""
        # 简单的最近邻插值
        h_ratio = self.height / new_height
        w_ratio = self.width / new_width
        
        new_data = np.zeros((new_height, new_width))
        
        for i in range(new_height):
            for j in range(new_width):
                old_i = int(i * h_ratio)
                old_j = int(j * w_ratio)
                old_i = min(old_i, self.height - 1)
                old_j = min(old_j, self.width - 1)
                new_data[i, j] = self.data[old_i, old_j]
        
        return Image(new_data, self.channels)
    
    def normalize(self) -> 'Image':
        """归一化到0-1"""
        normalized = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return Image(normalized, self.channels)


class ImageFilter:
    """图像滤波器"""
    
    @staticmethod
    def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """生成高斯核"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        
        return kernel / np.sum(kernel)
    
    @staticmethod
    def sobel_x_kernel() -> np.ndarray:
        """Sobel X方向核"""
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    @staticmethod
    def sobel_y_kernel() -> np.ndarray:
        """Sobel Y方向核"""
        return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    @staticmethod
    def apply_filter(image: Image, kernel: np.ndarray) -> Image:
        """应用滤波器"""
        filtered = ImageFilter.convolve(image.data, kernel)
        return Image(filtered, image.channels)
    
    @staticmethod
    def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """卷积操作"""
        h, w = image.shape
        kh, kw = kernel.shape
        
        # 填充
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        # 卷积
        result = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i + kh, j:j + kw] * kernel)
        
        return result


class EdgeDetector:
    """边缘检测器"""
    
    @staticmethod
    def sobel_edge_detection(image: Image) -> Tuple[Image, Image, Image]:
        """Sobel边缘检测"""
        # 转换为灰度图
        gray = image.to_grayscale()
        
        # 应用Sobel核
        sobel_x = ImageFilter.apply_filter(gray, ImageFilter.sobel_x_kernel())
        sobel_y = ImageFilter.apply_filter(gray, ImageFilter.sobel_y_kernel())
        
        # 计算梯度幅度
        magnitude = np.sqrt(sobel_x.data ** 2 + sobel_y.data ** 2)
        magnitude_image = Image(magnitude, 1)
        
        return sobel_x, sobel_y, magnitude_image
    
    @staticmethod
    def canny_edge_detection(image: Image, low_threshold: float = 0.1, 
                            high_threshold: float = 0.2) -> Image:
        """Canny边缘检测"""
        # 1. 高斯滤波
        gaussian_kernel = ImageFilter.gaussian_kernel(5, 1.4)
        smoothed = ImageFilter.apply_filter(image.to_grayscale(), gaussian_kernel)
        
        # 2. 计算梯度
        sobel_x, sobel_y, magnitude = EdgeDetector.sobel_edge_detection(smoothed)
        
        # 3. 非最大值抑制
        suppressed = EdgeDetector.non_maximum_suppression(magnitude, sobel_x, sobel_y)
        
        # 4. 双阈值处理
        edges = EdgeDetector.double_threshold(suppressed, low_threshold, high_threshold)
        
        return edges
    
    @staticmethod
    def non_maximum_suppression(magnitude: Image, grad_x: Image, grad_y: Image) -> Image:
        """非最大值抑制"""
        h, w = magnitude.height, magnitude.width
        suppressed = np.zeros((h, w))
        
        # 计算梯度方向
        angle = np.arctan2(grad_y.data, grad_x.data)
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # 量化角度
                angle_deg = np.degrees(angle[i, j]) % 180
                
                if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg < 180):
                    # 水平方向
                    if (magnitude.data[i, j] >= magnitude.data[i, j - 1] and
                        magnitude.data[i, j] >= magnitude.data[i, j + 1]):
                        suppressed[i, j] = magnitude.data[i, j]
                elif 22.5 <= angle_deg < 67.5:
                    # 对角线方向
                    if (magnitude.data[i, j] >= magnitude.data[i - 1, j + 1] and
                        magnitude.data[i, j] >= magnitude.data[i + 1, j - 1]):
                        suppressed[i, j] = magnitude.data[i, j]
                elif 67.5 <= angle_deg < 112.5:
                    # 垂直方向
                    if (magnitude.data[i, j] >= magnitude.data[i - 1, j] and
                        magnitude.data[i, j] >= magnitude.data[i + 1, j]):
                        suppressed[i, j] = magnitude.data[i, j]
                elif 112.5 <= angle_deg < 157.5:
                    # 反对角线方向
                    if (magnitude.data[i, j] >= magnitude.data[i - 1, j - 1] and
                        magnitude.data[i, j] >= magnitude.data[i + 1, j + 1]):
                        suppressed[i, j] = magnitude.data[i, j]
        
        return Image(suppressed, 1)
    
    @staticmethod
    def double_threshold(image: Image, low_threshold: float, high_threshold: float) -> Image:
        """双阈值处理"""
        h, w = image.height, image.width
        edges = np.zeros((h, w))
        
        # 归一化
        normalized = image.normalize()
        
        # 强边缘
        strong_edges = normalized.data > high_threshold
        # 弱边缘
        weak_edges = (normalized.data >= low_threshold) & (normalized.data <= high_threshold)
        
        edges[strong_edges] = 1.0
        edges[weak_edges] = 0.5
        
        # 连接弱边缘
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if edges[i, j] == 0.5:
                    # 检查周围是否有强边缘
                    if np.any(edges[i-1:i+2, j-1:j+2] == 1.0):
                        edges[i, j] = 1.0
                    else:
                        edges[i, j] = 0.0
        
        return Image(edges, 1)


class FeatureExtractor:
    """特征提取器"""
    
    @staticmethod
    def extract_hog_features(image: Image, cell_size: int = 8, 
                           block_size: int = 2, nbins: int = 9) -> np.ndarray:
        """提取HOG特征"""
        # 转换为灰度图
        gray = image.to_grayscale()
        
        # 计算梯度
        sobel_x, sobel_y, magnitude = EdgeDetector.sobel_edge_detection(gray)
        
        # 计算梯度方向
        angle = np.arctan2(sobel_y.data, sobel_x.data)
        angle = np.degrees(angle) % 180
        
        # 将图像分成细胞
        h, w = gray.height, gray.width
        cells_y = h // cell_size
        cells_x = w // cell_size
        
        # 计算每个细胞的直方图
        cell_histograms = []
        for i in range(cells_y):
            for j in range(cells_x):
                # 提取细胞
                cell_magnitude = magnitude.data[i*cell_size:(i+1)*cell_size, 
                                               j*cell_size:(j+1)*cell_size]
                cell_angle = angle[i*cell_size:(i+1)*cell_size, 
                                  j*cell_size:(j+1)*cell_size]
                
                # 计算直方图
                hist = np.zeros(nbins)
                for y in range(cell_size):
                    for x in range(cell_size):
                        angle_val = cell_angle[y, x]
                        magnitude_val = cell_magnitude[y, x]
                        
                        # 分配到相应的bin
                        bin_idx = int(angle_val / (180 / nbins))
                        bin_idx = min(bin_idx, nbins - 1)
                        hist[bin_idx] += magnitude_val
                
                cell_histograms.append(hist)
        
        # 块归一化
        block_histograms = []
        for i in range(cells_y - block_size + 1):
            for j in range(cells_x - block_size + 1):
                block_hist = []
                for bi in range(block_size):
                    for bj in range(block_size):
                        cell_idx = (i + bi) * cells_x + (j + bj)
                        block_hist.extend(cell_histograms[cell_idx])
                
                # L2归一化
                block_hist = np.array(block_hist)
                norm = np.linalg.norm(block_hist)
                if norm > 0:
                    block_hist = block_hist / norm
                
                block_histograms.extend(block_hist)
        
        return np.array(block_histograms)
    
    @staticmethod
    def extract_lbp_features(image: Image, radius: int = 3, n_points: int = 24) -> np.ndarray:
        """提取LBP（局部二值模式）特征"""
        gray = image.to_grayscale()
        h, w = gray.height, gray.width
        
        lbp = np.zeros((h, w))
        
        # 计算每个像素的LBP值
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray.data[i, j]
                lbp_value = 0
                
                # 采样周围的点
                for p in range(n_points):
                    # 计算采样点的坐标
                    angle = 2 * np.pi * p / n_points
                    x = j + radius * np.cos(angle)
                    y = i - radius * np.sin(angle)
                    
                    # 双线性插值
                    x1, y1 = int(np.floor(x)), int(np.floor(y))
                    x2, y2 = x1 + 1, y1 + 1
                    
                    if 0 <= x1 < w-1 and 0 <= y1 < h-1:
                        # 双线性插值
                        wa = (x2 - x) * (y2 - y)
                        wb = (x - x1) * (y2 - y)
                        wc = (x2 - x) * (y - y1)
                        wd = (x - x1) * (y - y1)
                        
                        interpolated = (wa * gray.data[y1, x1] + wb * gray.data[y1, x2] +
                                       wc * gray.data[y2, x1] + wd * gray.data[y2, x2])
                        
                        # 比较并设置位
                        if interpolated >= center:
                            lbp_value |= (1 << p)
                
                lbp[i, j] = lbp_value
        
        # 计算LBP直方图
        hist, _ = np.histogram(lbp.flatten(), bins=2**n_points, range=(0, 2**n_points))
        
        return hist / np.sum(hist)


class ObjectDetector:
    """目标检测器"""
    
    def __init__(self):
        self.templates = {}
        self.classifiers = {}
    
    def add_template(self, name: str, template: Image):
        """添加模板"""
        self.templates[name] = template
    
    def template_matching(self, image: Image, template_name: str, 
                         threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """模板匹配"""
        if template_name not in self.templates:
            return []
        
        template = self.templates[template_name]
        
        # 转换为灰度图
        gray_image = image.to_grayscale()
        gray_template = template.to_grayscale()
        
        h, w = gray_image.height, gray_image.width
        th, tw = gray_template.height, gray_template.width
        
        matches = []
        
        # 滑动窗口
        for i in range(h - th + 1):
            for j in range(w - tw + 1):
                # 提取窗口
                window = gray_image.data[i:i + th, j:j + tw]
                
                # 计算相关系数
                correlation = self.normalized_cross_correlation(window, gray_template.data)
                
                if correlation > threshold:
                    matches.append((j, i, correlation))
        
        return matches
    
    def normalized_cross_correlation(self, window: np.ndarray, template: np.ndarray) -> float:
        """归一化互相关"""
        # 零均值化
        window_mean = np.mean(window)
        template_mean = np.mean(template)
        
        window_centered = window - window_mean
        template_centered = template - template_mean
        
        # 计算相关系数
        numerator = np.sum(window_centered * template_centered)
        denominator = np.sqrt(np.sum(window_centered ** 2) * np.sum(template_centered ** 2))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def sliding_window_detection(self, image: Image, window_size: Tuple[int, int], 
                                step_size: int = 4) -> List[Tuple[int, int, np.ndarray]]:
        """滑动窗口检测"""
        h, w = image.height, image.width
        window_h, window_w = window_size
        
        windows = []
        
        for i in range(0, h - window_h + 1, step_size):
            for j in range(0, w - window_w + 1, step_size):
                window = image.data[i:i + window_h, j:j + window_w]
                windows.append((j, i, window))
        
        return windows


class ImageClassifier:
    """图像分类器"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.classifier = None
    
    def train(self, images: List[Image], labels: List[str], feature_type: str = 'hog'):
        """训练分类器"""
        # 提取特征
        features = []
        for image in images:
            if feature_type == 'hog':
                feature = self.feature_extractor.extract_hog_features(image)
            elif feature_type == 'lbp':
                feature = self.feature_extractor.extract_lbp_features(image)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            features.append(feature)
        
        # 训练简单的最近邻分类器
        self.classifier = {
            'features': features,
            'labels': labels,
            'feature_type': feature_type
        }
    
    def predict(self, image: Image) -> str:
        """预测图像类别"""
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        
        # 提取特征
        if self.classifier['feature_type'] == 'hog':
            feature = self.feature_extractor.extract_hog_features(image)
        elif self.classifier['feature_type'] == 'lbp':
            feature = self.feature_extractor.extract_lbp_features(image)
        else:
            raise ValueError(f"Unknown feature type: {self.classifier['feature_type']}")
        
        # 最近邻分类
        min_distance = float('inf')
        best_label = None
        
        for train_feature, label in zip(self.classifier['features'], self.classifier['labels']):
            distance = np.linalg.norm(feature - train_feature)
            if distance < min_distance:
                min_distance = distance
                best_label = label
        
        return best_label


class ImageSegmentation:
    """图像分割"""
    
    @staticmethod
    def k_means_segmentation(image: Image, k: int = 3, max_iter: int = 100) -> Image:
        """K-means图像分割"""
        # 转换为特征向量
        h, w = image.height, image.width
        if len(image.data.shape) == 3:
            pixels = image.data.reshape((-1, 3))
        else:
            pixels = image.data.reshape((-1, 1))
        
        # 初始化聚类中心
        centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]
        
        for _ in range(max_iter):
            # 分配像素到最近的聚类中心
            distances = np.sqrt(((pixels - centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # 更新聚类中心
            new_centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])
            
            # 检查收敛
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        # 创建分割图像
        segmented = labels.reshape((h, w))
        
        return Image(segmented, 1)
    
    @staticmethod
    def threshold_segmentation(image: Image, threshold: float = 0.5) -> Image:
        """阈值分割"""
        gray = image.to_grayscale()
        normalized = gray.normalize()
        
        binary = (normalized.data > threshold).astype(float)
        
        return Image(binary, 1)


def demo_edge_detection():
    """演示边缘检测"""
    print("边缘检测演示")
    print("=" * 30)
    
    # 创建一个简单的测试图像
    test_image = np.zeros((100, 100))
    test_image[25:75, 25:75] = 1.0  # 正方形
    test_image[40:60, 40:60] = 0.5  # 内部正方形
    
    image = Image(test_image, 1)
    
    # Sobel边缘检测
    sobel_x, sobel_y, magnitude = EdgeDetector.sobel_edge_detection(image)
    print(f"Sobel边缘检测完成，梯度幅度范围: {np.min(magnitude.data):.3f} - {np.max(magnitude.data):.3f}")
    
    # Canny边缘检测
    canny_edges = EdgeDetector.canny_edge_detection(image)
    print(f"Canny边缘检测完成，边缘像素数: {np.sum(canny_edges.data > 0)}")


def demo_feature_extraction():
    """演示特征提取"""
    print("特征提取演示")
    print("=" * 30)
    
    # 创建测试图像
    test_image = np.random.rand(64, 64)
    image = Image(test_image, 1)
    
    # HOG特征
    hog_features = FeatureExtractor.extract_hog_features(image)
    print(f"HOG特征维度: {len(hog_features)}")
    print(f"HOG特征范围: {np.min(hog_features):.3f} - {np.max(hog_features):.3f}")
    
    # LBP特征
    lbp_features = FeatureExtractor.extract_lbp_features(image)
    print(f"LBP特征维度: {len(lbp_features)}")
    print(f"LBP特征范围: {np.min(lbp_features):.3f} - {np.max(lbp_features):.3f}")


def demo_object_detection():
    """演示目标检测"""
    print("目标检测演示")
    print("=" * 30)
    
    # 创建测试图像和模板
    test_image = np.random.rand(100, 100)
    template = np.random.rand(20, 20)
    
    # 在图像中放置模板
    test_image[30:50, 30:50] = template
    
    image = Image(test_image, 1)
    template_image = Image(template, 1)
    
    # 目标检测
    detector = ObjectDetector()
    detector.add_template("test_object", template_image)
    
    matches = detector.template_matching(image, "test_object", threshold=0.8)
    print(f"找到 {len(matches)} 个匹配")
    
    for i, (x, y, score) in enumerate(matches):
        print(f"匹配 {i+1}: 位置 ({x}, {y}), 得分 {score:.3f}")


def demo_image_classification():
    """演示图像分类"""
    print("图像分类演示")
    print("=" * 30)
    
    # 创建训练数据
    train_images = []
    train_labels = []
    
    # 类别1：左上角有亮点
    for _ in range(5):
        img = np.random.rand(32, 32) * 0.3
        img[5:15, 5:15] = np.random.rand(10, 10) * 0.7 + 0.3
        train_images.append(Image(img, 1))
        train_labels.append("bright_topleft")
    
    # 类别2：右下角有亮点
    for _ in range(5):
        img = np.random.rand(32, 32) * 0.3
        img[17:27, 17:27] = np.random.rand(10, 10) * 0.7 + 0.3
        train_images.append(Image(img, 1))
        train_labels.append("bright_bottomright")
    
    # 训练分类器
    classifier = ImageClassifier()
    classifier.train(train_images, train_labels, feature_type='hog')
    
    # 测试
    test_img = np.random.rand(32, 32) * 0.3
    test_img[5:15, 5:15] = np.random.rand(10, 10) * 0.7 + 0.3
    test_image = Image(test_img, 1)
    
    prediction = classifier.predict(test_image)
    print(f"预测结果: {prediction}")


def demo_image_segmentation():
    """演示图像分割"""
    print("图像分割演示")
    print("=" * 30)
    
    # 创建测试图像
    test_image = np.zeros((100, 100))
    test_image[20:40, 20:40] = 0.3  # 暗色区域
    test_image[60:80, 60:80] = 0.7  # 亮色区域
    test_image[40:60, 40:60] = 0.5  # 中等亮度区域
    
    image = Image(test_image, 1)
    
    # K-means分割
    segmented = ImageSegmentation.k_means_segmentation(image, k=3)
    print(f"K-means分割完成，分割区域数: {len(np.unique(segmented.data))}")
    
    # 阈值分割
    binary = ImageSegmentation.threshold_segmentation(image, threshold=0.4)
    print(f"阈值分割完成，前景像素数: {np.sum(binary.data > 0)}")


if __name__ == "__main__":
    # 演示不同的计算机视觉技术
    demo_edge_detection()
    print("\n" + "="*50)
    demo_feature_extraction()
    print("\n" + "="*50)
    demo_object_detection()
    print("\n" + "="*50)
    demo_image_classification()
    print("\n" + "="*50)
    demo_image_segmentation()
    
    print("\n✅ 计算机视觉演示完成！") 