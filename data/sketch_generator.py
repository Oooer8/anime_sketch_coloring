"""简笔画生成模块"""

import cv2
import numpy as np
from typing import Literal

SketchMethod = Literal['canny', 'xdog', 'hed', 'sobel']


class SketchGenerator:
    """简笔画生成器 - 支持多种边缘检测算法"""
    
    METHODS = ['canny', 'xdog', 'hed', 'sobel']
    
    def __init__(self, method: SketchMethod = 'canny'):
        """
        Args:
            method: 简笔画生成方法
        """
        if method not in self.METHODS:
            raise ValueError(f"不支持的方法: {method}. 可选: {self.METHODS}")
        
        self.method = method
    
    def generate(self, image: np.ndarray) -> np.ndarray:
        """
        生成简笔画
        
        Args:
            image: RGB 图像 (H, W, 3), uint8
        
        Returns:
            sketch: RGB 简笔画 (H, W, 3), uint8
        """
        if self.method == 'canny':
            return self._canny_edge(image)
        elif self.method == 'xdog':
            return self._xdog_edge(image)
        elif self.method == 'hed':
            return self._hed_edge(image)
        elif self.method == 'sobel':
            return self._sobel_edge(image)
    
    @staticmethod
    def _canny_edge(image: np.ndarray) -> np.ndarray:
        """Canny 边缘检测 - 快速且效果好"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        sketch = 255 - edges
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    
    @staticmethod
    def _xdog_edge(image: np.ndarray) -> np.ndarray:
        """XDoG - 艺术化的简笔画效果"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        sigma1, sigma2 = 0.5, 0.8
        blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
        blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
        dog = blur1 - blur2
        
        epsilon, phi = 0.01, 10
        xdog = dog / (1 + np.exp(-phi * (dog - epsilon)))
        xdog = np.clip(xdog * 255, 0, 255).astype(np.uint8)
        
        _, sketch = cv2.threshold(xdog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    
    @staticmethod
    def _hed_edge(image: np.ndarray) -> np.ndarray:
        """HED-like 边缘检测（简化版）"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        _, sketch = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        sketch = 255 - sketch
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    
    @staticmethod
    def _sobel_edge(image: np.ndarray) -> np.ndarray:
        """Sobel 边缘检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(magnitude / magnitude.max() * 255)
        
        _, sketch = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)
        sketch = 255 - sketch
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
