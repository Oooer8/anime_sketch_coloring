"""简笔画-上色图像对数据集"""

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from .sketch_generator import SketchGenerator


class SketchColorPairDataset(Dataset):
    """简笔画-上色图像对数据集"""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        sketch_method: str = 'canny',
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        color_transform: Optional[Callable] = None,
        sketch_transform: Optional[Callable] = None
    ):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train' 或 'val'
            sketch_method: 简笔画生成方法
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            color_transform: 上色图像变换
            sketch_transform: 简笔画变换
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sketch_method = sketch_method
        self.use_cache = use_cache
        
        # 缓存目录
        if cache_dir is None:
            self.cache_dir = self.root_dir / 'sketches' / sketch_method / split
        else:
            self.cache_dir = Path(cache_dir) / sketch_method / split
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载原始数据集
        self.color_dataset = ImageFolder(root=str(self.root_dir / split))
        
        # 变换
        self.color_transform = color_transform
        self.sketch_transform = sketch_transform
        
        # 简笔画生成器
        self.sketch_generator = SketchGenerator(method=sketch_method)
        
        print(f"[{split}] 数据集大小: {len(self.color_dataset)}")
        print(f"[{split}] 简笔画方法: {sketch_method}")
        print(f"[{split}] 缓存目录: {self.cache_dir}")

    def _get_cache_path(self, idx: int) -> Path:
        img_path, label = self.color_dataset.samples[idx]
        img_name = Path(img_path).stem
        class_name = self.color_dataset.classes[label]
        
        cache_path = self.cache_dir / class_name / f"{img_name}.png"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        return cache_path
    
    def _load_or_generate_sketch(self, idx: int) -> Image.Image:
        """加载或生成简笔画"""
        cache_path = self._get_cache_path(idx)
        
        # 从缓存加载
        if self.use_cache and cache_path.exists():
            return Image.open(cache_path).convert('RGB')
        
        # 生成简笔画
        color_img, _ = self.color_dataset[idx]
        color_np = np.array(color_img)
        
        sketch_np = self.sketch_generator.generate(color_np)
        sketch_img = Image.fromarray(sketch_np)
        
        # 保存到缓存
        if self.use_cache:
            sketch_img.save(cache_path)
        
        return sketch_img
    
    def __len__(self) -> int:
        return len(self.color_dataset)
    
    def __getitem__(self, idx: int):
        """返回 (简笔画, 上色图像) 对"""
        # 加载上色图像
        color_img, _ = self.color_dataset[idx]
        
        # 加载或生成简笔画
        sketch_img = self._load_or_generate_sketch(idx)
        
        # 应用变换
        if self.color_transform:
            color_img = self.color_transform(color_img)
        
        if self.sketch_transform:
            sketch_img = self.sketch_transform(sketch_img)
        
        return sketch_img, color_img
