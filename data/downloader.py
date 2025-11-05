"""Anime Faces 数据集下载模块"""

import requests
import zipfile
import shutil
import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional


class AnimeFacesDownloader:
    """Anime Faces 数据集下载器"""
    
    DATASET_URL = "https://huggingface.co/datasets/huggan/anime-faces/resolve/main/anime-faces.zip"
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_dir = self.root_dir / 'train' / 'anime'
        self.val_dir = self.root_dir / 'val' / 'anime'
        self.download_dir = self.root_dir / 'downloads'
        self.metadata_file = self.root_dir / 'dataset_info.json'
    
    def check_dataset_exists(self) -> dict:
        """检查数据集是否存在"""
        train_exists = self.train_dir.exists() and len(list(self.train_dir.glob('*.png'))) > 0
        val_exists = self.val_dir.exists() and len(list(self.val_dir.glob('*.png'))) > 0
        
        return {
            'train': train_exists,
            'val': val_exists,
            'complete': train_exists and val_exists
        }
    
    def download_file(self, url: str, save_path: Path):
        """下载文件并显示进度条"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc="下载数据集",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    def download_and_extract(self) -> bool:
        """下载并解压数据集"""
        print("\n" + "="*70)
        print("从 Hugging Face 下载 Anime Faces 数据集")
        print("="*70)
        
        # 下载
        self.download_dir.mkdir(exist_ok=True)
        zip_path = self.download_dir / 'anime-faces.zip'
        
        print("\n下载数据集文件...")
        self.download_file(self.DATASET_URL, zip_path)
        
        # 解压
        print("\n解压数据集...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.download_dir)
        
        # 整理目录
        self._organize_dataset()
        
        # 保存元数据
        train_count = len(list(self.train_dir.glob('*.jpg')))
        val_count = len(list(self.val_dir.glob('*.jpg')))
        
        self._save_metadata({
            'source': 'huggan',
            'train_samples': train_count,
            'val_samples': val_count
        })
        
        print(f"\n✓ 数据集下载完成！")
        print(f"  训练集: {train_count} 张图像")
        print(f"  验证集: {val_count} 张图像")
        
        return True
    
    def _organize_dataset(self):
        """整理数据集目录结构"""
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
        images_dir = self.download_dir / 'anime-faces'
        if not images_dir.exists():
            raise FileNotFoundError(f"未找到解压后的图像目录: {images_dir}")
        
        image_files = list(images_dir.glob('*.jpg'))
        total = len(image_files)
        
        if total == 0:
            raise ValueError("未找到图像文件")
        
        # 90% 训练, 10% 验证
        val_split = int(total * 0.1)
        train_files = image_files[val_split:]
        val_files = image_files[:val_split]
        
        print(f"\n整理数据集:")
        print(f"  总图像数: {total}")
        print(f"  训练集: {len(train_files)}")
        print(f"  验证集: {len(val_files)}")
        
        # 复制文件
        for i, file in enumerate(tqdm(train_files, desc="准备训练集")):
            shutil.copy(file, self.train_dir / f"train_{i:06d}.jpg")
        
        for i, file in enumerate(tqdm(val_files, desc="准备验证集")):
            shutil.copy(file, self.val_dir / f"val_{i:06d}.jpg")
    
    def _save_metadata(self, info: dict):
        """保存元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(info, f, indent=2)


def check_and_download_dataset(root_dir: str, auto_download: bool = True) -> bool:
    """
    检查数据集，如果不存在则下载
    
    Args:
        root_dir: 数据集根目录
        auto_download: 是否自动下载
    
    Returns:
        bool: 数据集是否可用
    """
    downloader = AnimeFacesDownloader(root_dir)
    status = downloader.check_dataset_exists()
    
    if status['complete']:
        print(f"✓ 数据集已存在: {root_dir}")
        return True
    
    print(f"\n{'='*70}")
    print("数据集不存在或不完整")
    print(f"{'='*70}")
    
    if auto_download:
        response = input("\n是否从 Hugging Face 下载数据集? (y/n): ")
        if response.lower() == 'y':
            try:
                return downloader.download_and_extract()
            except Exception as e:
                print(f"\n下载失败: {e}")
                return False
    
    return False
