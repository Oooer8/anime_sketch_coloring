"""可视化工具"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_pairs(
    loader,
    num_samples: int = 8,
    save_path: str = 'visualization.png'
):
    """可视化简笔画-上色图像对"""
    sketches, colors = next(iter(loader))
    num_samples = min(num_samples, len(sketches))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # 简笔画
        sketch = sketches[i].permute(1, 2, 0).cpu().numpy()
        sketch = np.clip(sketch, 0, 1)
        axes[0, i].imshow(sketch)
        axes[0, i].set_title('简笔画', fontsize=10)
        axes[0, i].axis('off')
        
        # 上色图像
        color = colors[i].permute(1, 2, 0).cpu().numpy()
        if color.min() < 0:
            color = (color + 1) / 2
        color = np.clip(color, 0, 1)
        axes[1, i].imshow(color)
        axes[1, i].set_title('彩色图像', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")
    plt.close()


def save_comparison(
    original,
    sketch,
    colored,
    save_path: str = 'comparison.png'
):
    """保存对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("原始图像")
    axes[0].axis("off")
    
    axes[1].imshow(sketch)
    axes[1].set_title("简笔画")
    axes[1].axis("off")
    
    axes[2].imshow(colored)
    axes[2].set_title("上色结果")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
