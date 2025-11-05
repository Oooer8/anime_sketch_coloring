"""数据预处理脚本 - 预先生成所有简笔画"""

import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SketchColorPairDataset
from utils import get_transforms


def preprocess_dataset(
    root_dir: str,
    split: str,
    sketch_method: str,
    num_workers: int = 4
):
    """预处理数据集，生成所有简笔画"""
    print(f"\n{'='*70}")
    print(f"预处理 {split} 数据集")
    print(f"  简笔画方法: {sketch_method}")
    print(f"  并行进程数: {num_workers}")
    print(f"{'='*70}\n")
    
    color_transform, sketch_transform = get_transforms()
    
    dataset = SketchColorPairDataset(
        root_dir=root_dir,
        split=split,
        sketch_method=sketch_method,
        use_cache=True,
        color_transform=color_transform,
        sketch_transform=sketch_transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"总共 {len(dataset)} 张图像\n")
    
    for _ in tqdm(loader, desc=f"生成 {split} 简笔画"):
        pass
    
    print(f"\n✓ 预处理完成！")
    print(f"简笔画已保存到: {dataset.cache_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="预处理数据集")
    parser.add_argument("--root_dir", type=str, default="datasets/anime_faces",
                        help="数据集根目录")
    parser.add_argument("--split", type=str, default="all",
                        choices=['train', 'val', 'all'],
                        help="预处理哪个数据集")
    parser.add_argument("--sketch_method", type=str, default="canny",
                        choices=['canny', 'xdog', 'hed', 'sobel'],
                        help="简笔画生成方法")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="并行进程数")
    
    args = parser.parse_args()
    
    if args.split == 'all':
        preprocess_dataset(args.root_dir, 'train', args.sketch_method, args.num_workers)
        preprocess_dataset(args.root_dir, 'val', args.sketch_method, args.num_workers)
    else:
        preprocess_dataset(args.root_dir, args.split, args.sketch_method, args.num_workers)


if __name__ == '__main__':
    main()
