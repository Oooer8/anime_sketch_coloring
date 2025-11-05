"""组织现有的 Anime Faces 数据集"""

import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def organize_existing_dataset(root_dir: str, val_ratio: float = 0.1):
    """
    组织现有数据集到正确的目录结构
    
    Args:
        root_dir: 数据集根目录
        val_ratio: 验证集比例
    """
    root_path = Path(root_dir)
    
    # 创建目标目录
    train_dir = root_path / 'train' / 'anime'
    val_dir = root_path / 'val' / 'anime'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    print("扫描图像文件...")
    for ext in image_extensions:
        image_files.extend(root_path.glob(f'*{ext}'))
        image_files.extend(root_path.glob(f'*{ext.upper()}'))
    
    # 过滤掉已经在 train/val 目录中的文件
    image_files = [f for f in image_files 
                   if 'train' not in str(f) and 'val' not in str(f)]
    
    total_images = len(image_files)
    
    if total_images == 0:
        print("❌ 未找到图像文件！")
        print(f"请确保图像文件在 {root_dir} 目录下")
        return False
    
    print(f"\n找到 {total_images} 张图像")
    
    # 计算分割点
    val_count = int(total_images * val_ratio)
    train_count = total_images - val_count
    
    print(f"训练集: {train_count} 张")
    print(f"验证集: {val_count} 张")
    
    # 分割数据集
    train_files = image_files[val_count:]
    val_files = image_files[:val_count]
    
    # 复制训练集
    print("\n复制训练集...")
    for i, src_file in enumerate(tqdm(train_files, desc="训练集")):
        dst_file = train_dir / f"train_{i:06d}{src_file.suffix}"
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
    
    # 复制验证集
    print("复制验证集...")
    for i, src_file in enumerate(tqdm(val_files, desc="验证集")):
        dst_file = val_dir / f"val_{i:06d}{src_file.suffix}"
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
    
    print("\n✓ 数据集组织完成！")
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="组织现有数据集")
    parser.add_argument("--root_dir", type=str, default="datasets/anime_faces",
                        help="数据集根目录")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--move", action="store_true",
                        help="移动文件而不是复制（节省空间）")
    
    args = parser.parse_args()
    
    if args.move:
        print("⚠️  警告：将移动文件而不是复制！")
        response = input("确认继续？(y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    organize_existing_dataset(args.root_dir, args.val_ratio)


if __name__ == '__main__':
    main()
