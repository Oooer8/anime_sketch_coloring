"""训练脚本"""

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from data import SketchColorPairDataset, check_and_download_dataset
from models import SimpleFlowMatchingModel
from utils import get_transforms, visualize_pairs


def train_one_epoch(model, loader, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for sketch, color in pbar:
        sketch = sketch.to(device)
        color = color.to(device)
        
        # Flow Matching 训练
        t = torch.rand(sketch.size(0), 1, device=device)
        noise = torch.randn_like(color)
        noisy_color = t.view(-1, 1, 1, 1) * color + (1 - t.view(-1, 1, 1, 1)) * noise
        
        pred_velocity = model(sketch, noisy_color, t)
        target_velocity = color - noise
        
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    """验证"""
    model.eval()
    total_loss = 0
    
    for sketch, color in tqdm(loader, desc="验证"):
        sketch = sketch.to(device)
        color = color.to(device)
        
        t = torch.ones(sketch.size(0), 1, device=device) * 0.5
        noise = torch.randn_like(color)
        noisy_color = 0.5 * color + 0.5 * noise
        
        pred_velocity = model(sketch, noisy_color, t)
        target_velocity = color - noise
        
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="训练简笔画上色模型")
    parser.add_argument("--data_dir", type=str, default="datasets/anime_faces")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--sketch_method", type=str, default="canny")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 检查数据集
    if not check_and_download_dataset(args.data_dir):
        return
    
    # 加载数据
    color_transform, sketch_transform = get_transforms(args.image_size)
    
    train_dataset = SketchColorPairDataset(
        root_dir=args.data_dir,
        split='train',
        sketch_method=args.sketch_method,
        use_cache=True,
        color_transform=color_transform,
        sketch_transform=sketch_transform
    )
    
    val_dataset = SketchColorPairDataset(
        root_dir=args.data_dir,
        split='val',
        sketch_method=args.sketch_method,
        use_cache=True,
        color_transform=color_transform,
        sketch_transform=sketch_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 可视化样本
    visualize_pairs(train_loader, num_samples=8, save_path='train_samples.png')
    
    # 创建模型
    model = SimpleFlowMatchingModel().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 训练
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*70}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / "best_model.pth")
            print(f"✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
        
        # 定期保存检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, save_dir / f"checkpoint_epoch_{epoch}.pth")
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()
