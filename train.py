"""优化后的训练脚本 - 完整版"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys

from data import SketchColorPairDataset, check_and_download_dataset
from models import AdvancedFlowMatchingModel
from utils import get_transforms, visualize_pairs


class PerceptualLoss(nn.Module):
    """感知损失，使用预训练的 VGG 特征"""
    def __init__(self):
        super().__init__()
        # 使用 VGG16 的前几层
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        self.layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ])
        
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False
        
        self.weights = [1.0, 1.0, 1.0]
    
    def forward(self, pred, target):
        loss = 0.0
        
        for i, layer in enumerate(self.layers):
            pred = layer(pred)
            target = layer(target)
            loss += self.weights[i] * nn.functional.l1_loss(pred, target)
        
        return loss


class EMA:
    """指数移动平均 (Exponential Moving Average)
    
    用于平滑模型权重，提升生成质量和训练稳定性
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化 shadow 参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新 shadow 参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用 shadow 参数到模型（用于验证/推理）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数（继续训练）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, 
                   perceptual_loss, ema, accumulation_steps=1, writer=None):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_mse_loss = 0
    total_perceptual_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for i, (sketch, color) in enumerate(pbar):
        sketch = sketch.to(device)
        color = color.to(device)
        
        # 混合精度训练
        with autocast(device_type='cuda'):
            # Flow Matching 训练
            # 使用重要性采样：在中间时间步采样更多
            t = torch.rand(sketch.size(0), 1, device=device)
            t = torch.sigmoid(torch.randn_like(t))
            
            noise = torch.randn_like(color)
            noisy_color = t.view(-1, 1, 1, 1) * color + (1 - t.view(-1, 1, 1, 1)) * noise
            
            pred_velocity = model(sketch, noisy_color, t)
            target_velocity = color - noise
            
            # MSE 损失
            mse_loss = nn.functional.mse_loss(pred_velocity, target_velocity)
            
            # 感知损失（仅在后期训练时使用）
            if epoch > 10:
                # 从速度场重建图像
                pred_color = noisy_color + pred_velocity * (1 - t.view(-1, 1, 1, 1))
                pred_color = torch.clamp(pred_color, -1, 1)
                
                # 归一化到 [0, 1] 用于 VGG
                pred_color_norm = (pred_color + 1) / 2
                color_norm = (color + 1) / 2
                
                perc_loss = perceptual_loss(pred_color_norm, color_norm)
                loss = mse_loss + 0.1 * perc_loss
                
                total_perceptual_loss += perc_loss.item()
            else:
                loss = mse_loss
                perc_loss = torch.tensor(0.0)
        
        # 梯度缩放和累积
        scaler.scale(loss / accumulation_steps).backward()
        
        if (i + 1) % accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # 更新学习率
            optimizer.zero_grad()
            
            # 更新 EMA
            if ema is not None:
                ema.update()
        
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f'{mse_loss.item():.4f}',
            'perc': f'{perc_loss.item():.4f}' if epoch > 10 else 'N/A',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 记录到 TensorBoard
        if writer is not None and i % 50 == 0:
            global_step = (epoch - 1) * len(loader) + i
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/mse_loss', mse_loss.item(), global_step)
            if epoch > 10:
                writer.add_scalar('train/perceptual_loss', perc_loss.item(), global_step)
    
    avg_loss = total_loss / len(loader)
    avg_mse = total_mse_loss / len(loader)
    avg_perc = total_perceptual_loss / len(loader) if epoch > 10 else 0
    
    if writer is not None:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/epoch_mse', avg_mse, epoch)
        if epoch > 10:
            writer.add_scalar('train/epoch_perceptual', avg_perc, epoch)
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
    
    return avg_loss


@torch.no_grad()
def validate(model, loader, device, perceptual_loss, epoch=0, writer=None):
    """验证"""
    model.eval()
    total_loss = 0
    total_mse_loss = 0
    total_perceptual_loss = 0
    
    for sketch, color in tqdm(loader, desc="验证"):
        sketch = sketch.to(device)
        color = color.to(device)
        
        # 使用固定的时间步进行验证
        t = torch.ones(sketch.size(0), 1, device=device) * 0.5
        noise = torch.randn_like(color)
        noisy_color = 0.5 * color + 0.5 * noise
        
        pred_velocity = model(sketch, noisy_color, t)
        target_velocity = color - noise
        
        mse_loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        
        # 感知损失
        pred_color = noisy_color + pred_velocity * 0.5
        pred_color = torch.clamp(pred_color, -1, 1)
        
        pred_color_norm = (pred_color + 1) / 2
        color_norm = (color + 1) / 2
        perc_loss = perceptual_loss(pred_color_norm, color_norm)
        
        loss = mse_loss + 0.1 * perc_loss
        
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_perceptual_loss += perc_loss.item()
    
    avg_loss = total_loss / len(loader)
    avg_mse = total_mse_loss / len(loader)
    avg_perc = total_perceptual_loss / len(loader)
    
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/mse_loss', avg_mse, epoch)
        writer.add_scalar('val/perceptual_loss', avg_perc, epoch)
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="训练简笔画上色模型")
    parser.add_argument("--data_dir", type=str, default="datasets/anime_faces")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--sketch_method", type=str, default="canny")
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--base_channels", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stochastic_depth", type=float, default=0.1)
    parser.add_argument("--time_emb_dim", type=int, default=384)
    
    args = parser.parse_args()
    
    device_type='cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    print("="*70)
    print("Flow Matching 简笔画上色训练")
    print("="*70)
    print(f"使用设备: {device}")
    print(f"批次大小: {args.batch_size}")
    print(f"图像大小: {args.image_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率: {args.lr}")
    print(f"Dropout: {args.dropout}")
    print(f"Stochastic Depth: {args.stochastic_depth}")
    print(f"使用 EMA: {args.use_ema}")
    print("="*70 + "\n")
    
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
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"训练批次数: {len(train_loader)}\n")
    
    # 可视化样本
    visualize_pairs(train_loader, num_samples=8, save_path='train_samples.png')
    
    # 创建模型
    model = AdvancedFlowMatchingModel(
        base_channels=args.base_channels,
        time_emb_dim=args.time_emb_dim,
        num_heads=4,
        dropout=args.dropout,
        stochastic_depth=args.stochastic_depth
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)\n")
    
    # 优化器 (使用 AdamW)
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=0.01,  # L2 正则化
        betas=(0.9, 0.999)
    )
    
    # OneCycle 学习率调度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr * 1.5,
        epochs=args.num_epochs,
        steps_per_epoch=len(train_loader) // args.accumulation_steps,
        pct_start=0.15,  # 用于warm-up
        anneal_strategy='cos',
        div_factor=1,  # 初始 lr = max_lr / 1 = 3e-4
        final_div_factor=1e4  # 最终 lr = max_lr / 1e4 = 3e-8
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 感知损失
    perceptual_loss = PerceptualLoss().to(device)
    
    # EMA
    ema = EMA(model, decay=0.9999) if args.use_ema else None
    
    # TensorBoard
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 训练
    best_val_loss = float('inf')
    
    try:
        for epoch in range(1, args.num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{args.num_epochs}")
            print(f"{'='*70}")
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, epoch,
                perceptual_loss, ema, args.accumulation_steps, writer
            )
            
            # 使用 EMA 模型进行验证
            if ema is not None:
                ema.apply_shadow()
            
            val_loss = validate(model, val_loader, device, perceptual_loss, epoch, writer)
            
            if ema is not None:
                ema.restore()
            
            print(f"\n训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'args': vars(args)
                }
                if ema is not None:
                    save_dict['ema_shadow'] = ema.shadow
                
                torch.save(save_dict, save_dir / "best_model.pth")
                print(f"✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
            
            # 定期保存检查点
            if epoch % 10 == 0:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'args': vars(args)
                }
                if ema is not None:
                    save_dict['ema_shadow'] = ema.shadow
                
                torch.save(save_dict, save_dir / f"checkpoint_epoch_{epoch}.pth")
                
                # 可视化结果
                with torch.no_grad():
                    if len(val_loader) > 0:
                        sketch_samples, color_samples = next(iter(val_loader))
                        sketch_samples = sketch_samples[:4].to(device)
                        color_samples = color_samples[:4].to(device)
                        
                        writer.add_images('val/sketches', (sketch_samples + 1) / 2, epoch)
                        writer.add_images('val/ground_truth', (color_samples + 1) / 2, epoch)
        
        print("\n" + "="*70)
        print("✓ 训练完成！")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"模型保存在: {save_dir}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n训练被中断，保存当前模型...")
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args)
        }
        if ema is not None:
            save_dict['ema_shadow'] = ema.shadow
        
        torch.save(save_dict, save_dir / "interrupted_checkpoint.pth")
        print(f"✓ 模型已保存到: {save_dir / 'interrupted_checkpoint.pth'}")
    
    finally:
        writer.close()


if __name__ == '__main__':
    main()
