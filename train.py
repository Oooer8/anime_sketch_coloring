"""优化后的训练脚本 - 完整版（支持断点续训和灵活的学习率策略）

场景 1：初始训练
python train.py \
    --scheduler_type onecycle \
    --num_epochs 300 \
    --lr 2e-4

    
场景 2：延长训练
python train.py \
    --resume checkpoints/best_model.pth \
    --num_epochs 500 \
    --reset_scheduler  # 重新开始学习率计划

    
场景 3：使用周期性重启（推荐）
# 初始训练
python train.py --scheduler_type cosine_restart --cosine_t0 50

# 继续训练（自动重启学习率）
python train.py \
    --resume checkpoints/best_model.pth \
    --scheduler_type cosine_restart \
    --num_epochs 500



场景 4：微调
python train.py \
    --resume checkpoints/best_model.pth \
    --finetune \
    --finetune_lr_ratio 0.1 \
    --num_epochs 350


"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import numpy as np
import sys
import json

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


def save_checkpoint(model, optimizer, scheduler, scaler, ema, epoch, 
                   train_loss, val_loss, args, save_path, is_best=False):
    """保存检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        scaler: 梯度缩放器
        ema: EMA 对象
        epoch: 当前 epoch
        train_loss: 训练损失
        val_loss: 验证损失
        args: 训练参数
        save_path: 保存路径
        is_best: 是否为最佳模型
    """
    # 需要保存的模型架构参数（用于重建模型）
    model_config = {
        'base_channels': args.base_channels,
        'time_emb_dim': args.time_emb_dim,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'stochastic_depth': args.stochastic_depth,
    }
    
    # 需要保存的训练参数（用于恢复训练状态）
    training_config = {
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'lr': args.lr,
        'accumulation_steps': args.accumulation_steps,
        'use_ema': args.use_ema,
        'sketch_method': args.sketch_method,
        'num_workers': args.num_workers,
        'scheduler_type': args.scheduler_type,
    }
    
    save_dict = {
        # 训练状态
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'is_best': is_best,
        
        # 模型和优化器状态
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict(),
        
        # 配置
        'model_config': model_config,
        'training_config': training_config,
        'all_args': vars(args),  # 保存所有参数以供参考
    }
    
    # 保存 EMA
    if ema is not None:
        save_dict['ema_shadow'] = ema.shadow
        save_dict['ema_decay'] = ema.decay
    
    torch.save(save_dict, save_path)
    
    # 同时保存一份可读的配置文件
    config_path = save_path.parent / f"{save_path.stem}_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'model_config': model_config,
            'training_config': training_config,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 检查点已保存: {save_path}")


def load_checkpoint(checkpoint_path, device='cuda'):
    """加载检查点并返回配置
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        checkpoint: 检查点字典
        model_config: 模型配置
        training_config: 训练配置
    """
    print(f"\n{'='*70}")
    print(f"从检查点加载: {checkpoint_path}")
    print(f"{'='*70}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取配置
    model_config = checkpoint.get('model_config', {})
    training_config = checkpoint.get('training_config', {})
    
    # 显示检查点信息
    print(f"检查点信息:")
    print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  训练损失: {checkpoint.get('train_loss', 'Unknown'):.4f}")
    print(f"  验证损失: {checkpoint.get('val_loss', 'Unknown'):.4f}")
    print(f"  是否最佳: {checkpoint.get('is_best', False)}")
    
    print(f"\n模型配置:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    print(f"\n训练配置:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    print(f"{'='*70}\n")
    
    return checkpoint, model_config, training_config


def verify_config_match(current_args, saved_config, config_type='model'):
    """验证配置是否匹配
    
    Args:
        current_args: 当前参数
        saved_config: 保存的配置
        config_type: 配置类型 ('model' 或 'training')
    
    Returns:
        bool: 是否匹配
        list: 不匹配的参数列表
    """
    mismatches = []
    
    for key, saved_value in saved_config.items():
        current_value = getattr(current_args, key, None)
        if current_value is not None and current_value != saved_value:
            mismatches.append({
                'param': key,
                'current': current_value,
                'saved': saved_value
            })
    
    return len(mismatches) == 0, mismatches


def create_scheduler(optimizer, args, train_loader, start_epoch=1):
    """创建学习率调度器
    
    Args:
        optimizer: 优化器
        args: 参数
        train_loader: 训练数据加载器
        start_epoch: 起始 epoch（用于计算剩余 epoch）
    
    Returns:
        scheduler: 学习率调度器
        scheduler_needs_metric: 是否需要传入 metric（ReduceLROnPlateau）
    """
    remaining_epochs = args.num_epochs - start_epoch + 1
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    
    scheduler_needs_metric = False
    
    if args.scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr * 1.5,
            epochs=remaining_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            anneal_strategy='cos',
            div_factor=1,
            final_div_factor=1e4
        )
        print(f"✓ OneCycleLR: {remaining_epochs} epochs, {steps_per_epoch} steps/epoch")
        print(f"  max_lr: {args.lr * 1.5:.2e}, final_lr: {args.lr * 1.5 / 1e4:.2e}")
    
    elif args.scheduler_type == 'cosine_restart':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.cosine_t0,
            T_mult=args.cosine_tmult,
            eta_min=args.min_lr
        )
        print(f"✓ CosineAnnealingWarmRestarts: T_0={args.cosine_t0}, T_mult={args.cosine_tmult}")
        print(f"  eta_min: {args.min_lr:.2e}")
    
    elif args.scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
            verbose=True
        )
        scheduler_needs_metric = True
        print(f"✓ ReduceLROnPlateau: factor={args.plateau_factor}, patience={args.plateau_patience}")
        print(f"  min_lr: {args.min_lr:.2e}")
    
    elif args.scheduler_type == 'exponential':
        scheduler = ExponentialLR(
            optimizer,
            gamma=args.exp_gamma
        )
        print(f"✓ ExponentialLR: gamma={args.exp_gamma}")
        print(f"  每 epoch 学习率衰减到原来的 {args.exp_gamma:.2%}")
    
    elif args.scheduler_type == 'none':
        scheduler = None
        print(f"✓ 不使用学习率调度器，固定学习率: {args.lr:.2e}")
    
    else:
        raise ValueError(f"未知的调度器类型: {args.scheduler_type}")
    
    return scheduler, scheduler_needs_metric


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, 
                   perceptual_loss, ema, accumulation_steps=1, 
                   scheduler_needs_metric=False, writer=None):
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
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
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
            
            # 更新学习率（除了 ReduceLROnPlateau）
            if scheduler is not None and not scheduler_needs_metric:
                scheduler.step()
            
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
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="datasets/anime_faces")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--sketch_method", type=str, default="canny")
    parser.add_argument("--num_workers", type=int, default=16)
    
    # 模型参数（影响模型架构）
    parser.add_argument("--base_channels", type=int, default=96)
    parser.add_argument("--time_emb_dim", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stochastic_depth", type=float, default=0.1)
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--use_ema", action="store_true", default=True)
    
    # 学习率调度器
    parser.add_argument("--scheduler_type", type=str, 
                       default="onecycle",
                       choices=["onecycle", "cosine_restart", "reduce_on_plateau", "exponential", "none"],
                       help="学习率调度器类型")
    parser.add_argument("--reset_scheduler", action="store_true",
                       help="重置学习率调度器（用于延长训练）")
    
    # OneCycleLR 参数（已在 create_scheduler 中设置）
    
    # CosineAnnealingWarmRestarts 参数
    parser.add_argument("--cosine_t0", type=int, default=50,
                       help="CosineRestart: 第一个周期的 epoch 数")
    parser.add_argument("--cosine_tmult", type=int, default=1,
                       help="CosineRestart: 周期长度倍增因子")
    
    # ReduceLROnPlateau 参数
    parser.add_argument("--plateau_factor", type=float, default=0.5,
                       help="ReduceLROnPlateau: 学习率衰减因子")
    parser.add_argument("--plateau_patience", type=int, default=10,
                       help="ReduceLROnPlateau: 容忍多少 epoch 不改善")
    
    # ExponentialLR 参数
    parser.add_argument("--exp_gamma", type=float, default=0.95,
                       help="ExponentialLR: 学习率衰减因子")
    
    # 通用调度器参数
    parser.add_argument("--min_lr", type=float, default=1e-7,
                       help="最小学习率")
    
    # 微调模式
    parser.add_argument("--finetune", action="store_true",
                       help="微调模式（使用更小的学习率）")
    parser.add_argument("--finetune_lr_ratio", type=float, default=0.1,
                       help="微调学习率比例（相对于原学习率）")
    
    # 保存和日志
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    
    # 恢复训练
    parser.add_argument("--resume", type=str, default=None, 
                       help="恢复训练的检查点路径")
    parser.add_argument("--resume_epoch", type=int, default=None,
                       help="从指定 epoch 恢复（覆盖检查点中的 epoch）")
    parser.add_argument("--ignore_config_mismatch", action="store_true",
                       help="忽略配置不匹配的警告（不推荐）")
    
    args = parser.parse_args()
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    
    # ========== 处理恢复训练 ==========
    start_epoch = 1
    best_val_loss = float('inf')
    checkpoint = None
    
    if args.resume:
        checkpoint, model_config, training_config = load_checkpoint(args.resume, device)
        
        # 验证模型配置
        model_match, model_mismatches = verify_config_match(args, model_config, 'model')
        if not model_match:
            print("⚠️  警告：模型配置不匹配！")
            print("不匹配的参数:")
            for mismatch in model_mismatches:
                print(f"  {mismatch['param']}: 当前={mismatch['current']}, 保存={mismatch['saved']}")
            
            if not args.ignore_config_mismatch:
                print("\n❌ 模型配置必须匹配才能加载权重！")
                print("解决方案:")
                print("  1. 使用保存的配置参数重新运行")
                print("  2. 使用 --ignore_config_mismatch 强制加载（可能导致错误）")
                sys.exit(1)
            else:
                print("⚠️  使用 --ignore_config_mismatch，强制使用当前配置")
                # 使用保存的配置创建模型
                for key, value in model_config.items():
                    setattr(args, key, value)
        else:
            # 使用保存的模型配置
            for key, value in model_config.items():
                setattr(args, key, value)
        
        # 验证训练配置（仅警告）
        training_match, training_mismatches = verify_config_match(args, training_config, 'training')
        if not training_match:
            print("\nℹ️  训练配置有变化:")
            for mismatch in training_mismatches:
                print(f"  {mismatch['param']}: 当前={mismatch['current']}, 保存={mismatch['saved']}")
            print("将使用当前的训练配置继续训练\n")
    
    # ========== 打印配置 ==========
    print("="*70)
    print("Flow Matching 简笔画上色训练")
    print("="*70)
    print(f"使用设备: {device}")
    print(f"\n模型配置:")
    print(f"  Base Channels: {args.base_channels}")
    print(f"  Time Embedding Dim: {args.time_emb_dim}")
    print(f"  Num Heads: {args.num_heads}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Stochastic Depth: {args.stochastic_depth}")
    print(f"\n训练配置:")
    print(f"  批次大小: {args.batch_size}")
    print(f"  图像大小: {args.image_size}")
    print(f"  训练轮数: {args.num_epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  梯度累积步数: {args.accumulation_steps}")
    print(f"  使用 EMA: {args.use_ema}")
    print(f"  调度器类型: {args.scheduler_type}")
    if args.finetune:
        print(f"  微调模式: 是 (lr ratio: {args.finetune_lr_ratio})")
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
    
    # 可视化样本（仅在新训练时）
    if not args.resume:
        visualize_pairs(train_loader, num_samples=8, save_path='train_samples.png')
    
    # 创建模型
    model = AdvancedFlowMatchingModel(
        base_channels=args.base_channels,
        time_emb_dim=args.time_emb_dim,
        num_heads=args.num_heads,
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
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 感知损失
    perceptual_loss = PerceptualLoss().to(device)
    
    # EMA
    ema = EMA(model, decay=0.9999) if args.use_ema else None
    
    # ========== 加载检查点状态 ==========
    if checkpoint is not None:
        # 加载模型
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ 模型参数已加载")
        
        # 加载优化器
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ 优化器状态已加载")
        
        # 加载 GradScaler
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✓ GradScaler 状态已加载")
        
        # 加载 EMA
        if ema is not None and 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']
            if 'ema_decay' in checkpoint:
                ema.decay = checkpoint['ema_decay']
            print("✓ EMA 参数已加载")
        
        # 恢复 epoch
        start_epoch = checkpoint['epoch'] + 1
        if args.resume_epoch is not None:
            start_epoch = args.resume_epoch
            print(f"⚠️  手动设置起始 epoch 为 {start_epoch}")
        
        # 恢复最佳验证损失
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
            print(f"✓ 最佳验证损失: {best_val_loss:.4f}")
        
        # 微调模式
        if args.finetune:
            print(f"\n{'='*70}")
            print("🎯 微调模式")
            print(f"{'='*70}")
            finetune_lr = args.lr * args.finetune_lr_ratio
            for param_group in optimizer.param_groups:
                param_group['lr'] = finetune_lr
            print(f"微调学习率: {finetune_lr:.2e} (原始 lr 的 {args.finetune_lr_ratio:.1%})")
            print(f"{'='*70}\n")
    
    # ========== 创建学习率调度器 ==========
    scheduler_needs_metric = False
    
    if args.resume and not args.reset_scheduler and not args.finetune:
        # 尝试加载原调度器
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            # 先创建同类型的调度器
            scheduler, scheduler_needs_metric = create_scheduler(
                optimizer, args, train_loader, start_epoch
            )
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✓ 学习率调度器已加载，当前 lr: {optimizer.param_groups[0]['lr']:.2e}\n")
            except Exception as e:
                print(f"⚠️  加载调度器失败: {e}")
                print("将创建新的调度器\n")
                scheduler, scheduler_needs_metric = create_scheduler(
                    optimizer, args, train_loader, start_epoch
                )
        else:
            scheduler, scheduler_needs_metric = create_scheduler(
                optimizer, args, train_loader, start_epoch
            )
    else:
        # 创建新调度器
        if args.reset_scheduler:
            print("⚠️  重置学习率调度器\n")
        if args.finetune:
            # 微调模式使用指数衰减
            print("使用微调专用的指数衰减调度器")
            scheduler = ExponentialLR(optimizer, gamma=0.95)
            scheduler_needs_metric = False
            print(f"✓ ExponentialLR: gamma=0.95\n")
        else:
            scheduler, scheduler_needs_metric = create_scheduler(
                optimizer, args, train_loader, start_epoch
            )
    
    if checkpoint is not None:
        print(f"✓ 将从 Epoch {start_epoch} 继续训练")
        print(f"{'='*70}\n")
    
    # TensorBoard
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 训练
    try:
        for epoch in range(start_epoch, args.num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{args.num_epochs}")
            print(f"{'='*70}")
            print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, device, epoch,
                perceptual_loss, ema, args.accumulation_steps, 
                scheduler_needs_metric, writer
            )
            
            # 使用 EMA 模型进行验证
            if ema is not None:
                ema.apply_shadow()
            
            val_loss = validate(model, val_loader, device, perceptual_loss, epoch, writer)
            
            if ema is not None:
                ema.restore()
            
            # 对于 ReduceLROnPlateau，需要传入 val_loss
            if scheduler is not None and scheduler_needs_metric:
                scheduler.step(val_loss)
            
            print(f"\n训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            
            # 保存最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler, ema, epoch,
                    train_loss, val_loss, args,
                    save_dir / "best_model.pth",
                    is_best=True
                )
                print(f"✓ 保存最佳模型 (验证损失: {val_loss:.4f})")
            
            # 定期保存检查点
            if epoch % 10 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, ema, epoch,
                    train_loss, val_loss, args,
                    save_dir / f"checkpoint_epoch_{epoch}.pth",
                    is_best=False
                )
                
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
        save_checkpoint(
            model, optimizer, scheduler, scaler, ema, epoch,
            train_loss, val_loss, args,
            save_dir / "interrupted_checkpoint.pth",
            is_best=False
        )
        print(f"✓ 模型已保存到: {save_dir / 'interrupted_checkpoint.pth'}")
    
    finally:
        writer.close()


if __name__ == '__main__':
    main()
