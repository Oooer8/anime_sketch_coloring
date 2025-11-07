"""推理脚本 - 自动读取配置版本

使用示例

单张推理
# 随机选择一张简笔画
python inference.py --model checkpoints/best_model.pth

# 指定输入图像（彩色图）
python inference.py --model checkpoints/best_model.pth --input my_image.png

# 指定输入图像（已经是简笔画）
python inference.py --model checkpoints/best_model.pth --input sketch.png --use_sketch

# 使用更高质量的采样
python inference.py --model checkpoints/best_model.pth --steps 100 --method heun



批量推理
# 批量处理验证集
python inference.py \
    --model checkpoints/best_model.pth \
    --batch \
    --batch_output_dir outputs/validation

# 批量处理前 20 个样本
python inference.py \
    --model checkpoints/best_model.pth \
    --batch \
    --num_samples 20 \
    --batch_output_dir outputs/sample_20

# 使用高质量采样
python inference.py \
    --model checkpoints/best_model.pth \
    --batch \
    --steps 100 \
    --method heun


设置随机种子（可复现）
python inference.py --model checkpoints/best_model.pth --seed 42

"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
import os
import json

from models import AdvancedFlowMatchingModel
from data import SketchGenerator
from utils import save_comparison


def load_checkpoint_with_config(checkpoint_path, device='cuda'):
    """加载检查点并返回模型配置
    
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
    model_config = checkpoint.get('model_config', None)
    training_config = checkpoint.get('training_config', None)
    
    # 如果没有保存配置，使用默认值
    if model_config is None:
        print("⚠️  检查点中未找到模型配置，使用默认配置")
        model_config = {
            'base_channels': 96,
            'time_emb_dim': 384,
            'num_heads': 4,
            'dropout': 0.1,
            'stochastic_depth': 0.1,
        }
    
    if training_config is None:
        print("⚠️  检查点中未找到训练配置")
        training_config = {}
    
    # 显示检查点信息
    print(f"\n检查点信息:")
    print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"  训练损失: {checkpoint.get('train_loss', 'Unknown'):.4f}")
    print(f"  验证损失: {checkpoint.get('val_loss', 'Unknown'):.4f}")
    print(f"  是否最佳: {checkpoint.get('is_best', False)}")
    
    print(f"\n模型配置:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    if training_config:
        print(f"\n训练配置:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
    
    print(f"{'='*70}\n")
    
    return checkpoint, model_config, training_config


def create_model_from_config(model_config, device='cuda'):
    """根据配置创建模型
    
    Args:
        model_config: 模型配置字典
        device: 设备
        
    Returns:
        model: 创建的模型
    """
    model = AdvancedFlowMatchingModel(
        base_channels=model_config.get('base_channels', 96),
        time_emb_dim=model_config.get('time_emb_dim', 384),
        num_heads=model_config.get('num_heads', 4),
        dropout=model_config.get('dropout', 0.1),
        stochastic_depth=model_config.get('stochastic_depth', 0.1)
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)\n")
    
    return model


def load_image(image_path, size=None):
    """加载并预处理图像
    
    Args:
        image_path: 图像路径
        size: 目标尺寸，如果为 None 则保持原始尺寸
    """
    img = Image.open(image_path).convert('RGB')
    if size is not None:
        img = img.resize((size, size))
    return np.array(img)


def get_random_sketch(sketch_dir="datasets/anime_faces/sketches/canny/val/anime"):
    """从指定目录随机选择一个简笔画"""
    if not os.path.exists(sketch_dir):
        raise FileNotFoundError(f"简笔画目录不存在: {sketch_dir}")
    
    sketch_files = [f for f in os.listdir(sketch_dir) if f.endswith('.png')]
    if not sketch_files:
        raise FileNotFoundError(f"在 {sketch_dir} 中未找到简笔画文件")
    
    random_sketch = random.choice(sketch_files)
    sketch_path = os.path.join(sketch_dir, random_sketch)
    print(f"随机选择简笔画: {sketch_path}")
    return sketch_path


def preprocess(img_np, normalize=True):
    """预处理为模型输入
    
    Args:
        img_np: numpy 图像 (H, W, 3)
        normalize: 是否归一化到 [-1, 1]
    """
    tensor = torch.from_numpy(img_np).float() / 255.0
    
    if normalize:
        # 归一化到 [-1, 1]
        tensor = tensor * 2 - 1
    
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def postprocess(tensor):
    """后处理为可显示图像"""
    tensor = tensor.squeeze(0)
    
    # 如果是 [-1, 1] 范围，转换到 [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    return (img_np * 255).astype(np.uint8)


@torch.no_grad()
def sample_from_model(model, sketch_tensor, device, steps=50, method='euler'):
    """从模型采样生成彩色图像
    
    Args:
        model: 模型
        sketch_tensor: 简笔画张量
        device: 设备
        steps: 采样步数
        method: 采样方法 ('euler' 或 'heun')
    """
    x = torch.randn_like(sketch_tensor).to(device)
    dt = 1.0 / steps
    
    if method == 'euler':
        # Euler 方法（一阶）
        for i in tqdm(range(steps), desc="生成中 (Euler)"):
            t = torch.ones(x.size(0), 1, device=device) * (i / steps)
            velocity = model(sketch_tensor, x, t)
            x = x + velocity * dt
    
    elif method == 'heun':
        # Heun 方法（二阶，更准确但慢一倍）
        for i in tqdm(range(steps), desc="生成中 (Heun)"):
            t = torch.ones(x.size(0), 1, device=device) * (i / steps)
            
            # 第一步
            v1 = model(sketch_tensor, x, t)
            x_temp = x + v1 * dt
            
            # 第二步
            t_next = torch.ones(x.size(0), 1, device=device) * ((i + 1) / steps)
            v2 = model(sketch_tensor, x_temp, t_next)
            
            # 平均
            x = x + (v1 + v2) / 2 * dt
    
    else:
        raise ValueError(f"未知的采样方法: {method}")
    
    return x


def batch_inference(model, sketch_dir, output_dir, device, steps=50, 
                   method='euler', num_samples=None, image_size=None):
    """批量推理
    
    Args:
        model: 模型
        sketch_dir: 简笔画目录
        output_dir: 输出目录
        device: 设备
        steps: 采样步数
        method: 采样方法
        num_samples: 处理的样本数量，None 表示全部
        image_size: 图像尺寸
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sketch_files = sorted([f for f in os.listdir(sketch_dir) if f.endswith('.png')])
    
    if num_samples is not None:
        sketch_files = sketch_files[:num_samples]
    
    print(f"\n开始批量推理，共 {len(sketch_files)} 个样本")
    
    for sketch_file in tqdm(sketch_files, desc="批量推理"):
        sketch_path = os.path.join(sketch_dir, sketch_file)
        
        # 加载简笔画
        sketch_np = load_image(sketch_path, size=image_size)
        
        # 推理
        sketch_tensor = preprocess(sketch_np).to(device)
        colored_tensor = sample_from_model(model, sketch_tensor, device, steps, method)
        colored_np = postprocess(colored_tensor)
        
        # 保存结果
        output_path = output_dir / sketch_file
        Image.fromarray(colored_np).save(output_path)
        
        # 尝试找到原始图像并保存对比图
        sketch_name = os.path.splitext(sketch_file)[0]
        original_path = os.path.join("datasets/anime_faces/val", "anime", f"{sketch_name}.png")
        
        if os.path.exists(original_path):
            original_np = load_image(original_path, size=image_size)
            comparison_path = output_dir / f"{sketch_name}_comparison.png"
            save_comparison(original_np, sketch_np, colored_np, str(comparison_path))
    
    print(f"\n✓ 批量推理完成！结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="简笔画上色推理 - 自动读取配置")
    
    # 模型和输入
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", 
                       help="模型检查点路径")
    parser.add_argument("--input", type=str, default=None,
                       help="输入图像路径（如果不指定则随机选择）")
    parser.add_argument("--output", type=str, default="colored_output.png",
                       help="输出图像路径")
    
    # 简笔画相关
    parser.add_argument("--sketch_method", type=str, default="canny",
                       choices=["canny", "hed", "pidinet"],
                       help="简笔画生成方法")
    parser.add_argument("--use_sketch", action="store_true",
                       help="输入已经是简笔画")
    parser.add_argument("--sketch_dir", type=str, 
                       default="datasets/anime_faces/sketches/canny/val/anime",
                       help="随机选择简笔画的目录")
    
    # 采样参数
    parser.add_argument("--steps", type=int, default=50,
                       help="采样步数")
    parser.add_argument("--method", type=str, default="euler",
                       choices=["euler", "heun"],
                       help="采样方法")
    
    # 批量推理
    parser.add_argument("--batch", action="store_true",
                       help="批量推理模式")
    parser.add_argument("--batch_output_dir", type=str, default="outputs/batch",
                       help="批量推理输出目录")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="批量推理的样本数量")
    
    # 其他
    parser.add_argument("--seed", type=int, default=None,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"设置随机种子: {args.seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ========== 加载检查点和配置 ==========
    checkpoint, model_config, training_config = load_checkpoint_with_config(
        args.model, device
    )
    
    # 从训练配置中获取图像尺寸
    image_size = training_config.get('image_size', 64)
    print(f"使用图像尺寸: {image_size}x{image_size}")
    
    # ========== 创建模型 ==========
    model = create_model_from_config(model_config, device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ 模型权重已加载\n")
    
    # 检查是否使用 EMA
    if 'ema_shadow' in checkpoint:
        print("检测到 EMA 权重，加载 EMA 参数...")
        for name, param in model.named_parameters():
            if name in checkpoint['ema_shadow']:
                param.data = checkpoint['ema_shadow'][name]
        print("✓ EMA 权重已加载\n")
    
    # ========== 批量推理模式 ==========
    if args.batch:
        batch_inference(
            model, 
            args.sketch_dir, 
            args.batch_output_dir, 
            device, 
            args.steps,
            args.method,
            args.num_samples,
            image_size
        )
        return
    
    # ========== 单张推理模式 ==========
    # 如果未提供输入图像，则随机选择一个简笔画
    if args.input is None:
        input_path = get_random_sketch(args.sketch_dir)
        args.use_sketch = True
    else:
        input_path = args.input
    
    print(f"输入图像: {input_path}")
    
    # 加载图像
    image_np = load_image(input_path, size=image_size)
    
    # 生成或加载简笔画
    if args.use_sketch:
        sketch_np = image_np
        
        # 对于随机选择的简笔画，尝试找到对应的原始彩色图像
        sketch_filename = os.path.basename(input_path)
        sketch_name = os.path.splitext(sketch_filename)[0]
        original_path = os.path.join("datasets/anime_faces/val", "anime", f"{sketch_name}.png")
        
        if os.path.exists(original_path):
            image_np = load_image(original_path, size=image_size)
            print(f"找到对应的原始图像: {original_path}")
        else:
            print(f"未找到对应的原始图像，将使用空白图像作为占位符")
            image_np = np.zeros_like(sketch_np)
    else:
        print(f"使用 {args.sketch_method} 方法生成简笔画...")
        generator = SketchGenerator(method=args.sketch_method)
        sketch_np = generator.generate(image_np)
    
    # 推理
    print(f"\n开始推理...")
    print(f"  采样步数: {args.steps}")
    print(f"  采样方法: {args.method}")
    
    sketch_tensor = preprocess(sketch_np).to(device)
    colored_tensor = sample_from_model(model, sketch_tensor, device, args.steps, args.method)
    colored_np = postprocess(colored_tensor)
    
    # 保存结果
    Image.fromarray(colored_np).save(args.output)
    print(f"\n✓ 结果已保存到: {args.output}")
    
    # 保存对比图
    comparison_path = os.path.splitext(args.output)[0] + "_comparison.png"
    save_comparison(image_np, sketch_np, colored_np, comparison_path)
    print(f"✓ 对比图已保存到: {comparison_path}")


if __name__ == "__main__":
    main()
