"""推理脚本"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from models import SimpleFlowMatchingModel
from data import SketchGenerator
from utils import save_comparison


def load_image(image_path, size=256):
    """加载并预处理图像"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size))
    return np.array(img)


def preprocess(img_np):
    """预处理为模型输入"""
    tensor = torch.from_numpy(img_np).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def postprocess(tensor):
    """后处理为可显示图像"""
    tensor = tensor.squeeze(0)
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    return (img_np * 255).astype(np.uint8)


@torch.no_grad()
def sample_from_model(model, sketch_tensor, device, steps=50):
    """从模型采样生成彩色图像"""
    x = torch.randn_like(sketch_tensor).to(device)
    dt = 1.0 / steps
    
    for i in tqdm(range(steps), desc="生成中"):
        t = torch.ones(x.size(0), 1, device=device) * (1.0 - i / steps)
        velocity = model(sketch_tensor, x, t)
        x = x + velocity * dt
    
    return x


def main():
    parser = argparse.ArgumentParser(description="简笔画上色推理")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--input", type=str, required=True, help="输入图像")
    parser.add_argument("--output", type=str, default="colored_output.png")
    parser.add_argument("--sketch_method", type=str, default="canny")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--use_sketch", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = SimpleFlowMatchingModel().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载图像
    image_np = load_image(args.input)
    
    # 生成简笔画
    if args.use_sketch:
        sketch_np = image_np
    else:
        generator = SketchGenerator(method=args.sketch_method)
        sketch_np = generator.generate(image_np)
    
    # 推理
    sketch_tensor = preprocess(sketch_np).to(device)
    colored_tensor = sample_from_model(model, sketch_tensor, device, args.steps)
    colored_np = postprocess(colored_tensor)
    
    # 保存结果
    Image.fromarray(colored_np).save(args.output)
    print(f"结果已保存到: {args.output}")
    
    # 保存对比图
    save_comparison(image_np, sketch_np, colored_np, "comparison.png")


if __name__ == "__main__":
    main()
