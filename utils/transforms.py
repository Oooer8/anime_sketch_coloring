"""数据变换工具"""

from torchvision import transforms
from typing import Tuple


def get_transforms(
    image_size: int = 256,
    normalize_color: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    获取数据变换
    
    Args:
        image_size: 图像大小
        normalize_color: 是否归一化上色图像到 [-1, 1]
    
    Returns:
        (color_transform, sketch_transform)
    """
    # 上色图像变换
    color_ops = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    
    if normalize_color:
        color_ops.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )
    
    color_transform = transforms.Compose(color_ops)
    
    # 简笔画变换
    sketch_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    return color_transform, sketch_transform
