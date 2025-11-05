"""Flow Matching 模型"""

import torch
import torch.nn as nn


class SimpleFlowMatchingModel(nn.Module):
    """简单的 Flow Matching 模型"""
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + out_channels + 1, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, sketch, noisy_color, t):
        """
        Args:
            sketch: 简笔画 [B, 3, H, W]
            noisy_color: 带噪声的上色图像 [B, 3, H, W]
            t: 时间步 [B, 1]
        
        Returns:
            velocity: 预测的速度场 [B, 3, H, W]
        """
        # 扩展时间步到空间维度
        B, _, H, W = sketch.shape
        t_expanded = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        
        # 拼接输入
        x = torch.cat([sketch, noisy_color, t_expanded], dim=1)
        
        # 前向传播
        features = self.encoder(x)
        velocity = self.decoder(features)
        
        return velocity
