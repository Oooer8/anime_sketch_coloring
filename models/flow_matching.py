"""优化后的 Flow Matching 模型 - 添加 Stochastic Depth"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码，用于时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: [B, 1]
        Returns:
            embeddings: [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time * embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalBatchNorm2d(nn.Module):
    """条件批归一化，用于注入时间信息"""
    def __init__(self, num_features, time_emb_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        
        # 从时间嵌入预测 scale 和 shift
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, num_features * 2)
        )
    
    def forward(self, x, time_emb):
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_emb_dim]
        """
        # 标准批归一化
        x = self.bn(x)
        
        # 从时间嵌入获取 scale 和 shift
        time_params = self.time_proj(time_emb)
        scale, shift = time_params.chunk(2, dim=1)
        
        # 应用条件变换
        scale = scale.view(-1, self.num_features, 1, 1)
        shift = shift.view(-1, self.num_features, 1, 1)
        
        return x * (1 + scale) + shift


class StochasticDepth(nn.Module):
    """随机深度 (Stochastic Depth)
    
    在训练时随机丢弃整个残差分支，提升模型泛化能力
    论文: Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
    """
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x, residual):
        """
        Args:
            x: 主路径特征 [B, C, H, W]
            residual: 残差分支特征 [B, C, H, W]
        Returns:
            融合后的特征 [B, C, H, W]
        """
        if not self.training or self.drop_prob == 0:
            return x + residual
        
        # 生成随机mask (每个样本独立)
        keep_prob = 1 - self.drop_prob
        mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < keep_prob
        
        # 缩放以保持期望值不变
        # E[output] = keep_prob * (x + residual) / keep_prob + (1-keep_prob) * x = x + residual
        return x + residual * mask.float() / keep_prob


class ResidualBlock(nn.Module):
    """改进的残差块，使用条件批归一化和随机深度"""
    def __init__(self, in_channels, out_channels, time_emb_dim, 
                 dropout=0.1, stochastic_depth=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = ConditionalBatchNorm2d(out_channels, time_emb_dim)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = ConditionalBatchNorm2d(out_channels, time_emb_dim)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.stochastic_depth = StochasticDepth(stochastic_depth)
        
        # 如果输入输出通道数不同，添加一个1x1卷积进行调整
        self.skip_connection = nn.Identity() if in_channels == out_channels else \
                              nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x, time_emb):
        identity = self.skip_connection(x)
        
        # 残差分支
        out = self.conv1(x)
        out = self.norm1(out, time_emb)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out, time_emb)
        
        # 使用随机深度融合
        out = self.stochastic_depth(identity, out)
        
        return self.activation(out)


class MultiHeadAttentionBlock(nn.Module):
    """多头自注意力模块"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        
        # 重塑为多头格式
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # 3, B, num_heads, H*W, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        out = attn @ v  # B, num_heads, H*W, head_dim
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # 投影并添加残差连接
        out = self.proj(out)
        return out + residual


class DownSampleBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class UpSampleBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        x = self.upsample(x)
        return self.activation(self.norm(self.conv(x)))


class AdvancedFlowMatchingModel(nn.Module):
    """优化后的 Flow Matching 模型
    
    主要特性:
    - 条件批归一化 (Conditional Batch Normalization)
    - 多头自注意力 (Multi-Head Attention)
    - 随机深度 (Stochastic Depth)
    - U-Net 架构与跳跃连接
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, 
                 time_emb_dim=128, num_heads=4, dropout=0.1, stochastic_depth=0.1):
        super().__init__()
        
        # 时间嵌入
        self.time_embedder = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels + out_channels, base_channels, 3, padding=1)
        
        # 编码器
        self.down_blocks = nn.ModuleList([
            # Level 1: base_channels
            nn.ModuleList([
                ResidualBlock(base_channels, base_channels, time_emb_dim, dropout, stochastic_depth),
                ResidualBlock(base_channels, base_channels, time_emb_dim, dropout, stochastic_depth),
                MultiHeadAttentionBlock(base_channels, num_heads),
                DownSampleBlock(base_channels, base_channels * 2)
            ]),
            # Level 2: base_channels * 2
            nn.ModuleList([
                ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, dropout, stochastic_depth),
                ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, dropout, stochastic_depth),
                MultiHeadAttentionBlock(base_channels * 2, num_heads),
                DownSampleBlock(base_channels * 2, base_channels * 4)
            ]),
            # Level 3: base_channels * 4
            nn.ModuleList([
                ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, dropout, stochastic_depth),
                ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, dropout, stochastic_depth),
                MultiHeadAttentionBlock(base_channels * 4, num_heads)
            ])
        ])
        
        # 瓶颈层
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, dropout, stochastic_depth),
            MultiHeadAttentionBlock(base_channels * 4, num_heads),
            ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, dropout, stochastic_depth)
        ])
        
        # 解码器
        self.up_blocks = nn.ModuleList([
            # Level 3: base_channels * 4
            nn.ModuleList([
                ResidualBlock(base_channels * 8, base_channels * 4, time_emb_dim, dropout, stochastic_depth),
                ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, dropout, stochastic_depth),
                MultiHeadAttentionBlock(base_channels * 4, num_heads),
                UpSampleBlock(base_channels * 4, base_channels * 2)
            ]),
            # Level 2: base_channels * 2
            nn.ModuleList([
                ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim, dropout, stochastic_depth),
                ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, dropout, stochastic_depth),
                MultiHeadAttentionBlock(base_channels * 2, num_heads),
                UpSampleBlock(base_channels * 2, base_channels)
            ]),
            # Level 1: base_channels
            nn.ModuleList([
                ResidualBlock(base_channels * 2, base_channels, time_emb_dim, dropout, stochastic_depth),
                ResidualBlock(base_channels, base_channels, time_emb_dim, dropout, stochastic_depth),
                MultiHeadAttentionBlock(base_channels, num_heads)
            ])
        ])
        
        # 输出层
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
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
        # 拼接输入
        x = torch.cat([sketch, noisy_color], dim=1)
        x = self.init_conv(x)
        
        # 时间嵌入
        time_emb = self.time_embedder(t)
        time_emb = self.time_mlp(time_emb)
        
        # ==================== 编码器 ====================
        skips = []
        
        # Level 1: 64 channels, 128x128
        res1, res2, attn, down = self.down_blocks[0]
        x = res1(x, time_emb)
        x = res2(x, time_emb)
        x = attn(x)
        skips.append(x)  # 保存 128x128 的特征
        x = down(x)      # -> 64x64
        
        # Level 2: 128 channels, 64x64
        res1, res2, attn, down = self.down_blocks[1]
        x = res1(x, time_emb)
        x = res2(x, time_emb)
        x = attn(x)
        skips.append(x)  # 保存 64x64 的特征
        x = down(x)      # -> 32x32
        
        # Level 3: 256 channels, 32x32
        res1, res2, attn = self.down_blocks[2]
        x = res1(x, time_emb)
        x = res2(x, time_emb)
        x = attn(x)
        skips.append(x)  # 保存 32x32 的特征
        
        # ==================== 瓶颈层 ====================
        res1, attn, res2 = self.mid_blocks
        x = res1(x, time_emb)
        x = attn(x)
        x = res2(x, time_emb)
        
        # ==================== 解码器 ====================
        
        # Level 3: 256 channels, 32x32
        skip3 = skips.pop()
        x = torch.cat([x, skip3], dim=1)  # 512 channels
        res1, res2, attn, up = self.up_blocks[0]
        x = res1(x, time_emb)  # 512 -> 256
        x = res2(x, time_emb)
        x = attn(x)
        x = up(x)  # -> 64x64, 128 channels
        
        # Level 2: 128 channels, 64x64
        skip2 = skips.pop()
        x = torch.cat([x, skip2], dim=1)  # 256 channels
        res1, res2, attn, up = self.up_blocks[1]
        x = res1(x, time_emb)  # 256 -> 128
        x = res2(x, time_emb)
        x = attn(x)
        x = up(x)  # -> 128x128, 64 channels
        
        # Level 1: 64 channels, 128x128
        skip1 = skips.pop()
        x = torch.cat([x, skip1], dim=1)  # 128 channels
        res1, res2, attn = self.up_blocks[2]
        x = res1(x, time_emb)  # 128 -> 64
        x = res2(x, time_emb)
        x = attn(x)
        
        # ==================== 输出 ====================
        velocity = self.final_conv(x)
        
        return velocity


# 向后兼容
SimpleFlowMatchingModel = AdvancedFlowMatchingModel


if __name__ == "__main__":
    """可视化网络结构"""
    import torch
    from torchview import draw_graph
    
    print("="*70)
    print("生成网络结构可视化")
    print("="*70)
    
    # 创建模型
    model = AdvancedFlowMatchingModel(
        base_channels=64,
        time_emb_dim=256,
        num_heads=4,
        dropout=0.1,
        stochastic_depth=0.1
    )
    
    # 准备输入
    sketch = torch.randn(1, 3, 128, 128)
    noisy_color = torch.randn(1, 3, 128, 128)
    t = torch.rand(1, 1)
    
    # 生成可视化图
    model_graph = draw_graph(
        model, 
        input_data=[sketch, noisy_color, t],
        expand_nested=True,  # 展开嵌套模块
        depth=4,  # 显示深度
        device='cpu',
        save_graph=True,  # 保存图形
        filename='flow_matching_architecture',  # 文件名
        directory='./visualizations',  # 保存目录
    )
    
    # 也可以在 Jupyter 中直接显示
    model_graph.visual_graph
    
    print("\n✓ 网络结构图已保存到: ./visualizations/flow_matching_architecture.png")
    print("="*70)

