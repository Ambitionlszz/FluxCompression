"""
ELIC 辅助编码器 - 用于生成 z_aux

从 DiT-IC 项目移植的 ELIC g_a (Analysis Transform) 实现。
用于将图像编码为 320 通道的辅助 latent 特征。

参考: /data2/luosheng/code/DiT-IC/ELIC/elic_official.py
"""

import torch
import torch.nn as nn
from compressai.layers import AttentionBlock, conv3x3


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    """5x5 convolution with padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


class ResidualBottleneck(nn.Module):
    """残差瓶颈块"""
    def __init__(self, N=192, act=nn.ReLU):
        super().__init__()
        self.branch = nn.Sequential(
            conv1x1(N, N // 2),
            act(),
            nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1),
            act(),
            conv1x1(N // 2, N)
        )

    def forward(self, x):
        out = x + self.branch(x)
        return out


class AnalysisTransformEX(nn.Module):
    """
    ELIC 分析变换网络 (g_a)
    
    将 RGB 图像编码为高维 latent 特征
    
    Args:
        N: 中间层通道数 (default: 192)
        M: 输出通道数 (default: 320)
    
    Input:
        x: RGB 图像 (B, 3, H, W), 范围 [0, 1]
    
    Output:
        latent: (B, M, H/16, W/16)
    """
    def __init__(self, N=192, M=320, act=nn.ReLU):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            # Stage 1: 初始特征提取
            conv(3, N),                          # (B, N, H/2, W/2)
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            
            # Stage 2: 进一步下采样和特征提取
            conv(N, N),                          # (B, N, H/4, W/4)
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            AttentionBlock(N),
            
            # Stage 3: 最终压缩到 M 通道
            conv(N, N),                          # (B, N, H/8, W/8)
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            ResidualBottleneck(N, act=act),
            conv(N, M),                          # (B, M, H/16, W/16)
            AttentionBlock(M),
        )

    def forward(self, x):
        """
        Args:
            x: RGB 图像 (B, 3, H, W), 范围 [0, 1]
        
        Returns:
            latent: (B, M, H/16, W/16)
        """
        x = self.analysis_transform(x)
        return x


class ELICAuxEncoder(nn.Module):
    """
    ELIC 辅助编码器包装类
    
    提供简化的接口用于生成 z_aux
    
    Example:
        >>> elic = ELICAuxEncoder()
        >>> elic.load_state_dict(torch.load("elic_official.pth"))
        >>> z_aux = elic(x01)  # x01: [0, 1] 范围的图像
    """
    def __init__(self, N=192, M=320):
        super().__init__()
        self.g_a = AnalysisTransformEX(N=N, M=M, act=nn.ReLU)
    
    def forward(self, x):
        """
        Args:
            x: RGB 图像 (B, 3, H, W), 范围 [0, 1]
        
        Returns:
            z_aux: (B, 320, H/16, W/16)
        """
        return self.g_a(x)


def load_elic_encoder(elic_ckpt_path: str, device: torch.device, N: int = 192, M: int = 320) -> ELICAuxEncoder:
    """
    加载预训练的 ELIC 辅助编码器
    
    Args:
        elic_ckpt_path: ELIC checkpoint 路径
        device: 运行设备
        N: 中间层通道数
        M: 输出通道数
    
    Returns:
        加载好权重的 ELICAuxEncoder 模型
    """
    model = ELICAuxEncoder(N=N, M=M)
    
    # 加载权重
    checkpoint = torch.load(elic_ckpt_path, map_location="cpu")
    
    # 处理不同的 checkpoint 格式
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    
    # 移除可能的前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 'g_a.' 前缀（如果存在）
        if k.startswith("g_a."):
            new_key = k
        elif k.startswith("analysis_transform."):
            new_key = "g_a." + k
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # 加载状态字典
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model
