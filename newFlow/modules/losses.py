"""
损失函数模块 - 参考 Flow/modules/losses.py

为 newFlow (DiT-IC 架构) 提供训练所需的损失函数,包括:
- BPP (码率损失)
- MSE (像素级重建损失)
- LPIPS (感知损失)
- CLIP L2 (语义一致性损失)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
import lpips


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class CLIPL2Loss(nn.Module):
    """
    CLIP 特征空间的 L2 距离损失
    
    用于衡量重建图像与原始图像在语义层面的一致性。
    使用预训练的 CLIP 模型提取图像特征,计算归一化后的 MSE。
    
    Args:
        clip_path: CLIP 模型路径
    """
    def __init__(self, clip_path: str):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_path)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    def _preprocess(self, x01: torch.Tensor) -> torch.Tensor:
        """
        预处理图像以适配 CLIP 输入要求
        
        - Resize 到 224x224
        - 使用 CLIP 的 mean/std 进行归一化
        
        Args:
            x01: 输入图像 [0, 1] 范围 (B, 3, H, W)
        
        Returns:
            归一化后的图像
        """
        x = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False)
        mean = CLIP_MEAN.to(device=x.device, dtype=x.dtype)
        std = CLIP_STD.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        return x

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        计算 CLIP L2 损失
        
        Args:
            x: 原始图像 (B, 3, H, W)
            x_hat: 重建图像 (B, 3, H, W)
        
        Returns:
            CLIP 特征空间的 MSE 损失
        """
        x_clip = self._preprocess(x)
        x_hat_clip = self._preprocess(x_hat)

        feat_x = self.clip.get_image_features(pixel_values=x_clip)
        feat_hat = self.clip.get_image_features(pixel_values=x_hat_clip)
        feat_x = F.normalize(feat_x, dim=-1)
        feat_hat = F.normalize(feat_hat, dim=-1)
        return F.mse_loss(feat_hat, feat_x)


class Stage1Loss(nn.Module):
    """
    Stage1 训练的综合损失函数
    
    包含四个组成部分:
    1. **BPP (Rate)**: 码率损失,基于熵模型的似然估计
    2. **MSE (Distortion)**: 像素级重建质量
    3. **LPIPS (Perceptual)**: 感知质量,使用 VGG 网络
    4. **CLIP L2 (Semantic)**: 语义一致性,使用 CLIP 模型
    
    总损失 = λ_rate * BPP + d1_mse * MSE + d2_lpips * LPIPS + d3_clip * CLIP_L2
    
    Args:
        clip_path: CLIP 模型路径
        lambda_rate: 码率损失权重 (默认 0.5)
        d1_mse: MSE 损失权重 (默认 2.0)
        d2_lpips: LPIPS 损失权重 (默认 1.0)
        d3_clip: CLIP 损失权重 (默认 0.1)
    
    Example:
        >>> loss_fn = Stage1Loss(clip_path="/path/to/clip")
        >>> losses = loss_fn(x01, x_hat, likelihoods)
        >>> total_loss = losses["loss"]
    """
    def __init__(
        self,
        clip_path: str,
        lambda_rate: float = 0.5,
        d1_mse: float = 2.0,
        d2_lpips: float = 1.0,
        d3_clip: float = 0.1,
    ):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.d1_mse = d1_mse
        self.d2_lpips = d2_lpips
        self.d3_clip = d3_clip

        # LPIPS 感知损失 (VGG backbone)
        self.lpips = lpips.LPIPS(net="vgg").eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

        # CLIP 语义损失
        self.clip_l2 = CLIPL2Loss(clip_path)

    def _bpp_loss(self, likelihoods: dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
        """
        计算 BPP (Bits Per Pixel) 损失
        
        基于熵模型输出的似然概率,计算编码每个像素所需的平均比特数。
        
        Args:
            likelihoods: 熵模型输出的似然字典 {"z": ..., "y": ...}
            num_pixels: 总像素数 (B * H * W)
        
        Returns:
            BPP 损失值
        """
        total = 0.0
        eps = 1e-9
        for v in likelihoods.values():
            total = total + torch.log(v + eps).sum() / (-math.log(2) * num_pixels)
        return total

    def forward(self, x01: torch.Tensor, x_hat01: torch.Tensor, likelihoods: dict[str, torch.Tensor]) -> dict:
        """
        计算综合损失
        
        Args:
            x01: 原始图像 [0, 1] 范围 (B, 3, H, W)
            x_hat01: 重建图像 [0, 1] 范围 (B, 3, H, W)
            likelihoods: 熵模型输出的似然字典
        
        Returns:
            包含以下键的字典:
            - loss: 总损失
            - bpp: BPP 损失
            - mse: MSE 损失
            - lpips: LPIPS 损失
            - clip_l2: CLIP L2 损失
            - psnr: PSNR (Peak Signal-to-Noise Ratio)
        """
        n, _, h, w = x01.shape
        num_pixels = n * h * w

        # 1. 码率损失
        bpp = self._bpp_loss(likelihoods, num_pixels)
        
        # 2. 像素级重建损失
        mse = F.mse_loss(x_hat01, x01)
        
        # 3. 计算 PSNR (Peak Signal-to-Noise Ratio)
        # PSNR = 10 * log10(MAX^2 / MSE)，其中MAX=1（归一化图像）
        if mse > 0:
            psnr = 10 * torch.log10(1.0 / mse)
        else:
            psnr = torch.tensor(float('inf'), device=x01.device)
        
        # 4. 感知损失 (LPIPS 期望输入 [-1, 1])
        x_lpips = x01 * 2.0 - 1.0
        x_hat_lpips = x_hat01 * 2.0 - 1.0
        lpips_loss = self.lpips(x_hat_lpips, x_lpips).mean()
        
        # 5. 语义一致性损失
        clip_l2 = self.clip_l2(x01, x_hat01)

        # 加权求和
        total = self.lambda_rate * bpp + self.d1_mse * mse + self.d2_lpips * lpips_loss + self.d3_clip * clip_l2

        return {
            "loss": total,
            "bpp": bpp.detach(),
            "mse": mse.detach(),
            "lpips": lpips_loss.detach(),
            "clip_l2": clip_l2.detach(),
            "psnr": psnr.detach(),
        }
