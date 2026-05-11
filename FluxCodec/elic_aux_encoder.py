"""
ELIC 辅助编码器加载模块。
ELIC g_a: 像素域 → 16x 下采样的特征 (B, 320, H/16, W/16)
"""
import sys
import os
import torch
import torch.nn as nn


def load_elic_encoder(elic_ckpt: str, device: torch.device = torch.device("cpu")) -> nn.Module:
    """
    加载 ELIC 模型并提取 g_a (Analysis Transform) 作为辅助编码器。
    
    ELIC g_a 将 RGB 图像 [0,1] 下采样 16x 到 (B, 320, H/16, W/16)。
    
    自动从 elic_ckpt 路径推导 ELIC 代码目录:
      /data2/.../DiT-IC/checkpoints/elic_official.pth
                 ↑ DiT-IC 目录，内含 ELIC/ 子目录
    """
    # 从 checkpoint 路径推导 DiT-IC 目录
    ckpt_abs = os.path.abspath(elic_ckpt)
    dit_ic_dir = os.path.dirname(os.path.dirname(ckpt_abs))  # checkpoints → DiT-IC
    if dit_ic_dir not in sys.path:
        sys.path.insert(0, dit_ic_dir)

    from ELIC.elic_official import ELIC

    model = ELIC()
    if os.path.exists(elic_ckpt):
        checkpoint = torch.load(elic_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint)
        print(f"[ELIC] Loaded weights from: {elic_ckpt}")
    else:
        print(f"[ELIC] WARNING: Checkpoint not found at {elic_ckpt}, using random init")

    aux_encoder = model.g_a
    aux_encoder.eval()
    aux_encoder.requires_grad_(False)
    aux_encoder.to(device)
    return aux_encoder

