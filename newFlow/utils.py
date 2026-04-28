"""
newFlow 工具函数

提供模型加载、ELIC 编码器初始化等通用功能。
"""
import torch
from typing import Tuple
from .elic_aux_encoder import ELICAuxEncoder


def load_elic_encoder(
    elic_ckpt_path: str,
    device: torch.device,
    N: int = 192,
    M: int = 320,
) -> ELICAuxEncoder:
    """
    加载 ELIC 辅助编码器
    
    从 checkpoint 中提取 g_a (Analysis Transform) 权重并加载到 ELICAuxEncoder。
    
    Args:
        elic_ckpt_path: ELIC checkpoint 文件路径
        device: 计算设备
        N: ELIC 中间层通道数 (默认 192)
        M: ELIC 输出通道数 (默认 320)
    
    Returns:
        加载好权重的 ELICAuxEncoder (eval 模式,参数冻结)
    
    Example:
        >>> elic = load_elic_encoder("/path/to/elic.pth", device)
        >>> z_aux = elic(x01)  # x01: [0, 1] 范围
    """
    print(f"Loading ELIC from {elic_ckpt_path}...")
    elic_model = ELICAuxEncoder(N=N, M=M)
    checkpoint = torch.load(elic_ckpt_path, map_location="cpu")
    
    # 处理 checkpoint 格式（可能包含 'state_dict' 键）
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 只加载 g_a 相关的权重（过滤掉 g_s、h_a、h_s、熵模型等不需要的部分）
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('g_a.'):
            # 移除 'g_a.' 前缀，因为我们的模型直接就是 AnalysisTransformEX
            new_key = key[4:]  # 去掉 'g_a.'
            filtered_state_dict[new_key] = value
    
    # 加载过滤后的权重
    missing_keys, unexpected_keys = elic_model.g_a.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in ELIC g_a: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in ELIC g_a: {unexpected_keys}")
    
    elic_model = elic_model.to(device)
    elic_model.eval()  # 设置为评估模式
    
    # 冻结 ELIC 参数
    for param in elic_model.parameters():
        param.requires_grad = False
    
    print(f"ELIC loaded successfully. Output channels: {M}")
    return elic_model
