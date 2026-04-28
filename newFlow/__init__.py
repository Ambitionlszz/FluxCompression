"""
newFlow 包

基于 DiT-IC 架构的 Flux2 图像压缩模型。

核心组件:
- FlowCompression: 主模型类,整合 Flux AE + DiT-IC Codec + Flux Transformer
- ELICAuxEncoder: ELIC 辅助编码器,生成 z_aux
"""

from .model import FlowCompression
from .elic_aux_encoder import ELICAuxEncoder
from .utils import load_elic_encoder

__all__ = [
    "FlowCompression",
    "ELICAuxEncoder",
    "load_elic_encoder",
]
