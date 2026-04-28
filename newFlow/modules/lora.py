"""
LoRA (Low-Rank Adaptation) 模块 - 参考 Flow/modules/lora.py

为 Flux Transformer 提供高效的参数微调方法,只训练少量低秩矩阵。

核心思想:
- 冻结预训练模型的原始权重
- 在每个 Linear 层旁添加低秩分解 A * B
- 前向传播: output = W * x + (B * A) * x * (alpha / rank)

优势:
- 大幅减少可训练参数量 (通常 < 1%)
- 保持预训练知识
- 快速适配新任务
"""
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    带 LoRA 的线性层
    
    在原始 Linear 层基础上添加低秩适应模块。
    
    Args:
        base: 原始的 nn.Linear 层
        rank: LoRA 的秩 (通常 4-64)
        alpha: LoRA 缩放系数
        dropout: Dropout 概率 (默认 0.0)
    
    Forward 公式:
        output = base(x) + (lora_B @ lora_A @ dropout(x)) * (alpha / rank)
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA 参数初始化
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

        # A 使用 Kaiming 初始化,B 初始化为零
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # 冻结原始层的参数
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
        
        Returns:
            base_output + lora_output
        """
        # 确保 LoRA 参数与输入具有相同的 dtype
        lora_A = self.lora_A.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        
        base_out = self.base(x)
        lora_out = F.linear(self.dropout(x), lora_A)
        lora_out = F.linear(lora_out, lora_B) * self.scaling
        return base_out + lora_out


@dataclass
class LoRAStats:
    """
    LoRA 注入统计信息
    
    Attributes:
        injected_layers: 被替换为 LoRA 的层数
        trainable_params: 可训练参数总数
    """
    injected_layers: int
    trainable_params: int


def _get_parent_module(root: nn.Module, module_name: str):
    """
    获取模块的父模块
    
    Args:
        root: 根模块
        module_name: 模块名称(可能包含点号分隔的路径)
    
    Returns:
        (parent_module, child_name) 元组
    """
    if "." not in module_name:
        return root, module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    return parent, child_name


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_regex: str,
) -> LoRAStats:
    """
    为模型注入 LoRA
    
    遍历模型的所有 Linear 层,匹配正则表达式的层将被替换为 LoRALinear。
    
    Args:
        model: 目标模型
        rank: LoRA 秩
        alpha: 缩放系数
        dropout: Dropout 概率
        target_regex: 匹配目标层的正则表达式
    
    Returns:
        LoRAStats 统计信息
    
    Example:
        >>> stats = inject_lora(
        ...     model=flux_transformer,
        ...     rank=32,
        ...     alpha=32.0,
        ...     dropout=0.0,
        ...     target_regex=r".*\.attn\..*"  # 只注入注意力层
        ... )
        >>> print(f"Injected {stats.injected_layers} layers")
    """
    pattern = re.compile(target_regex)

    # 首先冻结所有参数
    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not pattern.search(name):
            continue

        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return LoRAStats(injected_layers=replaced, trainable_params=trainable)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    提取 LoRA 参数的状态字典
    
    只保存 lora_A 和 lora_B 参数,不包含冻结的基础权重。
    
    Args:
        model: 包含 LoRA 层的模型
    
    Returns:
        LoRA 参数字典
    """
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]):
    """
    加载 LoRA 参数
    
    Args:
        model: 包含 LoRA 层的模型
        state_dict: LoRA 参数字典
    
    Returns:
        缺失的层名称列表
    """
    missing = []
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        a_key = f"{name}.lora_A"
        b_key = f"{name}.lora_B"
        if a_key not in state_dict or b_key not in state_dict:
            missing.append(name)
            continue
        module.lora_A.data.copy_(state_dict[a_key].to(module.lora_A.device, dtype=module.lora_A.dtype))
        module.lora_B.data.copy_(state_dict[b_key].to(module.lora_B.device, dtype=module.lora_B.dtype))

    return missing
