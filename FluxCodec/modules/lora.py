import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling
        return base_out + lora_out


class LoRAConv2d(nn.Module):
    """LoRA wrapper for nn.Conv2d using 1x1 conv decomposition in channel space."""

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Use nn.Conv2d instead of raw Parameter so autocast handles dtype casting
        self.lora_A = nn.Conv2d(base.in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    @property
    def lora_A_weight(self) -> torch.Tensor:
        return self.lora_A.weight

    @property
    def lora_B_weight(self) -> torch.Tensor:
        return self.lora_B.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_A(self.dropout(x))
        lora_out = self.lora_B(lora_out) * self.scaling
        return base_out + lora_out


@dataclass
class LoRAStats:
    injected_layers: int
    trainable_params: int


def _get_parent_module(root: nn.Module, module_name: str):
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
    pattern = re.compile(target_regex)

    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not pattern.search(name):
            continue

        parent, child_name = _get_parent_module(model, name)
        lora_mod = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        lora_mod = lora_mod.to(device=module.weight.device)
        setattr(parent, child_name, lora_mod)
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return LoRAStats(injected_layers=replaced, trainable_params=trainable)


def inject_lora_conv(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_regex: str,
) -> LoRAStats:
    """Inject LoRA into Conv2d layers of a model (e.g., AE decoder)."""
    pattern = re.compile(target_regex)

    for p in model.parameters():
        p.requires_grad = False

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Conv2d):
            continue
        if not pattern.search(name):
            continue

        parent, child_name = _get_parent_module(model, name)
        lora_mod = LoRAConv2d(module, rank=rank, alpha=alpha, dropout=dropout)
        lora_mod = lora_mod.to(device=module.weight.device)
        setattr(parent, child_name, lora_mod)
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return LoRAStats(injected_layers=replaced, trainable_params=trainable)


def _lora_a(tensor_or_module):
    """Return the lora_A tensor, handling both nn.Parameter and nn.Conv2d."""
    if isinstance(tensor_or_module, nn.Parameter):
        return tensor_or_module
    return tensor_or_module.weight


def _lora_b(tensor_or_module):
    if isinstance(tensor_or_module, nn.Parameter):
        return tensor_or_module
    return tensor_or_module.weight


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            state[f"{name}.lora_A"] = _lora_a(module.lora_A).detach().cpu()
            state[f"{name}.lora_B"] = _lora_b(module.lora_B).detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]):
    missing = []
    for name, module in model.named_modules():
        if not isinstance(module, (LoRALinear, LoRAConv2d)):
            continue
        a_key = f"{name}.lora_A"
        b_key = f"{name}.lora_B"
        if a_key not in state_dict or b_key not in state_dict:
            missing.append(name)
            continue
        a_tensor = _lora_a(module.lora_A)
        b_tensor = _lora_b(module.lora_B)
        a_tensor.data.copy_(state_dict[a_key].to(a_tensor.device, dtype=a_tensor.dtype))
        b_tensor.data.copy_(state_dict[b_key].to(b_tensor.device, dtype=b_tensor.dtype))

    return missing
