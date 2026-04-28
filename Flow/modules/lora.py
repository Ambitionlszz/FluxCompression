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
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return LoRAStats(injected_layers=replaced, trainable_params=trainable)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]):
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
