"""
newFlow 模块包

提供训练和评估所需的核心组件:
- data: 数据加载和预处理
- losses: 损失函数
- evaluators: 评估器
- lora: LoRA 微调
- utils: 工具函数
"""

from .data import (
    RecursiveImageDataset,
    ResizeIfSmall,
    build_train_transform,
    build_val_transform,
    build_dataloader,
    list_images,
)

from .losses import Stage1Loss, CLIPL2Loss

from .evaluators import Stage1Evaluator

from .lora import inject_lora, lora_state_dict, load_lora_state_dict, LoRALinear, LoRAStats

from .utils import (
    AverageMeter,
    set_global_seed,
    ensure_dir,
    save_json,
    save_checkpoint,
    write_csv,
)

__all__ = [
    # Data
    "RecursiveImageDataset",
    "ResizeIfSmall",
    "build_train_transform",
    "build_val_transform",
    "build_dataloader",
    "list_images",
    
    # Losses
    "Stage1Loss",
    "CLIPL2Loss",
    
    # Evaluators
    "Stage1Evaluator",
    
    # LoRA
    "inject_lora",
    "lora_state_dict",
    "load_lora_state_dict",
    "LoRALinear",
    "LoRAStats",
    
    # Utils
    "AverageMeter",
    "set_global_seed",
    "ensure_dir",
    "save_json",
    "save_checkpoint",
    "write_csv",
]
