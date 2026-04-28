"""
工具函数模块 - 参考 Flow/modules/utils.py

提供训练和推理过程中常用的辅助函数:
- 随机种子设置
- 平均值统计 (AverageMeter)
- 目录创建、JSON/CSV 保存等
"""
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int):
    """
    设置全局随机种子,确保实验可复现
    
    影响:
    - Python random
    - NumPy random
    - PyTorch CPU random
    - PyTorch CUDA random (所有 GPU)
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """
    平均值统计器
    
    用于在训练过程中跟踪损失、指标等的移动平均值。
    
    Example:
        >>> meter = AverageMeter()
        >>> for loss in losses:
        ...     meter.update(loss.item(), batch_size)
        >>> print(f"Average loss: {meter.avg:.4f}")
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计器"""
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        """
        更新统计值
        
        Args:
            value: 新的观测值
            n: 该值对应的样本数量 (默认 1)
        """
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        """
        获取当前平均值
        
        Returns:
            平均值,如果 count 为 0 则返回 0.0
        """
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def ensure_dir(path: str):
    """
    确保目录存在,如果不存在则创建
    
    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict[str, Any]):
    """
    保存对象为 JSON 文件
    
    Args:
        path: 输出文件路径
        obj: 要保存的对象(字典)
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_checkpoint(path: str, state: dict[str, Any]):
    """
    保存模型检查点
    
    Args:
        path: 检查点文件路径
        state: 包含模型状态、优化器状态等的字典
    """
    torch.save(state, path)


def write_csv(path: str, rows: list[dict[str, Any]]):
    """
    将数据写入 CSV 文件
    
    Args:
        path: CSV 文件路径
        rows: 字典列表,每个字典代表一行数据
    """
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
