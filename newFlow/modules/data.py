"""
数据加载模块 - 参考 Flow/modules/data.py 并根据 newFlow 需求适配

支持递归扫描图像目录、随机裁剪/中心裁剪、数据增强等功能。
"""
import os
import random
from pathlib import Path
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class RecursiveImageDataset(Dataset):
    """
    递归图像数据集
    
    自动扫描根目录及其子目录下的所有图像文件。
    
    Args:
        root: 数据集根目录路径
        transform: 可选的图像变换
    """
    def __init__(self, root: str, transform=None):
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        self.samples: list[str] = []
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            for name in filenames:
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append(str(Path(dirpath) / name))
        self.samples.sort()
        if not self.samples:
            raise RuntimeError(f"No images found under: {root}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.samples[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class ResizeIfSmall:
    """
    如果图像尺寸小于指定大小,则放大到该大小
    
    Args:
        min_size: 最小边长
    """
    def __init__(self, min_size: int):
        self.min_size = min_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < self.min_size or h < self.min_size:
            new_w = max(w, self.min_size)
            new_h = max(h, self.min_size)
            return transforms.Resize((new_h, new_w))(img)
        return img


def build_train_transform(image_size: int):
    """
    构建训练数据变换
    
    包含:
    - ResizeIfSmall: 确保图像不小于指定尺寸
    - RandomCrop: 随机裁剪到目标尺寸(数据增强)
    - RandomHorizontalFlip: 随机水平翻转(数据增强)
    - ToTensor: 转换为 [0, 1] 范围的 Tensor
    
    Args:
        image_size: 目标图像尺寸(正方形)
    
    Returns:
        transforms.Compose 对象
    """
    return transforms.Compose(
        [
            ResizeIfSmall(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def build_val_transform(image_size: int):
    """
    构建验证/测试数据变换
    
    包含:
    - ResizeIfSmall: 确保图像不小于指定尺寸
    - CenterCrop: 中心裁剪到目标尺寸(确定性)
    - ToTensor: 转换为 [0, 1] 范围的 Tensor
    
    Args:
        image_size: 目标图像尺寸(正方形)
    
    Returns:
        transforms.Compose 对象
    """
    return transforms.Compose(
        [
            ResizeIfSmall(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def seed_worker(worker_id: int):
    """
    DataLoader worker 的随机种子初始化函数
    
    确保每个 worker 有独立但可复现的随机状态。
    
    Args:
        worker_id: worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """
    构建 DataLoader
    
    Args:
        dataset: PyTorch Dataset
        batch_size: 批次大小
        num_workers: 数据加载 worker 数量
        shuffle: 是否打乱数据
        drop_last: 是否丢弃最后一个不完整的批次
    
    Returns:
        DataLoader 对象
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
    )


def list_images(root: str) -> list[str]:
    """
    列出目录下所有图像文件路径
    
    Args:
        root: 根目录路径
    
    Returns:
        图像文件路径列表
    """
    dataset = RecursiveImageDataset(root, transform=None)
    return dataset.samples
