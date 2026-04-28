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
    return transforms.Compose(
        [
            ResizeIfSmall(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


def build_val_transform(image_size: int):
    return transforms.Compose(
        [
            ResizeIfSmall(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
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
    dataset = RecursiveImageDataset(root, transform=None)
    return dataset.samples
