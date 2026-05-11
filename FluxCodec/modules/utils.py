import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_checkpoint(path: str, state: dict[str, Any]):
    torch.save(state, path)


def write_csv(path: str, rows: list[dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
