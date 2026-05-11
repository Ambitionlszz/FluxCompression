import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
import lpips


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class CLIPL2Loss(nn.Module):
    def __init__(self, clip_path: str):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_path)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    def _preprocess(self, x01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False)
        mean = CLIP_MEAN.to(device=x.device, dtype=x.dtype)
        std = CLIP_STD.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        return x

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        x_clip = self._preprocess(x)
        x_hat_clip = self._preprocess(x_hat)

        feat_x = self.clip.get_image_features(pixel_values=x_clip)
        feat_hat = self.clip.get_image_features(pixel_values=x_hat_clip)
        feat_x = F.normalize(feat_x, dim=-1)
        feat_hat = F.normalize(feat_hat, dim=-1)
        return F.mse_loss(feat_hat, feat_x)


class Stage1Loss(nn.Module):
    def __init__(
        self,
        clip_path: str,
        lambda_rate: float = 0.5,
        d1_mse: float = 2.0,
        d2_lpips: float = 1.0,
        d3_clip: float = 0.1,
    ):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.d1_mse = d1_mse
        self.d2_lpips = d2_lpips
        self.d3_clip = d3_clip

        self.lpips = lpips.LPIPS(net="vgg").eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

        self.clip_l2 = CLIPL2Loss(clip_path)

    def _bpp_loss(self, likelihoods: dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
        total = 0.0
        eps = 1e-9
        for v in likelihoods.values():
            total = total + torch.log(v + eps).sum() / (-math.log(2) * num_pixels)
        return total

    def forward(self, x01: torch.Tensor, x_hat01: torch.Tensor, likelihoods: dict[str, torch.Tensor]) -> dict:
        n, _, h, w = x01.shape
        num_pixels = n * h * w

        bpp = self._bpp_loss(likelihoods, num_pixels)
        mse = F.mse_loss(x_hat01, x01)
        x_lpips = x01 * 2.0 - 1.0
        x_hat_lpips = x_hat01 * 2.0 - 1.0
        lpips_loss = self.lpips(x_hat_lpips, x_lpips).mean()
        clip_l2 = self.clip_l2(x01, x_hat01)

        total = self.lambda_rate * bpp + self.d1_mse * mse + self.d2_lpips * lpips_loss + self.d3_clip * clip_l2

        return {
            "loss": total,
            "bpp": bpp.detach(),
            "mse": mse.detach(),
            "lpips": lpips_loss.detach(),
            "clip_l2": clip_l2.detach(),
        }
