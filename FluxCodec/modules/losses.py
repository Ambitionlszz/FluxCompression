import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import pyiqa


class DISTSMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric = pyiqa.create_metric("dists")
        self.metric.eval()
        for p in self.metric.parameters():
            p.requires_grad = False

    def forward(self, x01: torch.Tensor, x_hat01: torch.Tensor) -> torch.Tensor:
        return self.metric(x_hat01.clamp(0, 1), x01.clamp(0, 1)).mean()


class Stage1Loss(nn.Module):
    def __init__(
        self,
        lambda_rate: float = 0.5,
        d1_mse: float = 2.0,
        d2_lpips: float = 1.0,
        d3_dists: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.d1_mse = d1_mse
        self.d2_lpips = d2_lpips
        self.d3_dists = d3_dists

        self.lpips = lpips.LPIPS(net="vgg").eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

        self.dists = DISTSMetric()

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
        psnr = -10 * torch.log10(torch.clamp(mse, min=1e-8))
        x_lpips = x01 * 2.0 - 1.0
        x_hat_lpips = x_hat01 * 2.0 - 1.0
        lpips_loss = self.lpips(x_hat_lpips, x_lpips).mean()
        dists_loss = self.dists(x01, x_hat01)

        total = self.lambda_rate * bpp + self.d1_mse * mse + self.d2_lpips * lpips_loss + self.d3_dists * dists_loss

        return {
            "loss": total,
            "bpp": bpp.detach(),
            "mse": mse.detach(),
            "psnr": psnr.detach(),
            "lpips": lpips_loss.detach(),
            "dists": dists_loss.detach(),
        }
