"""
FluxCodec Stage2 loss.

Generator loss:
    lambda_rate * bpp
  + lambda_l2 * MSE
  + lambda_lpips * LPIPS
  + lambda_dists * DISTS
  + lambda_gan * GAN
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

from .losses import DISTSMetric


class Stage2Loss(nn.Module):
    def __init__(
        self,
        lambda_rate: float = 0.5,
        lambda_l2: float = 2.0,
        lambda_lpips: float = 1.0,
        lambda_dists: float = 1.0,
        lambda_gan: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.lambda_rate = lambda_rate
        self.lambda_l2 = lambda_l2
        self.lambda_lpips = lambda_lpips
        self.lambda_dists = lambda_dists
        self.lambda_gan = lambda_gan

        self.lpips_net = lpips.LPIPS(net="vgg").eval()
        for p in self.lpips_net.parameters():
            p.requires_grad = False

        self.dists = DISTSMetric()

    def _bpp_loss(self, likelihoods: dict[str, torch.Tensor], num_pixels: int) -> torch.Tensor:
        total = 0.0
        eps = 1e-9
        for v in likelihoods.values():
            total = total + torch.log(v + eps).sum() / (-math.log(2) * num_pixels)
        return total

    def forward(
        self,
        x01: torch.Tensor,
        x_hat01: torch.Tensor,
        likelihoods: dict[str, torch.Tensor],
        loss_adv: torch.Tensor = None,
    ) -> dict:
        n, _, h, w = x01.shape
        num_pixels = n * h * w

        bpp = self._bpp_loss(likelihoods, num_pixels)
        mse = F.mse_loss(x_hat01, x01)
        psnr = -10 * torch.log10(torch.clamp(mse, min=1e-8))

        x_lp = x01 * 2.0 - 1.0
        xh_lp = x_hat01 * 2.0 - 1.0
        lpips_loss = self.lpips_net(xh_lp, x_lp).mean()
        dists_loss = self.dists(x01, x_hat01)

        loss_D = (
            self.lambda_l2 * mse
            + self.lambda_lpips * lpips_loss
            + self.lambda_dists * dists_loss
        )
        if loss_adv is not None:
            loss_D = loss_D + self.lambda_gan * loss_adv

        total = self.lambda_rate * bpp + loss_D

        return {
            "loss": total,
            "bpp": bpp.detach(),
            "mse": mse.detach(),
            "psnr": psnr.detach(),
            "lpips": lpips_loss.detach(),
            "dists": dists_loss.detach(),
        }
