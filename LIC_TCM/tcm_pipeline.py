"""FLUX.2 VAE + TCM Latent Compression Pipeline.

Connects a frozen FLUX.2 VAE (encoder/decoder) with a trainable TCMLatent
model for learned image compression in latent space.

Pipeline: input image [0,1] -> VAE encode -> latent -> TCM -> latent_hat -> VAE decode -> output image
"""

import os
import sys

import torch
import torch.nn as nn
from torch import Tensor

# Add FLUX.2 source to path
FLUX2_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if FLUX2_SRC_DIR not in sys.path:
    sys.path.insert(0, FLUX2_SRC_DIR)

from flux2.autoencoder import AutoEncoder, AutoEncoderParams
from flux2.util import load_ae


class LatentCompressionPipeline(nn.Module):
    """Pipeline connecting frozen FLUX.2 VAE with trainable TCMLatent.

    Args:
        tcm: TCMLatent model instance (trainable)
        vae: Pre-loaded AutoEncoder, or None to load via model_name
        vae_model_name: FLUX.2 variant name for auto-loading VAE
        vae_checkpoint: Path to VAE weights (None = auto download from HF)
        vae_device: Device for VAE
        freeze_vae: Whether to freeze VAE weights (default True)
    """

    def __init__(
        self,
        tcm: nn.Module,
        vae: AutoEncoder = None,
        vae_model_name: str = "flux.2-klein-4b",
        vae_checkpoint: str = None,
        vae_device: str = "cuda",
        freeze_vae: bool = True,
    ):
        super().__init__()
        self.tcm = tcm

        # Load VAE if not provided
        if vae is not None:
            self.vae = vae
        else:
            if vae_checkpoint is not None:
                os.environ["AE_MODEL_PATH"] = vae_checkpoint
            self.vae = load_ae(vae_model_name, device=vae_device)

        # Detect VAE dtype
        self.vae_dtype = next(self.vae.parameters()).dtype

        # Freeze VAE
        if freeze_vae:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        """Override train to keep VAE in eval mode."""
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def encode_to_latent(self, x: Tensor) -> Tensor:
        """Encode RGB image [0,1] to VAE latent space.

        Args:
            x: (B, 3, H, W) in [0, 1]

        Returns:
            latent: (B, 128, H/16, W/16) float32
        """
        z = self.vae.encode(x.to(self.vae_dtype))
        return z.float()

    def decode_from_latent(self, z: Tensor, no_grad: bool = False) -> Tensor:
        """Decode VAE latent back to RGB image.

        Args:
            z: (B, 128, H/16, W/16)
            no_grad: If True, run without gradient (faster, less memory).
                     If False, allow gradient flow for pixel-domain loss.

        Returns:
            image: (B, 3, H, W) clamped to [0, 1]
        """
        z_input = z.to(self.vae_dtype)
        if no_grad:
            with torch.no_grad():
                x = self.vae.decode(z_input)
        else:
            x = self.vae.decode(z_input)
        return x.float().clamp(0, 1)

    def forward(self, x: Tensor, decode_pixel: bool = True) -> dict:
        """Full pipeline forward pass.

        Args:
            x: Input image (B, 3, H, W) in [0, 1]
            decode_pixel: Whether to decode to pixel domain (needed for pixel loss)

        Returns:
            dict with keys:
                x_hat: Reconstructed image (B, 3, H, W) if decode_pixel else None
                latent: Original VAE latent
                latent_hat: Reconstructed latent from TCM
                likelihoods: From TCM entropy model
                para: From TCM (means, scales, y)
        """
        # Encode to latent (no grad, detached from VAE)
        latent = self.encode_to_latent(x)

        # TCM compression in latent space
        tcm_out = self.tcm(latent)
        latent_hat = tcm_out["x_hat"]

        result = {
            "latent": latent,
            "latent_hat": latent_hat,
            "likelihoods": tcm_out["likelihoods"],
            "para": tcm_out["para"],
        }

        # Decode to pixel domain (with gradient flow for pixel-domain loss)
        if decode_pixel:
            result["x_hat"] = self.decode_from_latent(latent_hat, no_grad=False)
        else:
            result["x_hat"] = None

        return result
