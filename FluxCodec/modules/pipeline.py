"""
FluxCodec Pipeline - Based on Flow/modules/pipeline.py.
Replaced TCM with LatentCodec, adding ELIC auxiliary encoder.
"""
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flux2.sampling import (
    batched_prc_img, batched_prc_txt, denoise, get_schedule, scatter_ids,
)
from flux2.text_encoder import Qwen3Embedder
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

from .latent_codec import LatentCodec

FIXED_PROMPT = "Lossless quality, artifact-free, preserved textures, clean details."


def apply_color_fix(x_hat: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
    """
    Applies the color fix strategy from StableCodec.
    x_hat, x_orig: [B, C, H, W] tensors in [0, 1].
    """
    # Calculate mean and std per channel per image
    mu_x = x_orig.mean(dim=[-2, -1], keepdim=True)
    sigma_x = x_orig.std(dim=[-2, -1], keepdim=True, unbiased=False)
    
    # 16-bit quantization of mu_x and sigma_x
    max_val = (1 << 16) - 1
    mu_x_q = torch.floor(mu_x * max_val + 0.5) / max_val
    sigma_x_q = torch.floor(sigma_x * max_val + 0.5) / max_val
    
    # Calculate mean and std of the reconstructed image
    mu_x_hat = x_hat.mean(dim=[-2, -1], keepdim=True)
    sigma_x_hat = x_hat.std(dim=[-2, -1], keepdim=True, unbiased=False)
    
    # Avoid division by zero
    sigma_x_hat = torch.clamp(sigma_x_hat, min=1e-6)
    
    # Apply AdaIN
    x_hat_c = (x_hat - mu_x_hat) / sigma_x_hat * sigma_x_q + mu_x_q
    
    return x_hat_c.clamp(0.0, 1.0)


class FluxCodecPipeline(nn.Module):
    def __init__(
        self,
        model_name: str,
        flux_ckpt: str,
        ae_ckpt: str,
        qwen_ckpt: Optional[str],
        codec: LatentCodec,
        elic_aux_encoder: nn.Module,
        device: torch.device,
        guidance: float = 1.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        model_info = FLUX2_MODEL_INFO[model_name]
        os.environ[model_info["model_path"]] = flux_ckpt
        os.environ["AE_MODEL_PATH"] = ae_ckpt

        self.model_name = model_name
        self.guidance = guidance
        self.guidance_distilled = bool(model_info.get("guidance_distilled", True))

        # LatentCodec (Trainable)
        self.codec = codec
 
        # ELIC auxiliary encoder (Frozen)
        self.elic_aux_encoder = elic_aux_encoder
        if self.elic_aux_encoder is not None:
            self.elic_aux_encoder.eval()
            for p in self.elic_aux_encoder.parameters():
                p.requires_grad = False
 
        # Load Flux and AE
        print(f"Loading {flux_ckpt} for the FLUX.2 weights")
        self.flux = load_flow_model(model_name, device=device)
 
        print(f"Loading {ae_ckpt} for the AutoEncoder weights")
        self.ae = load_ae(model_name, device=device)
 
        # Load text encoder
        if qwen_ckpt:
            print(f"Loading text encoder from local path: {qwen_ckpt}")
            self.text_encoder = Qwen3Embedder(model_spec=qwen_ckpt, device=device)
        else:
            default_qwen_path = "/data2/luosheng/hf_models/hub/Qwen3-4B-FP8"
            if os.path.exists(default_qwen_path):
                print(f"Loading text encoder from default local path: {default_qwen_path}")
                self.text_encoder = Qwen3Embedder(model_spec=default_qwen_path, device=device)
            else:
                print("Warning: No local Qwen model found, trying to load from HuggingFace...")
                self.text_encoder = load_text_encoder(model_name, device=device)

        # Freeze fixed components
        self.ae.eval()
        self.text_encoder.eval()
        self.flux.eval()
        for p in self.ae.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.flux.parameters():
            p.requires_grad = False

        if use_gradient_checkpointing:
            if hasattr(self.flux, 'gradient_checkpointing_enable'):
                print("✓ Enabling gradient checkpointing for FLUX model")
                self.flux.gradient_checkpointing_enable()
            else:
                print("Warning: FLUX model does not support gradient checkpointing")

        self._prompt_cache = {}

    # ==================== Utility Methods ====================
 
    @torch.no_grad()
    def encode_images(self, x01: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Encode [0,1] image to latent with padding alignment."""
        h, w = x01.shape[-2], x01.shape[-1]
        # FLUX AE requires image dimensions aligned to its 16x latent stride.
        # The codec applies its own minimal latent-space padding for hyperprior
        # shape closure, avoiding unnecessary image-space padding during eval.
        alignment = 16
        pad_h = (alignment - h % alignment) % alignment
        pad_w = (alignment - w % alignment) % alignment

        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x01, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_padded = x01

        x = x_padded * 2.0 - 1.0
        z = self.ae.encode(x.to(next(self.ae.parameters()).dtype))
        pad_info = {"original_shape": x01.shape, "pad_h": pad_h, "pad_w": pad_w}
        return z.float(), pad_info

    def decode_latents(self, z: torch.Tensor, pad_info: dict = None) -> torch.Tensor:
        """Decode latent to [0,1] image, removing padding if provided."""
        x = self.ae.decode(z.to(next(self.ae.parameters()).dtype)).float()
        x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        if pad_info is not None:
            pad_h, pad_w = pad_info["pad_h"], pad_info["pad_w"]
            if pad_h > 0 or pad_w > 0:
                orig_h = pad_info["original_shape"][-2]
                orig_w = pad_info["original_shape"][-1]
                x = x[:, :, :orig_h, :orig_w]
        return x

    @torch.no_grad()
    def get_elic_features(self, x01: torch.Tensor, pad_info: dict) -> torch.Tensor:
        """Get ELIC auxiliary features. Spatially aligned with FLUX AE (16x downsample)."""
        if self.elic_aux_encoder is None:
            # No ELIC encoder: return zero tensor
            h, w = x01.shape[-2], x01.shape[-1]
            ph, pw = pad_info["pad_h"], pad_info["pad_w"]
            h_lat = (h + ph) // 16
            w_lat = (w + pw) // 16
            return torch.zeros(x01.shape[0], 320, h_lat, w_lat,
                               device=x01.device, dtype=x01.dtype)

        # ELIC expects [0,1] input with identical padding
        ph, pw = pad_info["pad_h"], pad_info["pad_w"]
        if ph > 0 or pw > 0:
            x01_padded = F.pad(x01, (0, pw, 0, ph), mode='reflect')
        else:
            x01_padded = x01

        return self.elic_aux_encoder(x01_padded).detach()

    @torch.no_grad()
    def get_text_context(self, batch_size: int, device: torch.device):
        if batch_size in self._prompt_cache:
            ctx, ctx_ids = self._prompt_cache[batch_size]
            if ctx.device != device:
                ctx = ctx.to(device)
                ctx_ids = ctx_ids.to(device)
                self._prompt_cache[batch_size] = (ctx, ctx_ids)
            return ctx, ctx_ids

        prompts = [FIXED_PROMPT] * batch_size
        if self.guidance_distilled:
            ctx = self.text_encoder(prompts).to(torch.bfloat16)
        else:
            ctx_empty = self.text_encoder([""] * batch_size).to(torch.bfloat16)
            ctx_prompt = self.text_encoder(prompts).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)

        ctx, ctx_ids = batched_prc_txt(ctx)
        self._prompt_cache[batch_size] = (ctx.to(device), ctx_ids.to(device))
        return ctx, ctx_ids

    def _latent_to_tokens(self, z: torch.Tensor):
        return batched_prc_img(z.to(torch.bfloat16))

    def _tokens_to_latent(self, x_tokens: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(scatter_ids(x_tokens, x_ids)).squeeze(2)

    # ==================== Training Interface ====================
 
    def forward_stage1_train(self, x01: torch.Tensor, train_schedule_steps: int) -> dict:
        """
        Training forward pass:
        AE encode -> ELIC -> LatentCodec -> FLUX single-step denoise -> AE decode
        """
        batch_size = x01.shape[0]
        device = x01.device
 
        # 1. AE Encode
        z_clean, pad_info = self.encode_images(x01)
 
        # 2. ELIC Auxiliary Features
        z_aux = self.get_elic_features(x01, pad_info)
 
        # 3. LatentCodec Compression
        codec_out = self.codec(z_clean, z_aux)
        z_tcm = codec_out["x_hat"].to(z_clean.dtype)
        res = codec_out.get("res", None)

        # 4. FLUX Denoise Preparation
        z_tokens, z_ids = self._latent_to_tokens(z_tcm)
 
        schedule = get_schedule(train_schedule_steps, z_tokens.shape[1])
        schedule_tensor = torch.tensor(schedule, dtype=z_tokens.dtype, device=device)
        step_idx = torch.randint(0, train_schedule_steps, (batch_size,), device=device)
        timesteps = schedule_tensor[step_idx]
 
        ctx, ctx_ids = self.get_text_context(batch_size=batch_size, device=device)
        guidance_vec = torch.full(
            (z_tokens.shape[0],), self.guidance, dtype=z_tokens.dtype, device=device)
 
        # 5. FLUX Predict Velocity
        v_pred_tokens = self.flux(
            x=z_tokens, x_ids=z_ids,
            timesteps=timesteps,
            ctx=ctx, ctx_ids=ctx_ids,
            guidance=guidance_vec,
        )
 
        # 6. Flow Matching Reconstruction: z_hat = z_tcm - t * v_pred
        z_hat_tokens = z_tokens - timesteps.view(-1, 1, 1) * v_pred_tokens
        z_hat = self._tokens_to_latent(z_hat_tokens, z_ids)
 
        # 7. Auxiliary Residual Addition
        if res is not None:
            z_hat = z_hat + res.to(z_hat.dtype)
 
        # 8. AE Decode
        x_hat01 = self.decode_latents(z_hat, pad_info)

        return {
            "x_hat": x_hat01,
            "likelihoods": codec_out["likelihoods"],
            "z_clean": z_clean,
            "z_tcm": z_tcm,
            "sigma": timesteps,
        }

    # ==================== Inference Interface ====================
 
    @torch.no_grad()
    def forward_stage1_infer(
        self,
        x01: torch.Tensor,
        infer_steps: int = 4,
        do_entropy_coding: bool = True,
        color_fix: bool = False,
    ) -> dict:
        """Inference: AE encode -> ELIC -> LatentCodec (entropy coding) -> FLUX multistep denoise -> AE decode"""
        batch_size = x01.shape[0]
        device = x01.device
 
        # 1. AE Encode
        z_clean, pad_info = self.encode_images(x01)
 
        # 2. ELIC Auxiliary Features
        z_aux = self.get_elic_features(x01, pad_info)

        # 3. Codec Compression/Decompression
        total_bytes = []
        if do_entropy_coding:
            self.codec.update(force=True)
            z_tcm_list, res_list = [], []
            for i in range(batch_size):
                comp = self.codec.compress(z_clean[i:i+1], z_aux[i:i+1])
                dec = self.codec.decompress(comp["strings"], comp["shape"])
                z_tcm_list.append(dec["x_hat"])
                res_list.append(dec["res"])
                y_bytes = sum(len(s) for s in comp["strings"][0])
                z_bytes = sum(len(s) for s in comp["strings"][1])
                total_bytes.append(float(y_bytes + z_bytes))
            z_tcm = torch.cat(z_tcm_list, dim=0)
            res_batch = torch.cat(res_list, dim=0) if res_list[0] is not None else None
        else:
            codec_out = self.codec(z_clean, z_aux)
            z_tcm = codec_out["x_hat"]
            res_batch = codec_out["res"]
            total_bytes = [0.0] * batch_size

        # 4. FLUX Multistep Denoise
        z_tcm_tokens, z_ids = self._latent_to_tokens(z_tcm)
        ctx, ctx_ids = self.get_text_context(batch_size=batch_size, device=device)
 
        timesteps = get_schedule(infer_steps, z_tcm_tokens.shape[1])
        z_out_tokens = denoise(
            self.flux,
            img=z_tcm_tokens, img_ids=z_ids,
            txt=ctx, txt_ids=ctx_ids,
            timesteps=timesteps,
            guidance=self.guidance,
        )
        z_out = self._tokens_to_latent(z_out_tokens, z_ids)
 
        # 5. Auxiliary Residual Addition
        if res_batch is not None:
            z_out = z_out + res_batch.to(z_out.dtype)
 
        # 6. AE Decode
        x_hat01 = self.decode_latents(z_out, pad_info)
 
        # 7. StableCodec Color Fix
        if color_fix:
            x_hat01 = apply_color_fix(x_hat01, x01)
            # Overhead: 3 channels * 2 stats * 16 bits = 12 bytes
            if do_entropy_coding:
                for i in range(len(total_bytes)):
                    total_bytes[i] += 12.0

        return {"x_hat": x_hat01, "bytes": total_bytes}

