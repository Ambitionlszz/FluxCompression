"""
FluxCodec Pipeline - 基于 Flow/modules/pipeline.py 改造。
将 TCM 替换为 DiT-IC 风格的 LatentCodec，增加 ELIC 辅助编码器。

Pipeline: AE encode → ELIC aux encode → LatentCodec (g_a→entropy→g_s+aux) → FLUX denoise → AE decode
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

        # LatentCodec (可训练)
        self.codec = codec

        # ELIC 辅助编码器 (冻结)
        self.elic_aux_encoder = elic_aux_encoder
        if self.elic_aux_encoder is not None:
            self.elic_aux_encoder.eval()
            for p in self.elic_aux_encoder.parameters():
                p.requires_grad = False

        # 加载 Flux 和 AE
        print(f"Loading {flux_ckpt} for the FLUX.2 weights")
        self.flux = load_flow_model(model_name, device=device)

        print(f"Loading {ae_ckpt} for the AutoEncoder weights")
        self.ae = load_ae(model_name, device=device)

        # 加载文本编码器
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

        # 冻结固定组件
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

    # ==================== 编解码工具 ====================

    @torch.no_grad()
    def encode_images(self, x01: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """[0,1] 图像 → latent，自动处理 padding 对齐。"""
        h, w = x01.shape[-2], x01.shape[-1]
        # AE 16x, codec g_a 4x → 总下采样 64x
        # h_a 再 4x → z 空间需要整除
        # 保守对齐到 64
        alignment = 64
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
        """latent → [0,1] 图像，可选去除 padding。"""
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
        """获取 ELIC 辅助特征。ELIC 16x 下采样，与 FLUX AE 空间对齐。"""
        if self.elic_aux_encoder is None:
            # 无 ELIC 编码器：返回零张量
            h, w = x01.shape[-2], x01.shape[-1]
            ph, pw = pad_info["pad_h"], pad_info["pad_w"]
            h_lat = (h + ph) // 16
            w_lat = (w + pw) // 16
            return torch.zeros(x01.shape[0], 320, h_lat, w_lat,
                               device=x01.device, dtype=x01.dtype)

        # ELIC 期望 [0,1] 输入，使用相同的 padding
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

    # ==================== 训练接口 ====================

    def forward_stage1_train(self, x01: torch.Tensor, train_schedule_steps: int) -> dict:
        """
        训练前向传播：
        AE encode → ELIC → LatentCodec → Flux 单步去噪 → AE decode
        """
        batch_size = x01.shape[0]
        device = x01.device

        # 1. AE 编码
        z_clean, pad_info = self.encode_images(x01)

        # 2. ELIC 辅助特征
        z_aux = self.get_elic_features(x01, pad_info)

        # 3. LatentCodec 压缩
        codec_out = self.codec(z_clean, z_aux)
        z_tcm = codec_out["x_hat"].to(z_clean.dtype)
        res = codec_out.get("res", None)

        # 4. Flux 去噪准备
        z_tokens, z_ids = self._latent_to_tokens(z_tcm)

        schedule = get_schedule(train_schedule_steps, z_tokens.shape[1])
        schedule_tensor = torch.tensor(schedule, dtype=z_tokens.dtype, device=device)
        step_idx = torch.randint(0, train_schedule_steps, (batch_size,), device=device)
        timesteps = schedule_tensor[step_idx]

        ctx, ctx_ids = self.get_text_context(batch_size=batch_size, device=device)
        guidance_vec = torch.full(
            (z_tokens.shape[0],), self.guidance, dtype=z_tokens.dtype, device=device)

        # 5. Flux 预测速度场
        v_pred_tokens = self.flux(
            x=z_tokens, x_ids=z_ids,
            timesteps=timesteps,
            ctx=ctx, ctx_ids=ctx_ids,
            guidance=guidance_vec,
        )

        # 6. Flow Matching 重建: z_hat = z_tcm - t * v_pred
        z_hat_tokens = z_tokens - timesteps.view(-1, 1, 1) * v_pred_tokens
        z_hat = self._tokens_to_latent(z_hat_tokens, z_ids)

        # 7. 辅助残差叠加
        if res is not None:
            z_hat = z_hat + res.to(z_hat.dtype)

        # 8. AE 解码
        x_hat01 = self.decode_latents(z_hat, pad_info)

        return {
            "x_hat": x_hat01,
            "likelihoods": codec_out["likelihoods"],
            "z_clean": z_clean,
            "z_tcm": z_tcm,
            "sigma": timesteps,
        }

    # ==================== 推理接口 ====================

    @torch.no_grad()
    def forward_stage1_infer(
        self,
        x01: torch.Tensor,
        infer_steps: int = 4,
        do_entropy_coding: bool = True,
    ) -> dict:
        """推理：AE encode → ELIC → LatentCodec (熵编码) → Flux 多步去噪 → AE decode"""
        batch_size = x01.shape[0]
        device = x01.device

        # 1. AE 编码
        z_clean, pad_info = self.encode_images(x01)

        # 2. ELIC 辅助特征
        z_aux = self.get_elic_features(x01, pad_info)

        # 3. Codec 压缩/解压
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
            res_batch = torch.cat(res_list, dim=0)
        else:
            codec_out = self.codec(z_clean, z_aux)
            z_tcm = codec_out["x_hat"]
            res_batch = codec_out["res"]
            total_bytes = [0.0] * batch_size

        # 4. Flux 多步去噪
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

        # 5. 辅助残差叠加
        if res_batch is not None:
            z_out = z_out + res_batch.to(z_out.dtype)

        # 6. AE 解码
        x_hat01 = self.decode_latents(z_out, pad_info)

        return {"x_hat": x_hat01, "bytes": total_bytes}
