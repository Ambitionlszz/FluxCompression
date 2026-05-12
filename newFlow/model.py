import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    get_schedule,
    scatter_ids,
)
from flux2.text_encoder import Qwen3Embedder
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model

from Block.latent_codec import LatentCodec

FIXED_PROMPT = "Lossless quality, artifact-free, preserved textures, clean details."


class FlowCompression(nn.Module):
    """
    基于 DiT-IC 架构的 Flux2 图像压缩模型。
    整合了 Flux AE + DiT-IC LatentCodec + Flux Transformer 的完整流程。

    Args:
        model_name:         Flux 模型名称 (如 "flux.2-klein-4b")
        flux_ckpt:          Flux 权重路径
        ae_ckpt:            AutoEncoder 权重路径
        qwen_ckpt:          Qwen tokenizer 路径
        codec_config:       LatentCodec 配置字典
        device:             运行设备
        guidance:           引导系数（默认 1.0）
        use_text_condition: True → 使用 FIXED_PROMPT，False → 使用空字符串 ""
        qwen_model_path:    Qwen 模型权重路径（默认与 qwen_ckpt 相同）
        elic_ckpt:          ELIC 辅助编码器权重路径（可选，None 则使用零张量）
    """

    def __init__(
        self,
        model_name: str,
        flux_ckpt: str,
        ae_ckpt: str,
        qwen_ckpt: str,
        codec_config: dict,
        device: torch.device,
        guidance: float = 1.0,
        use_text_condition: bool = False,
        qwen_model_path: str = None,
        elic_ckpt: str = None,
    ):
        super().__init__()
        model_info = FLUX2_MODEL_INFO[model_name]
        os.environ[model_info["model_path"]] = flux_ckpt
        os.environ["AE_MODEL_PATH"] = ae_ckpt

        self.model_name = model_name
        self.guidance = guidance
        self.guidance_distilled = bool(model_info.get("guidance_distilled", True))
        self.device = device
        self.use_text_condition = use_text_condition

        # 核心组件
        self.flux = load_flow_model(model_name, device=device)
        self.ae = load_ae(model_name, device=device)

        # 文本编码器（始终加载，用于构建缓存；cache_text_and_free_encoder() 会在缓存建立后释放）
        if qwen_model_path is None:
            qwen_model_path = qwen_ckpt
        self.text_encoder = Qwen3Embedder(
            model_spec=qwen_model_path,
            device=device,
            tokenizer_path=qwen_ckpt,
        )
        if use_text_condition:
            print("[FlowCompression] Text encoder loaded (FIXED_PROMPT mode)")
        else:
            print("[FlowCompression] Text encoder loaded (empty-text mode, will be freed after caching)")

        # Codec
        self.codec = self._build_codec(codec_config)

        # ELIC 辅助编码器
        if elic_ckpt is not None:
            self._load_elic_aux_encoder(elic_ckpt)
        else:
            self.elic_aux_encoder = None

        # 冻结固定模块
        self.ae.eval()
        self.flux.eval()
        for p in self.ae.parameters():
            p.requires_grad = False
        for p in self.flux.parameters():
            p.requires_grad = False
        if self.text_encoder is not None:
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        if self.elic_aux_encoder is not None:
            self.elic_aux_encoder.eval()
            for p in self.elic_aux_encoder.parameters():
                p.requires_grad = False

        self._prompt_cache = {}
        self._tcm_updated = True

    # ==================== 构建子模块 ====================

    def _build_codec(self, config: dict) -> LatentCodec:
        return LatentCodec(
            ch_emd=config.get("ch_emd", 128),
            channel=config.get("channel", 320),
            channel_out=config.get("channel_out", 128),
            num_slices=config.get("num_slices", 5),
        )

    def _load_elic_aux_encoder(self, elic_ckpt: str):
        from .elic_aux_encoder import load_elic_encoder
        if os.path.exists(elic_ckpt):
            self.elic_aux_encoder = load_elic_encoder(elic_ckpt, device=self.device, N=192, M=320)
            print(f"[FlowCompression] ELIC aux encoder loaded from: {elic_ckpt}")
        else:
            from .elic_aux_encoder import ELICAuxEncoder
            self.elic_aux_encoder = ELICAuxEncoder(N=192, M=320)
            self.elic_aux_encoder.to(self.device)
            print(f"[FlowCompression] WARNING: ELIC ckpt not found at {elic_ckpt}, using random init")
        self.elic_aux_encoder.eval()
        for p in self.elic_aux_encoder.parameters():
            p.requires_grad = False

    # ==================== 工具方法 ====================

    def cache_text_and_free_encoder(self):
        """
        预计算并缓存文本嵌入，随后卸载 text_encoder 释放显存。
        必须在 text_encoder 被卸载之前调用（例如训练开始前）。
        """
        if getattr(self, 'text_encoder', None) is None:
            print("[FlowCompression] text_encoder already freed, skip caching.")
            return

        print("[VRAM] Pre-computing text embeddings...")
        with torch.no_grad():
            old = self.use_text_condition
            self.use_text_condition = True
            self.get_text_context(batch_size=1, device=self.device)
            self.use_text_condition = False
            self.get_text_context(batch_size=1, device=self.device)
            self.use_text_condition = old

        print("[VRAM] Unloading text encoder...")
        del self.text_encoder
        self.text_encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[VRAM] Text encoder freed.\n")
        
        # 注册 eval 时是否需要 update 的标记
        self._tcm_updated = True

    def enable_gradient_checkpointing(self, enabled: bool = True):
        if not enabled:
            return
        if hasattr(self.flux, 'gradient_checkpointing_enable'):
            self.flux.gradient_checkpointing_enable()
            print("[GradCkpt] Flux: enabled")
        else:
            print("[GradCkpt] Flux: not supported")

    @torch.no_grad()
    def encode_images(self, x01: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """[0,1] 图像 → BN 归一化的 latent，并返回 padding 信息。"""
        h, w = x01.shape[-2], x01.shape[-1]
        alignment = 64  # AE 下采样 16x，Codec 再下采样 4x
        pad_h = (alignment - h % alignment) % alignment
        pad_w = (alignment - w % alignment) % alignment
        if pad_h > 0 or pad_w > 0:
            x01 = F.pad(x01, (0, pad_w, 0, pad_h), mode='reflect')
        x = x01 * 2.0 - 1.0
        z_main = self.ae.encode(x.to(next(self.ae.parameters()).dtype)).float()
        return z_main, {"pad_h": pad_h, "pad_w": pad_w}

    def decode_latents(self, z: torch.Tensor, pad_info: dict = None) -> torch.Tensor:
        """BN 归一化的 latent → [0,1] 图像，可选去除 padding。
        注意：不加 @torch.no_grad()，梯度需要从 MSE/LPIPS loss 经此回传到 z_hat。
        AE 参数已 frozen (requires_grad=False)，不会被更新。
        """
        x = self.ae.decode(z.to(next(self.ae.parameters()).dtype)).float()
        x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        if pad_info:
            ph, pw = pad_info["pad_h"], pad_info["pad_w"]
            if ph > 0 or pw > 0:
                x = x[:, :, :x.shape[2] - ph, :x.shape[3] - pw]
        return x

    @torch.no_grad()
    def get_text_context(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回文本上下文 (ctx, ctx_ids)，支持缓存复用。
        - use_text_condition=True:  编码 FIXED_PROMPT
        - use_text_condition=False: 编码空字符串 ""
        text_encoder 卸载后仍可通过缓存正常工作。
        """
        if not self.use_text_condition:
            cache_ctx_key = '_cached_empty_ctx'
            cache_ids_key = '_cached_empty_ctx_ids'
            encode_input = [""]
        else:
            cache_ctx_key = '_cached_ctx'
            cache_ids_key = '_cached_ctx_ids'
            encode_input = [FIXED_PROMPT]

        if not hasattr(self, cache_ctx_key):
            if self.text_encoder is None:
                raise RuntimeError(
                    f"[get_text_context] text_encoder 已被卸载但缓存未建立！"
                    f"请在调用 cache_text_and_free_encoder() 之前先触发一次 get_text_context()。"
                )
            if self.guidance_distilled:
                ctx_raw = self.text_encoder(encode_input).to(torch.bfloat16)
            else:
                ctx_empty = self.text_encoder([""]).to(torch.bfloat16)
                ctx_prompt = self.text_encoder(encode_input).to(torch.bfloat16)
                ctx_raw = torch.cat([ctx_empty, ctx_prompt], dim=0)
            ctx_single, ctx_ids_single = batched_prc_txt(ctx_raw)
            setattr(self, cache_ctx_key, ctx_single.cpu())
            setattr(self, cache_ids_key, ctx_ids_single.cpu())

        ctx = getattr(self, cache_ctx_key).to(device).repeat(batch_size, 1, 1)
        ctx_ids = getattr(self, cache_ids_key).to(device).repeat(batch_size, 1, 1)
        return ctx, ctx_ids

    def _latent_to_tokens(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return batched_prc_img(z.to(torch.bfloat16))

    def _tokens_to_latent(self, x_tokens: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat(scatter_ids(x_tokens, x_ids)).squeeze(2)

    def _get_z_aux(self, x01: torch.Tensor, pad_info: dict) -> torch.Tensor:
        """
        从像素域图像获取 ELIC 辅助特征。
        必须使用与 z_main 相同的 padded 图像，保证空间尺寸一致。
        """
        if self.elic_aux_encoder is not None:
            ph, pw = pad_info["pad_h"], pad_info["pad_w"]
            x01_padded = x01
            if ph > 0 or pw > 0:
                x01_padded = F.pad(x01, (0, pw, 0, ph), mode='reflect')
            return self.elic_aux_encoder(x01_padded)
        else:
            # 无 ELIC 编码器：使用全零占位（g_a 仍可运行，但辅助信息为零）
            h_lat = x01.shape[-2] // 16  # AE 下采样 16x
            w_lat = x01.shape[-1] // 16
            ph, pw = pad_info["pad_h"], pad_info["pad_w"]
            h_lat = (x01.shape[-2] + ph) // 16
            w_lat = (x01.shape[-1] + pw) // 16
            return torch.zeros(x01.shape[0], 320, h_lat, w_lat,
                               device=x01.device, dtype=x01.dtype)

    # ==================== 训练接口 ====================

    def forward(
        self,
        x01: torch.Tensor,
        z_aux: torch.Tensor,
        train_schedule_steps: int = 50,
    ) -> dict:
        """
        训练前向传播。

        Args:
            x01:                  [0,1] 图像 (B, 3, H, W)
            z_aux:                ELIC 辅助特征 (B, 320, H_pad/16, W_pad/16)
            train_schedule_steps: Flow Matching 调度步数

        Returns:
            dict: x_hat, likelihoods, z_clean, z_tcm, res
        """
        batch_size = x01.shape[0]
        device = x01.device

        # 1. VAE 编码
        z_main, pad_info = self.encode_images(x01)

        # 2. LatentCodec 压缩
        codec_out = self.codec(z_main, z_aux)
        z_tcm = codec_out["x_hat"].to(z_main.dtype)
        res = codec_out.get("res", None)

        # 3. Flux 去噪
        z_tokens, z_ids = self._latent_to_tokens(z_tcm)
        
        schedule = get_schedule(train_schedule_steps, z_tokens.shape[1])
        schedule_tensor = torch.tensor(schedule, dtype=z_tokens.dtype, device=device)
        step_idx = torch.randint(0, train_schedule_steps, (batch_size,), device=device)
        timesteps = schedule_tensor[step_idx]

        ctx, ctx_ids = self.get_text_context(batch_size=batch_size, device=device)
        guidance_vec = torch.full((batch_size,), self.guidance, dtype=z_tokens.dtype, device=device)

        v_pred_tokens = self.flux(
            x=z_tokens, x_ids=z_ids,
            timesteps=timesteps,
            ctx=ctx, ctx_ids=ctx_ids,
            guidance=guidance_vec,
        )

        # 4. Flow Matching 重建
        z_hat_tokens = z_tokens - timesteps.view(-1, 1, 1) * v_pred_tokens
        z_hat = self._tokens_to_latent(z_hat_tokens, z_ids)

        # 5. 辅助残差叠加
        if res is not None:
            if res.shape[-2:] != z_hat.shape[-2:]:
                res = F.interpolate(res, size=z_hat.shape[-2:], mode='bilinear', align_corners=False)
            z_hat = z_hat + res.to(z_hat.dtype)

        # 6. VAE 解码（使用普通 clamp，对齐 Flow 的实现）
        x_hat01 = self.decode_latents(z_hat, pad_info)

        return {
            "x_hat":       x_hat01,
            "likelihoods": codec_out["likelihoods"],
            "z_clean":     z_main,
            "z_tcm":       z_tcm,
            "res":         res,
        }

    # ==================== 推理接口 ====================

    @torch.no_grad()
    def forward_stage1_infer(
        self,
        x01: torch.Tensor,
        infer_steps: int = 4,
        do_entropy_coding: bool = True,
        debug: bool = False,
    ) -> Dict[str, any]:
        """
        完整推理：VAE encode → ELIC → LatentCodec（含熵编码）→ Flux denoise → VAE decode。

        Args:
            x01:              [0,1] 图像 (B, 3, H, W)
            infer_steps:      Flux 去噪步数
            do_entropy_coding: True → 真实熵编码/解码（准确 BPP），False → 直接前向
            debug:            True → 打印每阶段 tensor 统计，用于诊断

        Returns:
            dict: x_hat (B, 3, H, W), bytes (每张图字节数列表)
        """
        batch_size = x01.shape[0]
        device = x01.device

        # 1. VAE 编码
        z_main, pad_info = self.encode_images(x01)

        # 2. ELIC 辅助特征
        with torch.no_grad():
            z_aux = self._get_z_aux(x01, pad_info)

        # 3. Codec 压缩/解压
        total_bytes = []
        if do_entropy_coding:
            # ⭐ 关键修复：对于从零训练的模型，每次评估前必须更新概率模型的 CDF 表
            # 否则 Arithmetic Coder 会使用过期的 CDF，导致解码误差极大，PSNR 跌至 14dB 以下
            if debug:
                print("[TCM] Updating probability models for accurate evaluation...")
            self.codec.update(force=True)
            
            z_tcm_list, res_list = [], []
            for i in range(batch_size):
                comp = self.codec.compress(z_main[i:i+1], z_aux[i:i+1])
                dec = self.codec.decompress(comp["strings"], comp["shape"])
                z_tcm_list.append(dec["x_hat"])
                res_list.append(dec["res"])
                y_bytes = sum(len(s) for s in comp["strings"][0])
                z_bytes = sum(len(s) for s in comp["strings"][1])
                total_bytes.append(float(y_bytes + z_bytes))
            z_tcm = torch.cat(z_tcm_list, dim=0)
            res_batch = torch.cat(res_list, dim=0)
        else:
            codec_out = self.codec(z_main, z_aux)
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
        if res_batch.shape[-2:] != z_out.shape[-2:]:
            res_batch = F.interpolate(res_batch, size=z_out.shape[-2:], mode='bilinear', align_corners=False)
        z_out = z_out + res_batch.to(z_out.dtype)

        # 6. VAE 解码
        x_hat01 = self.decode_latents(z_out, pad_info)

        # ── Debug 日志（传入 debug=True 或在 train.py 中按需调用）──
        if debug:
            print(f"\n{'='*60}")
            print(f"[INFER DEBUG] entropy_coding={do_entropy_coding}, steps={infer_steps}")
            print(f"  x01       : range=[{x01.min():.3f}, {x01.max():.3f}]  mean={x01.mean():.3f}")
            print(f"  z_main    : range=[{z_main.min():.3f}, {z_main.max():.3f}]  std={z_main.std():.3f}")
            print(f"  z_aux     : range=[{z_aux.min():.3f}, {z_aux.max():.3f}]  std={z_aux.std():.3f}")
            print(f"  z_tcm     : range=[{z_tcm.min():.3f}, {z_tcm.max():.3f}]  std={z_tcm.std():.3f}")
            print(f"  res_batch : range=[{res_batch.min():.3f}, {res_batch.max():.3f}]  mean={res_batch.mean():.3f}")
            print(f"  z_out     : range=[{z_out.min():.3f}, {z_out.max():.3f}]  std={z_out.std():.3f}")
            print(f"  x_hat01   : range=[{x_hat01.min():.3f}, {x_hat01.max():.3f}]  mean={x_hat01.mean():.3f}")
            print(f"  bytes     : {total_bytes}")
            print(f"{'='*60}\n")

        return {"x_hat": x_hat01, "bytes": total_bytes}
