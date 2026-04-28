import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    get_schedule,
    scatter_ids,
)
from flux2.text_encoder import Qwen3Embedder
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

from LIC_TCM.models import TCMLatent


FIXED_PROMPT = "Lossless quality, artifact-free, preserved textures, clean details."


class FlowTCMStage1Pipeline(nn.Module):
    def __init__(
        self,
        model_name: str,
        flux_ckpt: str,
        ae_ckpt: str,
        qwen_ckpt: Optional[str],
        tcm: TCMLatent,
        device: torch.device,
        guidance: float = 1.0,
        use_gradient_checkpointing: bool = False,  # 新增参数：控制 Gradient Checkpointing
    ):
        super().__init__()
        model_info = FLUX2_MODEL_INFO[model_name]
        os.environ[model_info["model_path"]] = flux_ckpt
        os.environ["AE_MODEL_PATH"] = ae_ckpt

        self.model_name = model_name
        self.guidance = guidance
        self.guidance_distilled = bool(model_info.get("guidance_distilled", True))

        # 先加载 TCM（较小）
        self.tcm = tcm
        
        # 加载 Flux 和 AE
        print(f"Loading {flux_ckpt} for the FLUX.2 weights")
        self.flux = load_flow_model(model_name, device=device)
        
        print(f"Loading {ae_ckpt} for the AutoEncoder weights")
        self.ae = load_ae(model_name, device=device)
        
        # 加载文本编码器
        if qwen_ckpt:
            # 使用本地路径加载 Qwen 模型
            print(f"Loading text encoder from local path: {qwen_ckpt}")
            self.text_encoder = Qwen3Embedder(model_spec=qwen_ckpt, device=device)
        else:
            # 如果没有提供 qwen_ckpt，尝试使用默认路径
            default_qwen_path = "/data2/luosheng/hf_models/hub/Qwen3-4B-FP8"
            if os.path.exists(default_qwen_path):
                print(f"Loading text encoder from default local path: {default_qwen_path}")
                self.text_encoder = Qwen3Embedder(model_spec=default_qwen_path, device=device)
            else:
                # 最后才尝试从 HuggingFace 加载
                print("Warning: No local Qwen model found, trying to load from HuggingFace...")
                self.text_encoder = load_text_encoder(model_name, device=device)

        self.ae.eval()
        self.text_encoder.eval()
        self.flux.eval()  # ← 新增：确保 FLUX 模型也设置为 eval 模式
        for p in self.ae.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.flux.parameters():  # ← 新增：禁用 FLUX 的梯度
            p.requires_grad = False

        # 优化：根据参数决定是否启用 Gradient Checkpointing
        if use_gradient_checkpointing:
            if hasattr(self.flux, 'gradient_checkpointing_enable'):
                print("✓ Enabling gradient checkpointing for FLUX model (saves 4-6 GB VRAM, ~20-30% slower)")
                self.flux.gradient_checkpointing_enable()
            else:
                print("Warning: FLUX model does not support gradient checkpointing")
        else:
            print("✓ Gradient checkpointing is disabled (faster inference, higher VRAM usage)")
        
        self._prompt_cache = {}

    @torch.no_grad()
    def encode_images(self, x01: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        编码图像，自动处理尺寸对齐
        
        Returns:
            z: 编码后的 latent
            pad_info: padding 信息字典，包含 original_shape, pad_h, pad_w
        """
        # 记录原始形状
        original_shape = x01.shape  # [B, C, H, W]
        h, w = original_shape[-2], original_shape[-1]
        
        # 计算需要的对齐要求：
        # FLUX.2 AutoEncoder 下采样倍数：16
        # TCM g_a: conv3x3(stride=2) → y = x/2
        # TCM h_a: ResidualBlockWithStride(stride=2) + conv3x3(stride=2) → z = y/4
        # 
        # 约束条件：
        # 1. TCM window_size=8: y 需要能被 8 整除
        # 2. h_a atten_window_size=4: z 需要能被 4 整除 → y 需要能被 16 整除
        # 3. 综合：y 需要能被 lcm(8, 16) = 16 整除
        # 4. 因为 y = x/2，所以 x 需要能被 32 整除
        ae_downsample = 16  # FLUX.2 AE 的下采样倍数
        ga_downsample = 2   # TCM g_a 的下采样倍数 (stride=2)
        y_required_alignment = 16  # y 需要能被 16 整除
        required_alignment = ae_downsample * ga_downsample * (y_required_alignment // ga_downsample)
        # 简化后：required_alignment = 32
        
        pad_h = (required_alignment - h % required_alignment) % required_alignment
        pad_w = (required_alignment - w % required_alignment) % required_alignment
        
        # 如果需要 padding
        if pad_h > 0 or pad_w > 0:
            # reflect padding 通常比 zero padding 效果更好
            x_padded = F.pad(x01, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_padded = x01
        
        # 编码
        x = x_padded * 2.0 - 1.0
        z = self.ae.encode(x.to(next(self.ae.parameters()).dtype))
        
        # 保存 padding 信息
        pad_info = {
            "original_shape": original_shape,
            "pad_h": pad_h,
            "pad_w": pad_w,
        }
        
        return z.float(), pad_info

    def decode_latents(self, z: torch.Tensor, pad_info: dict = None) -> torch.Tensor:
        """
        解码 latent，如果提供了 pad_info 则去除 padding
        
        Args:
            z: latent tensor
            pad_info: padding 信息字典（可选）
        """
        x = self.ae.decode(z.to(next(self.ae.parameters()).dtype)).float()
        x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        
        # 如果有 padding 信息，去除 padding
        if pad_info is not None:
            pad_h = pad_info["pad_h"]
            pad_w = pad_info["pad_w"]
            if pad_h > 0 or pad_w > 0:
                orig_h, orig_w = pad_info["original_shape"][-2], pad_info["original_shape"][-1]
                x = x[:, :, :orig_h, :orig_w]
        
        return x

    @torch.no_grad()
    def get_text_context(self, batch_size: int, device: torch.device):
        if batch_size in self._prompt_cache:
            # 缓存命中，不打印
            ctx, ctx_ids = self._prompt_cache[batch_size]
            # 优化：缓存 GPU tensor，避免每次传输
            if ctx.device != device:
                ctx = ctx.to(device)
                ctx_ids = ctx_ids.to(device)
                self._prompt_cache[batch_size] = (ctx, ctx_ids)
            return ctx, ctx_ids

        import time
        t_text_start = time.time()
        print(f"  [TextEncoder] ⚠️  Cache MISS for batch_size={batch_size}, encoding...")
        prompts = [FIXED_PROMPT] * batch_size
        if self.guidance_distilled:
            ctx = self.text_encoder(prompts).to(torch.bfloat16)
        else:
            ctx_empty = self.text_encoder([""] * batch_size).to(torch.bfloat16)
            ctx_prompt = self.text_encoder(prompts).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
        torch.cuda.synchronize()
        t_text_end = time.time()
        print(f"  [TextEncoder] Encoding took {t_text_end - t_text_start:.3f}s")

        ctx, ctx_ids = batched_prc_txt(ctx)
        # 优化：直接缓存到 GPU
        self._prompt_cache[batch_size] = (ctx.to(device), ctx_ids.to(device))
        print(f"  [TextEncoder] ✓ Cached for batch_size={batch_size}")
        return ctx, ctx_ids

    def _latent_to_tokens(self, z: torch.Tensor):
        z = z.to(torch.bfloat16)
        return batched_prc_img(z)

    def _tokens_to_latent(self, x_tokens: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        z = torch.cat(scatter_ids(x_tokens, x_ids)).squeeze(2)
        return z

    def sample_sigmas(self, image_seq_len: int, batch_size: int, steps: int, device: torch.device) -> torch.Tensor:
        schedule = get_schedule(steps, image_seq_len)
        sched = torch.tensor(schedule, dtype=torch.float32, device=device)
        idx = torch.randint(0, sched.shape[0], (batch_size,), device=device)
        return sched[idx]

    def forward_stage1_train(self, x01: torch.Tensor, train_schedule_steps: int) -> dict:
        """
        训练模式的前向传播（参照 DiT-IC + FLUX.2 原生调度）
        
        核心策略：
        - 直接使用压缩特征 z_tcm 作为 Flux 输入（不与 z_clean 混合）
        - 使用 FLUX.2 原生的多步时间步调度（与推理一致）
        - 随机选择一个时间步进行单步训练（Flow Matching 标准做法）
        """
        batch_size = x01.shape[0]
        device = x01.device

        # 1. 编码得到干净 latent（训练时通常不需要 padding，因为数据已经预处理过）
        z_clean, _ = self.encode_images(x01)
        
        # === DEBUG: 检查 z_clean 的范围 ===
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        if self._debug_step % 1000 == 0:
            print(f"\n[DEBUG Step {self._debug_step}] ===== Training Forward Debug =====")
            print(f"  Input x01 range: [{x01.min():.4f}, {x01.max():.4f}], mean={x01.mean():.4f}")
            print(f"  z_clean range: [{z_clean.min():.4f}, {z_clean.max():.4f}], mean={z_clean.mean():.4f}, std={z_clean.std():.4f}")
        
        # 2. TCM 压缩（使用 ste_round 量化）
        tcm_out = self.tcm(z_clean)
        z_tcm = tcm_out["x_hat"]
        z_tcm = z_tcm.to(z_clean.dtype)
        
        # === DEBUG: 检查 TCM 输出范围 ===
        if self._debug_step % 1000 == 0:
            print(f"  z_tcm (after TCM) range: [{z_tcm.min():.4f}, {z_tcm.max():.4f}], mean={z_tcm.mean():.4f}, std={z_tcm.std():.4f}")
            print(f"  z_clean vs z_tcm MSE: {F.mse_loss(z_clean, z_tcm).item():.6f}")

        # 3. 准备 Flux 输入：直接使用 z_tcm（参照 DiT-IC，不混合 z_clean）
        z_tokens, z_ids = self._latent_to_tokens(z_tcm)
        
        # === DEBUG: 检查 Flux 输入 ===
        if self._debug_step % 1000 == 0:
            print(f"  Flux input (z_tokens from z_tcm) shape: {z_tokens.shape}")
            print(f"  z_tokens range: [{z_tokens.min():.4f}, {z_tokens.max():.4f}], mean={z_tokens.mean():.4f}, std={z_tokens.std():.4f}")
        
        # 4. 获取 FLUX.2 原生的多步时间步调度（与推理一致）
        # 这会生成 [t_0=1.0, t_1, t_2, t_3, t_4=0.0] 共 5 个时间点
        schedule = get_schedule(train_schedule_steps, z_tokens.shape[1])
        schedule_tensor = torch.tensor(schedule, dtype=z_tokens.dtype, device=device)
        
        # 5. 随机选择一个时间步索引（除了最后一个 t=0）
        # 这样训练时可以看到不同的噪声水平，但都是从 z_tcm 开始
        step_idx = torch.randint(0, train_schedule_steps, (batch_size,), device=device)
        timesteps = schedule_tensor[step_idx]  # 形状: [batch_size]
        
        if self._debug_step % 1000 == 0:
            print(f"  Schedule steps: {train_schedule_steps}, selected timesteps range: [{timesteps.min():.4f}, {timesteps.max():.4f}]")
            print(f"  Full schedule: {[f'{t:.3f}' for t in schedule[:5]]}...\n")
        
        # 6. 获取文本上下文
        ctx, ctx_ids = self.get_text_context(batch_size=batch_size, device=device)
        guidance_vec = torch.full((z_tokens.shape[0],), self.guidance, dtype=z_tokens.dtype, device=device)
        
        # 7. Flux 预测速度场（单步，随机时间步）
        v_pred_tokens = self.flux(
            x=z_tokens,
            x_ids=z_ids,
            timesteps=timesteps,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=guidance_vec,
        )

        # 8. 重建：z_hat = z_tcm - t * v_pred（Flow Matching 更新公式）
        z_hat_tokens = z_tokens - timesteps.view(-1, 1, 1) * v_pred_tokens
        z_hat = self._tokens_to_latent(z_hat_tokens, z_ids)
        x_hat01 = self.decode_latents(z_hat)
        
        self._debug_step += 1

        return {
            "x_hat": x_hat01,
            "likelihoods": tcm_out["likelihoods"],
            "z_clean": z_clean,
            "z_tcm": z_tcm,
            "sigma": timesteps,  # 返回实际使用的时间步用于日志
        }

    @torch.no_grad()
    def forward_stage1_infer(
        self,
        x01: torch.Tensor,
        infer_steps: int = 4,
        do_entropy_coding: bool = True,
    ) -> dict:
        batch_size = x01.shape[0]
        device = x01.device

        # 编码图像，获取 padding 信息
        z_clean, pad_info = self.encode_images(x01)

        total_bytes = []
        z_tcm_list = []
        if do_entropy_coding:
            self.tcm.update(force=True)
            for i in range(batch_size):
                zi = z_clean[i : i + 1]
                
                comp = self.tcm.compress(zi)
                dec = self.tcm.decompress(comp["strings"], comp["shape"])
                zi_hat = dec["x_hat"]
                y_bytes = sum(len(s) for s in comp["strings"][0])
                z_bytes = sum(len(s) for s in comp["strings"][1])
                total_bytes.append(float(y_bytes + z_bytes))
                z_tcm_list.append(zi_hat)
            z_tcm = torch.cat(z_tcm_list, dim=0)
        else:
            out = self.tcm(z_clean)
            z_tcm = out["x_hat"]
            total_bytes = [0.0] * batch_size

        z_tcm_tokens, z_ids = self._latent_to_tokens(z_tcm)
        ctx, ctx_ids = self.get_text_context(batch_size=batch_size, device=device)

        timesteps = get_schedule(infer_steps, z_tcm_tokens.shape[1])
        z_out_tokens = denoise(
            self.flux,
            img=z_tcm_tokens,
            img_ids=z_ids,
            txt=ctx,
            txt_ids=ctx_ids,
            timesteps=timesteps,
            guidance=self.guidance,
        )
        z_out = self._tokens_to_latent(z_out_tokens, z_ids)
        
        # 解码时使用 padding 信息去除 padding
        x_hat01 = self.decode_latents(z_out, pad_info)

        return {
            "x_hat": x_hat01,
            "bytes": total_bytes,
        }

    @torch.no_grad()
    def forward_stage1_infer_with_timing(
        self,
        x01: torch.Tensor,
        infer_steps: int = 4,
        do_entropy_coding: bool = True,
    ) -> dict:
        """带时间测量的推理方法"""
        import time
        import torch
        
        batch_size = x01.shape[0]
        device = x01.device
        
        # 1. AE Encode
        t_encode_start = time.time()
        z_clean, pad_info = self.encode_images(x01)
        torch.cuda.synchronize()
        time_encode = time.time() - t_encode_start
        
        # 2. TCM Compress + Decompress
        t_compress_start = time.time()
        total_bytes = []
        z_tcm_list = []
        if do_entropy_coding:
            # 优化1: 只在第一次调用时更新概率模型，后续复用
            if not hasattr(self, '_tcm_updated') or self._tcm_updated:
                print(f"  [TCM] Updating probability models (one-time cost)...")
                t_update_start = time.time()
                self.tcm.update(force=True)
                torch.cuda.synchronize()
                time_update = time.time() - t_update_start
                print(f"  [TCM] Probability model update took {time_update:.3f}s")
                self._tcm_updated = False
            
            # 优化2: 批量处理而不是逐张处理
            # 如果batch_size较小，仍然逐张处理以保证正确性
            # 但如果batch_size较大，可以考虑真正的批量化
            for i in range(batch_size):
                zi = z_clean[i : i + 1]
                
                t_comp_start = time.time()
                comp = self.tcm.compress(zi)
                torch.cuda.synchronize()
                time_comp = time.time() - t_comp_start
                
                t_decomp_start = time.time()
                dec = self.tcm.decompress(comp["strings"], comp["shape"])
                torch.cuda.synchronize()
                time_decomp = time.time() - t_decomp_start
                
                if i == 0:  # 只打印第一张的详细信息
                    print(f"  [TCM] Compress: {time_comp:.3f}s, Decompress: {time_decomp:.3f}s")
                
                zi_hat = dec["x_hat"]
                y_bytes = sum(len(s) for s in comp["strings"][0])
                z_bytes = sum(len(s) for s in comp["strings"][1])
                total_bytes.append(float(y_bytes + z_bytes))
                z_tcm_list.append(zi_hat)
            z_tcm = torch.cat(z_tcm_list, dim=0)
        else:
            # 快速模式：跳过熵编码
            out = self.tcm(z_clean)
            z_tcm = out["x_hat"]
            total_bytes = [0.0] * batch_size
        torch.cuda.synchronize()
        time_compress = time.time() - t_compress_start
        
        # 3. Text Encoding (separate from denoise)
        t_text_start = time.time()
        z_tcm_tokens, z_ids = self._latent_to_tokens(z_tcm)
        ctx, ctx_ids = self.get_text_context(batch_size=batch_size, device=device)
        torch.cuda.synchronize()
        time_text = time.time() - t_text_start
        
        # 4. Flux Denoise
        t_denoise_start = time.time()
        
        # DEBUG: 检查输入（只在第一次调用时打印）
        if not hasattr(self, '_denoise_debug_printed'):
            print(f"\n[DEBUG Denoise Input]")
            print(f"  z_tcm_tokens shape: {z_tcm_tokens.shape}")
            print(f"  z_tcm_tokens dtype: {z_tcm_tokens.dtype}")
            print(f"  z_tcm_tokens device: {z_tcm_tokens.device}")
            print(f"  ctx shape: {ctx.shape}")
            print(f"  ctx dtype: {ctx.dtype}")
            print(f"  infer_steps: {infer_steps}")
            image_seq_len = z_tcm_tokens.shape[1]
            print(f"  image_seq_len: {image_seq_len}")
            self._denoise_debug_printed = True
        
        timesteps = get_schedule(infer_steps, z_tcm_tokens.shape[1])
        
        if not hasattr(self, '_denoise_debug_printed2'):
            print(f"  timesteps: {timesteps}")
            print(f"  Starting denoise...")
            import time as _time
            _denoise_start = _time.time()
            self._denoise_debug_printed2 = True
        
        z_out_tokens = denoise(
            self.flux,
            img=z_tcm_tokens,
            img_ids=z_ids,
            txt=ctx,
            txt_ids=ctx_ids,
            timesteps=timesteps,
            guidance=self.guidance,
        )
        
        if hasattr(self, '_denoise_debug_printed2') and self._denoise_debug_printed2:
            torch.cuda.synchronize()
            _denoise_end = _time.time()
            print(f"  Denoise completed in {_denoise_end - _denoise_start:.3f}s")
            print(f"  z_out_tokens shape: {z_out_tokens.shape}\n")
            self._denoise_debug_printed2 = False  # 只打印一次
        
        z_out = self._tokens_to_latent(z_out_tokens, z_ids)
        torch.cuda.synchronize()
        time_denoise = time.time() - t_denoise_start
        
        # 5. AE Decode
        t_decode_start = time.time()
        x_hat01 = self.decode_latents(z_out, pad_info)
        torch.cuda.synchronize()
        time_decode = time.time() - t_decode_start
        
        return {
            "x_hat": x_hat01,
            "bytes": total_bytes,
            "time_encode": time_encode,
            "time_compress": time_compress,
            "time_denoise": time_denoise,
            "time_text": time_text,  # 新增：文本编码时间
            "time_decode": time_decode,
        }
