import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保 src 目录在路径中，以便导入 flux2 包
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
    基于 DiT-IC 架构的 Flux2 压缩模型
    
    整合了 Flux AE + DiT-IC Codec + Flux Transformer 的完整流程
    
    Args:
        model_name: 模型名称 (如 "flux.2-klein-4b")
        flux_ckpt: Flux 模型权重路径
        ae_ckpt: AutoEncoder 权重路径
        qwen_ckpt: Qwen 文本编码器路径
        codec_config: DiT-IC Codec 配置参数字典
        device: 运行设备
        guidance: 引导系数 (默认 1.0)
        use_text_condition: 是否使用真实文本作为条件 (默认 True)
    
    Example:
        >>> # 初始化模型
        >>> model = FlowCompression(
        ...     model_name="flux.2-klein-4b",
        ...     flux_ckpt="/path/to/flux.safetensors",
        ...     ae_ckpt="/path/to/ae.safetensors",
        ...     qwen_ckpt="/path/to/qwen",
        ...     codec_config=codec_args,
        ...     device=torch.device("cuda"),
        ...     use_text_condition=True,
        ... )
        
        >>> # 训练模式
        >>> output = model.forward(x01, train_schedule_steps=50)
        
        >>> # 推理模式 - 压缩
        >>> compressed = model.compress(z_clean)
        
        >>> # 推理模式 - 解压缩
        >>> reconstructed = model.decompress(compressed["strings"], compressed["shape"])
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
        use_text_condition: bool = True,
        qwen_model_path: str = None,  # Qwen 模型权重路径（可选，默认与 qwen_ckpt 相同）
        elic_ckpt: str = None,  # ELIC 辅助编码器权重路径（可选）
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

        # 加载核心组件
        self.flux = load_flow_model(model_name, device=device)
        self.ae = load_ae(model_name, device=device)
        
        # ⭐ 可选：加载文本编码器（如果use_text_condition=True）
        if use_text_condition:
            if qwen_model_path is None:
                qwen_model_path = qwen_ckpt
            self.text_encoder = Qwen3Embedder(
                model_spec=qwen_model_path,
                device=device,
                tokenizer_path=qwen_ckpt,  # 传递 tokenizer 路径
            )
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            print("[FlowCompression] Text encoder loaded and frozen")
        else:
            self.text_encoder = None
            print("[FlowCompression] Text encoder disabled (using empty text condition)")

        # 构建 DiT-IC Codec
        self.codec = self._build_codec(codec_config)
        
        # 加载 ELIC 辅助编码器（参考 DiT-IC 实现）
        if elic_ckpt is not None:
            self._load_elic_aux_encoder(elic_ckpt)
        else:
            self.elic_aux_encoder = None

        # 冻结不需要训练的组件
        self.ae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        self.flux.eval()  # ← 参照Flow项目，Flux初始设为eval模式
        for p in self.ae.parameters():
            p.requires_grad = False
        for p in self.flux.parameters():  # ← 冻结Flux参数（LoRA注入后会重新启用部分参数）
            p.requires_grad = False
        
        # 冻结 ELIC 辅助编码器
        if self.elic_aux_encoder is not None:
            self.elic_aux_encoder.eval()
            for p in self.elic_aux_encoder.parameters():
                p.requires_grad = False

        self._prompt_cache = {}

    def _build_codec(self, config: dict) -> LatentCodec:
        """从配置字典构建 DiT-IC Codec"""
        return LatentCodec(
            ch_emd=config.get("ch_emd", 128),      # Flux AE Main Enc 输出通道 (FLUX.2-klein-4b 是 128)
            channel=config.get("channel", 320),   # ELIC Aux Enc 输出通道
            channel_out=config.get("channel_out", 128), # 匹配 Flux AE Latent 通道
            num_slices=config.get("num_slices", 5),
        )

    def _load_elic_aux_encoder(self, elic_ckpt: str):
        """
        加载 ELIC 辅助编码器（参考 DiT-IC 实现）
        
        Args:
            elic_ckpt: ELIC 模型权重路径
        """
        from .elic_aux_encoder import ELICAuxEncoder
        
        # 创建 ELIC 辅助编码器
        self.elic_aux_encoder = ELICAuxEncoder(N=192, M=320)
        
        # 加载权重
        if os.path.exists(elic_ckpt):
            checkpoint = torch.load(elic_ckpt, map_location="cpu")
            # ELIC checkpoint 通常包含整个模型的 state_dict
            # 我们只需要 g_a (AnalysisTransform) 的部分
            if "g_a" in checkpoint:
                # 如果 checkpoint 是完整模型，提取 g_a
                g_a_state = {k.replace("g_a.", ""): v for k, v in checkpoint.items() if k.startswith("g_a.")}
                self.elic_aux_encoder.load_state_dict(g_a_state, strict=False)
            else:
                # 如果 checkpoint 直接是 g_a 的 state_dict
                self.elic_aux_encoder.load_state_dict(checkpoint, strict=False)
            
            print(f"ELIC auxiliary encoder loaded from: {elic_ckpt}")
        else:
            print(f"Warning: ELIC checkpoint not found at {elic_ckpt}, using random initialization")
        
        # 移动到设备并设置为评估模式
        self.elic_aux_encoder.to(self.device)
        self.elic_aux_encoder.eval()
        for p in self.elic_aux_encoder.parameters():
            p.requires_grad = False

    def enable_gradient_checkpointing(self, enabled: bool = True):
        """
        启用或禁用梯度检查点（Gradient Checkpointing）
        
        Args:
            enabled: 是否启用梯度检查点
        
        说明:
            - 梯度检查点可以显著减少显存占用（约 30-50%）
            - 但会增加计算时间（约 20-30%），因为需要重新计算前向传播
            - 适用于显存受限但训练时间不敏感的场景
            - 只对可训练的模块（Flux + Codec）生效
        """
        if not enabled:
            print("[Gradient Checkpointing] Disabled")
            return
        
        print("[Gradient Checkpointing] Enabling for trainable modules...")
        
        # 1. 对 Flux Transformer 启用梯度检查点
        if hasattr(self.flux, 'gradient_checkpointing_enable'):
            try:
                self.flux.gradient_checkpointing_enable()
                print("  ✓ Flux Transformer: gradient checkpointing enabled")
            except Exception as e:
                print(f"  ✗ Flux Transformer: failed to enable ({e})")
        elif hasattr(self.flux, 'transformer'):
            # 某些实现中 transformer 是子模块
            if hasattr(self.flux.transformer, 'gradient_checkpointing_enable'):
                try:
                    self.flux.transformer.gradient_checkpointing_enable()
                    print("  ✓ Flux Transformer: gradient checkpointing enabled")
                except Exception as e:
                    print(f"  ✗ Flux Transformer: failed to enable ({e})")
        else:
            print("  ⚠ Flux Transformer: gradient checkpointing not supported")
        
        # 2. 对 Codec 启用梯度检查点（如果支持）
        if hasattr(self.codec, 'gradient_checkpointing_enable'):
            try:
                self.codec.gradient_checkpointing_enable()
                print("  ✓ Codec: gradient checkpointing enabled")
            except Exception as e:
                print(f"  ✗ Codec: failed to enable ({e})")
        else:
            print("  ℹ Codec: gradient checkpointing not available (using manual implementation)")
        
        print("[Gradient Checkpointing] Setup complete\n")

    @torch.no_grad()
    def encode_images(self, x01: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        将图像编码为 Main latent (Flux AE)，自动处理尺寸对齐
        """
        h, w = x01.shape[-2], x01.shape[-1]
        required_alignment = 64  # 16 (AE) * 4 (Codec)
        
        pad_h = (required_alignment - h % required_alignment) % required_alignment
        pad_w = (required_alignment - w % required_alignment) % required_alignment
        
        if pad_h > 0 or pad_w > 0:
            x01 = F.pad(x01, (0, pad_w, 0, pad_h), mode='reflect')
        
        x = x01 * 2.0 - 1.0
        z_main = self.ae.encode(x.to(next(self.ae.parameters()).dtype)).float()
        
        pad_info = {
            "original_shape": x01.shape,
            "pad_h": pad_h,
            "pad_w": pad_w,
        }
        
        return z_main, pad_info

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor, pad_info: dict = None) -> torch.Tensor:
        """
        将 latent 解码为图像，如果提供了 pad_info 则去除 padding
        """
        x = self.ae.decode(z.to(next(self.ae.parameters()).dtype)).float()
        x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        
        if pad_info:
            pad_h, pad_w = pad_info["pad_h"], pad_info["pad_w"]
            if pad_h > 0 or pad_w > 0:
                x = x[:, :, :x.shape[2] - pad_h, :x.shape[3] - pad_w]
        
        return x

    @torch.no_grad()
    def get_text_context(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取文本上下文（支持禁用文本条件）
        
        如果use_text_condition=False，返回空的文本条件（全零向量）
        """
        # ⭐ 如果不使用文本条件，返回dummy的空文本条件
        if not self.use_text_condition or self.text_encoder is None:
            # 创建空的文本条件（Flux期望的形状：batch_size, seq_len=1, hidden_dim）
            # 使用全零向量作为空文本条件
            # 从txt_in的权重形状推断输入维度，兼容LoRA包装的情况
            if hasattr(self.flux.txt_in, 'in_features'):
                # 情况1：原始Linear层（未注入LoRA）
                text_dim = self.flux.txt_in.in_features
            elif hasattr(self.flux.txt_in, 'base') and hasattr(self.flux.txt_in.base, 'in_features'):
                # 情况2：LoRALinear包装的情况（通过base访问）
                text_dim = self.flux.txt_in.base.in_features
            else:
                # 情况3：尝试其他可能的属性名
                raise AttributeError(
                    f"Cannot determine text input dimension from txt_in. "
                    f"Type: {type(self.flux.txt_in)}, "
                    f"Available attrs: {[attr for attr in dir(self.flux.txt_in) if not attr.startswith('_')]}"
                )
            
            empty_ctx = torch.zeros(
                batch_size, 1, text_dim, 
                dtype=torch.bfloat16, device=device
            )
            empty_ctx_ids = torch.zeros(batch_size, 1, 4, dtype=torch.long, device=device)
            return empty_ctx, empty_ctx_ids
        
        # ⭐ 如果使用文本条件，全局只编码一次（batch_size=1），然后repeat到目标batch_size
        if not hasattr(self, '_cached_ctx') or not hasattr(self, '_cached_ctx_ids'):
            # 第一次调用时编码单个prompt
            prompts = [FIXED_PROMPT]
            if self.guidance_distilled:
                ctx_single = self.text_encoder(prompts).to(torch.bfloat16)
            else:
                ctx_empty = self.text_encoder([""]).to(torch.bfloat16)
                ctx_prompt = self.text_encoder(prompts).to(torch.bfloat16)
                ctx_single = torch.cat([ctx_empty, ctx_prompt], dim=0)
            
            # 编码并缓存单样本的结果
            ctx_single, ctx_ids_single = batched_prc_txt(ctx_single)
            self._cached_ctx = ctx_single.cpu()  # (1, seq_len, dim)
            self._cached_ctx_ids = ctx_ids_single.cpu()  # (1, seq_len, 4)
        
        # 从缓存中取出并扩展到目标batch_size
        ctx_single = self._cached_ctx.to(device)
        ctx_ids_single = self._cached_ctx_ids.to(device)
        
        # Repeat到目标batch_size
        # ctx: (1, seq_len, dim) -> (batch_size, seq_len, dim)
        ctx = ctx_single.repeat(batch_size, 1, 1)
        # ctx_ids: (1, seq_len, 4) -> (batch_size, seq_len, 4)
        ctx_ids = ctx_ids_single.repeat(batch_size, 1, 1)
        
        return ctx, ctx_ids

    def _latent_to_tokens(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = z.to(torch.bfloat16)
        return batched_prc_img(z)

    def _tokens_to_latent(self, x_tokens: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        z = torch.cat(scatter_ids(x_tokens, x_ids)).squeeze(2)
        return z

    # ==================== 训练接口 ====================
    
    def forward(
        self, 
        x01: torch.Tensor, 
        z_aux: torch.Tensor,
        train_schedule_steps: int = 50,
        codec_mode: str = "kl_mean",  # 保留参数但不再使用
        global_step: int = 0,
    ) -> dict:
        """
        训练模式的前向传播
        
        Args:
            x01: 输入图像 [0, 1] 范围 (B, 3, H, W)
            z_aux: ELIC 辅助编码器输出 (B, 320, H/16, W/16)
            train_schedule_steps: 训练时的时间步数量
            codec_mode: Codec工作模式（保留参数以兼容，但不再影响逻辑）
        
        Returns:
            包含重建结果和码率信息的字典
        """
        batch_size = x01.shape[0]
        device = x01.device

        # 1. 编码得到干净 latent（带 padding）
        z_main, pad_info = self.encode_images(x01)
        
        # 2. DiT-IC Codec 压缩（返回字典，与 Flow 项目的 TCMLatent 保持一致）
        codec_out = self.codec(z_main, z_aux)
        
        # 3. 直接使用量化后的latent（对齐TCMLatent方案，不使用KL分布建模）
        z_tcm = codec_out["x_hat"]  # 直接使用x_hat，不计算KL损失
        z_tcm = z_tcm.to(z_main.dtype)
        
        # 获取残差（用于后续补偿）
        res = codec_out.get("res", None)
        
        # 4. 准备 Flux 输入
        z_tokens, z_ids = self._latent_to_tokens(z_tcm)
        
        # 5. 随机选择一个时间步
        schedule = get_schedule(train_schedule_steps, z_tokens.shape[1])
        schedule_tensor = torch.tensor(schedule, dtype=z_tokens.dtype, device=device)
        step_idx = torch.randint(0, train_schedule_steps, (batch_size,), device=device)
        timesteps = schedule_tensor[step_idx]
        
        # 6. 获取文本上下文
        ctx, ctx_ids = self.get_text_context(batch_size=batch_size, device=device)
        guidance_vec = torch.full(
            (z_tokens.shape[0],), 
            self.guidance, 
            dtype=z_tokens.dtype, 
            device=device,
        )
        
        # 7. Flux 预测速度场
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
        
        # ⭐ 关键修复：添加辅助分支的残差（参考 DiT-IC 实现）
        # res 包含高频细节信息，需要加回到重建的 latent 中
        if res is not None:
            if res.shape[-2:] != z_hat.shape[-2:]:
                # 如果维度不匹配，使用双线性插值对齐
                res_aligned = F.interpolate(res, size=z_hat.shape[-2:], mode='bilinear', align_corners=False)
            else:
                res_aligned = res
            z_hat = z_hat + res_aligned
        
        # 9. 解码并使用 padding 信息去除 padding
        x_hat01 = self.decode_latents(z_hat, pad_info)  # 传入 pad_info 进行裁剪
        
        return {
            "x_hat": x_hat01,
            "likelihoods": codec_out["likelihoods"],  # {"z": ..., "y": ...}
            "z_clean": z_main,
            "z_tcm": z_tcm,
            "res": res,
            # ❌ 移除 kl_loss，对齐TCMLatent方案
        }

    # ==================== 推理接口 ====================
    
    @torch.no_grad()
    def compress(
        self,
        z_main: torch.Tensor,
        z_aux: torch.Tensor,
        do_entropy_coding: bool = True,
    ) -> Dict:
        """
        压缩 latent 表示
        """
        if do_entropy_coding:
            self.codec.update(force=True)
            comp = self.codec.compress(z_main, z_aux)
            return comp
        else:
            out = self.codec(z_main, z_aux)
            return {"x_hat": out["mean"]}

    @torch.no_grad()
    def decompress(
        self,
        strings: List,
        shape: Tuple[int, int],
        infer_steps: int = 4,
        pad_info: dict = None,
    ) -> torch.Tensor:
        """
        解压缩并重建图像
        
        Args:
            strings: 压缩的比特流
            shape: latent 形状
            infer_steps: Flux 去噪步数
            pad_info: padding 信息（可选）
        """
        codec_out = self.codec.decompress(strings, shape)
        z_tcm = codec_out["x_hat"]  # 使用字典访问
        
        # Flux 多步去噪
        z_tcm_tokens, z_ids = self._latent_to_tokens(z_tcm)
        ctx, ctx_ids = self.get_text_context(batch_size=z_tcm.shape[0], device=self.device)
        
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
        x_hat01 = self.decode_latents(z_out, pad_info)  # 传入 pad_info 进行裁剪
        
        return x_hat01

    @torch.no_grad()
    def forward_stage1_infer(
        self,
        x01: torch.Tensor,
        infer_steps: int = 4,
        do_entropy_coding: bool = True,
    ) -> Dict[str, any]:
        """
        推理模式的前向传播（与 Flow 项目的 forward_stage1_infer 对齐）
        
        使用真实的熵编码 + 多步去噪流程，准确评估推理性能。
        
        Args:
            x01: 输入图像 [0, 1] 范围 (B, 3, H, W)
            infer_steps: Flux 去噪步数（默认 4）
            do_entropy_coding: 是否执行真正的熵编码（默认 True）
        
        Returns:
            {
                "x_hat": 重建图像 (B, 3, H, W),
                "bytes": 每个样本的字节数列表,
            }
        """
        batch_size = x01.shape[0]
        device = x01.device

        # 1. 编码图像，获取 padding 信息
        z_main, pad_info = self.encode_images(x01)
        
        # 2. 获取 ELIC 辅助特征
        if self.elic_aux_encoder is not None:
            z_aux = self.elic_aux_encoder(x01)
        else:
            # 如果没有 ELIC 编码器，使用 dummy z_aux
            z_aux = torch.zeros(
                z_main.shape[0], 320, 
                z_main.shape[2], z_main.shape[3],
                device=z_main.device, dtype=z_main.dtype
            )

        total_bytes = []
        z_tcm_list = []
        res_list = []  # ⭐ 初始化残差列表
        
        if do_entropy_coding:
            # 3a. 真正的熵编码压缩 + 解压（逐张处理）
            self.codec.update(force=True)
            for i in range(batch_size):
                zi = z_main[i : i + 1]
                z_aux_i = z_aux[i : i + 1]
                
                # 压缩
                comp = self.codec.compress(zi, z_aux_i)
                
                # 解压
                dec = self.codec.decompress(comp["strings"], comp["shape"])
                zi_hat = dec["x_hat"]
                res_i = dec["res"]  # ⭐ 提取残差
                
                # 计算字节数
                y_bytes = sum(len(s) for s in comp["strings"][0])  # hyper prior
                z_bytes = sum(len(s) for s in comp["strings"][1])  # main latent
                total_bytes.append(float(y_bytes + z_bytes))
                
                z_tcm_list.append(zi_hat)
                res_list.append(res_i)  # ⭐ 保存残差
            
            z_tcm = torch.cat(z_tcm_list, dim=0)
            res_batch = torch.cat(res_list, dim=0)  # ⭐ 拼接残差
        else:
            # 3b. 不使用熵编码，直接前向传播
            codec_out = self.codec(z_main, z_aux)
            z_tcm = codec_out["x_hat"]
            res_batch = codec_out["res"]  # ⭐ 提取残差
            total_bytes = [0.0] * batch_size

        # 4. 准备 Flux 输入
        z_tcm_tokens, z_ids = self._latent_to_tokens(z_tcm)
        ctx, ctx_ids = self.get_text_context(batch_size=batch_size, device=device)

        # 5. Flux 多步去噪
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
        
        # ⭐ 关键修复：添加辅助分支的残差（参考 DiT-IC 实现）
        if res_batch.shape[-2:] != z_out.shape[-2:]:
            res_aligned = F.interpolate(res_batch, size=z_out.shape[-2:], mode='bilinear', align_corners=False)
        else:
            res_aligned = res_batch
        z_out = z_out + res_aligned
        
        # 6. 解码并使用 padding 信息去除 padding
        x_hat01 = self.decode_latents(z_out, pad_info)
        
        # 调试日志（用于诊断eval全白问题，仅在主进程输出一次）
        is_main_process = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
        
        # ⭐ 使用实例变量确保只输出一次
        if not hasattr(self, '_eval_debug_logged'):
            self._eval_debug_logged = False
        
        if is_main_process and not self._eval_debug_logged:
            self._eval_debug_logged = True
            print(f"\n[DEBUG EVAL]")
            print(f"  Input x01 range: [{x01.min():.4f}, {x01.max():.4f}], mean={x01.mean():.4f}")
            print(f"  z_main range: [{z_main.min():.4f}, {z_main.max():.4f}], std={z_main.std():.4f}")
            print(f"  z_tcm (after codec) range: [{z_tcm.min():.4f}, {z_tcm.max():.4f}], std={z_tcm.std():.4f}")
            if res_batch is not None:
                print(f"  res_batch range: [{res_batch.min():.4f}, {res_batch.max():.4f}], mean={res_batch.mean():.4f}")
            print(f"  z_out (after Flux denoise) range: [{z_out.min():.4f}, {z_out.max():.4f}], std={z_out.std():.4f}")
            print(f"  Output x_hat range: [{x_hat01.min():.4f}, {x_hat01.max():.4f}], mean={x_hat01.mean():.4f}")
            print(f"  Total bytes: {total_bytes}")

        return {
            "x_hat": x_hat01,
            "bytes": total_bytes,
        }
