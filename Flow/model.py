"""
FLUXTCM Model - 统一的模型架构定义

整合 FLUX.2 + TCM 的完整流程，提供训练和推理两种模式的清晰接口。

主要功能:
- forward(): 训练模式的前向传播
- compress(): 推理时的压缩操作
- decompress(): 推理时的解压缩操作
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

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


class FluxTCMModel(nn.Module):
    """
    FLUXTCM 模型主类
    
    整合了 FLUX.2 + TCM 的完整流程，提供训练和推理两种模式
    
    Args:
        model_name: 模型名称 (如 "flux.2-klein-4b")
        flux_ckpt: FLUX 模型权重路径
        ae_ckpt: AutoEncoder 权重路径
        qwen_ckpt: Qwen 文本编码器路径 (可选)
        tcm_config: TCM 配置参数字典
        device: 运行设备
        guidance: 引导系数 (默认 1.0)
    
    Example:
        >>> # 初始化模型
        >>> model = FluxTCMModel(
        ...     model_name="flux.2-klein-4b",
        ...     flux_ckpt="/path/to/flux.safetensors",
        ...     ae_ckpt="/path/to/ae.safetensors",
        ...     tcm_config=tcm_args,
        ...     device=torch.device("cuda"),
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
        qwen_ckpt: Optional[str],
        tcm_config: dict,
        device: torch.device,
        guidance: float = 1.0,
    ):
        super().__init__()
        model_info = FLUX2_MODEL_INFO[model_name]
        os.environ[model_info["model_path"]] = flux_ckpt
        os.environ["AE_MODEL_PATH"] = ae_ckpt

        self.model_name = model_name
        self.guidance = guidance
        self.guidance_distilled = bool(model_info.get("guidance_distilled", True))
        self.device = device

        # 加载核心组件
        self.flux = load_flow_model(model_name, device=device)
        self.ae = load_ae(model_name, device=device)
        if qwen_ckpt:
            self.text_encoder = Qwen3Embedder(model_spec=qwen_ckpt, device=device)
        else:
            self.text_encoder = load_text_encoder(model_name, device=device)

        # 从配置构建 TCM 模型
        self.tcm = self._build_tcm(tcm_config)

        # 冻结不需要训练的组件
        self.ae.eval()
        self.text_encoder.eval()
        for p in self.ae.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self._prompt_cache = {}
    
    def _build_tcm(self, config: dict) -> TCMLatent:
        """从配置字典构建 TCM 模型"""
        return TCMLatent(
            in_channels=config.get("tcm_in_channels", 16),
            out_channels=config.get("tcm_out_channels", 16),
            N=config.get("tcm_N", 192),
            M=config.get("tcm_M", 192),
            num_slices=config.get("tcm_num_slices", 12),
            max_support_slices=config.get("tcm_max_support_slices", 6),
            ga_config=config.get("tcm_ga_config", [[16, 8, 8], [16, 8, 8], [16, 8, 8]]),
            gs_config=config.get("tcm_gs_config", [[16, 8, 8], [16, 8, 8], [16, 8, 8]]),
            ha_config=config.get("tcm_ha_config", [[16, 8, 8], [16, 8, 8], [16, 8, 8]]),
            hs_config=config.get("tcm_hs_config", [[16, 8, 8], [16, 8, 8], [16, 8, 8]]),
            ga_head_dim=config.get("tcm_ga_head_dim", 32),
            gs_head_dim=config.get("tcm_gs_head_dim", 32),
            window_size=config.get("tcm_window_size", 8),
            atten_window_size=config.get("tcm_atten_window_size", 4),
            drop_path_rate=config.get("tcm_drop_path_rate", 0.0),
        )

    @torch.no_grad()
    def encode_images(self, x01: torch.Tensor) -> torch.Tensor:
        """
        将图像编码为 latent 表示
        
        Args:
            x01: [0, 1] 范围的图像张量 (B, 3, H, W)
        
        Returns:
            latent 张量 (B, C, H', W')
        """
        x = x01 * 2.0 - 1.0
        z = self.ae.encode(x.to(next(self.ae.parameters()).dtype))
        return z.float()

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """
        将 latent 解码为图像
        
        Args:
            z: latent 张量 (B, C, H', W')
        
        Returns:
            [0, 1] 范围的图像张量 (B, 3, H, W)
        """
        x = self.ae.decode(z.to(next(self.ae.parameters()).dtype)).float()
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def get_text_context(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取文本条件上下文
        
        Args:
            batch_size: batch size
        
        Returns:
            ctx: 文本特征
            ctx_ids: 文本 token IDs
        """
        if batch_size in self._prompt_cache:
            ctx, ctx_ids = self._prompt_cache[batch_size]
            return ctx.to(self.device), ctx_ids.to(self.device)

        prompts = [FIXED_PROMPT] * batch_size
        if self.guidance_distilled:
            ctx = self.text_encoder(prompts).to(torch.bfloat16)
        else:
            ctx_empty = self.text_encoder([""] * batch_size).to(torch.bfloat16)
            ctx_prompt = self.text_encoder(prompts).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)

        ctx, ctx_ids = batched_prc_txt(ctx)
        self._prompt_cache[batch_size] = (ctx.cpu(), ctx_ids.cpu())
        return ctx, ctx_ids

    def _latent_to_tokens(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将 latent 转换为 token 序列"""
        z = z.to(torch.bfloat16)
        return batched_prc_img(z)

    def _tokens_to_latent(self, x_tokens: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        """将 token 序列转换回 latent"""
        z = torch.cat(scatter_ids(x_tokens, x_ids)).squeeze(2)
        return z

    def sample_sigmas(
        self, 
        image_seq_len: int, 
        batch_size: int, 
        steps: int,
    ) -> torch.Tensor:
        """
        采样时间步 sigma
        
        Args:
            image_seq_len: 图像序列长度
            batch_size: batch size
            steps: 调度步数
        
        Returns:
            sigma 值 (batch_size,)
        """
        schedule = get_schedule(steps, image_seq_len)
        sched = torch.tensor(schedule, dtype=torch.float32, device=self.device)
        idx = torch.randint(0, sched.shape[0], (batch_size,), device=self.device)
        return sched[idx]

    # ==================== 训练接口 ====================
    
    def forward(
        self,
        x01: torch.Tensor,
        train_schedule_steps: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """
        训练模式的前向传播（参照 DiT-IC + FLUX.2 原生调度）
        
        核心策略：
        - 直接使用压缩特征 z_tcm 作为 Flux 输入（不与 z_clean 混合）
        - 使用 FLUX.2 原生的多步时间步调度（与推理一致）
        - 随机选择一个时间步进行单步训练（Flow Matching 标准做法）
        
        Args:
            x01: 输入图像 [0, 1] 范围 (B, 3, H, W)
            train_schedule_steps: 训练时的时间步数量
        
        Returns:
            包含以下键的字典:
            - x_hat: 重建图像 (B, 3, H, W)
            - likelihoods: TCM 的概率估计
            - z_clean: 干净的 latent
            - z_tcm: TCM 压缩后的 latent
            - sigma: 时间步
        """
        from flux2.sampling import get_schedule
        
        batch_size = x01.shape[0]
        device = x01.device

        # 1. 编码得到干净 latent
        z_clean = self.encode_images(x01)
        
        # 2. TCM 压缩（使用 ste_round 量化）
        tcm_out = self.tcm(z_clean)
        z_tcm = tcm_out["x_hat"]
        z_tcm = z_tcm.to(z_clean.dtype)

        # 3. 准备 Flux 输入：直接使用 z_tcm（不混合 z_clean）
        z_tokens, z_ids = self._latent_to_tokens(z_tcm)
        
        # 4. 获取 FLUX.2 原生的多步时间步调度（与推理一致）
        schedule = get_schedule(train_schedule_steps, z_tokens.shape[1])
        schedule_tensor = torch.tensor(schedule, dtype=z_tokens.dtype, device=device)
        
        # 5. 随机选择一个时间步索引
        step_idx = torch.randint(0, train_schedule_steps, (batch_size,), device=device)
        timesteps = schedule_tensor[step_idx]  # 形状: [batch_size]
        
        # 6. 获取文本上下文
        ctx, ctx_ids = self.get_text_context(batch_size=batch_size, device=device)
        guidance_vec = torch.full(
            (z_tokens.shape[0],), 
            self.guidance, 
            dtype=z_tokens.dtype, 
            device=device,
        )
        
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

        return {
            "x_hat": x_hat01,
            "likelihoods": tcm_out["likelihoods"],
            "z_clean": z_clean,
            "z_tcm": z_tcm,
            "sigma": timesteps,
        }

    # ==================== 推理接口 ====================
    
    @torch.no_grad()
    def compress(
        self,
        z_clean: torch.Tensor,
        do_entropy_coding: bool = True,
    ) -> Dict:
        """
        压缩 latent 表示
        
        Args:
            z_clean: 干净的 latent 张量 (B, C, H', W')
            do_entropy_coding: 是否使用熵编码（计算真实 bpp）
        
        Returns:
            包含压缩结果的字典:
            - strings: 压缩后的比特流
            - shape: 形状信息
            - x_hat: 解压后的 latent
            - bytes: 每个样本的字节数列表
        """
        batch_size = z_clean.shape[0]
        total_bytes = []
        z_tcm_list = []
        
        if do_entropy_coding:
            # 使用熵编码获取真实压缩率
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
            strings = [comp["strings"] for comp in [self.tcm.compress(z_clean[i:i+1]) for i in range(batch_size)]]
            shapes = [comp["shape"] for comp in [self.tcm.compress(z_clean[i:i+1]) for i in range(batch_size)]]
        else:
            # 不使用熵编码，仅用于快速评估
            out = self.tcm(z_clean)
            z_tcm = out["x_hat"]
            total_bytes = [0.0] * batch_size
            strings = None
            shapes = None
        
        return {
            "strings": strings,
            "shape": shapes,
            "x_hat": z_tcm,
            "bytes": total_bytes,
        }

    @torch.no_grad()
    def decompress(
        self,
        strings: List,
        shapes: List,
        infer_steps: int = 4,
    ) -> torch.Tensor:
        """
        解压缩并重建图像
        
        Args:
            strings: 压缩比特流列表
            shapes: 形状信息列表
            infer_steps: 去噪步数
        
        Returns:
            重建图像 [0, 1] 范围 (B, 3, H, W)
        """
        assert strings is not None and shapes is not None, "需要提供压缩数据才能解压"
        
        batch_size = len(strings)
        z_tcm_list = []
        
        # 逐帧解压
        for i in range(batch_size):
            dec = self.tcm.decompress(strings[i], shapes[i])
            z_tcm_list.append(dec["x_hat"])
        
        z_tcm = torch.cat(z_tcm_list, dim=0)
        
        # Flux 多步去噪
        z_tcm_tokens, z_ids = self._latent_to_tokens(z_tcm)
        ctx, ctx_ids = self.get_text_context(batch_size=batch_size)
        
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
        x_hat01 = self.decode_latents(z_out)
        
        return x_hat01

    @torch.no_grad()
    def forward_infer(
        self,
        x01: torch.Tensor,
        infer_steps: int = 4,
        do_entropy_coding: bool = True,
    ) -> Dict:
        """
        推理模式的一体化处理（压缩 + 解压）
        
        Args:
            x01: 输入图像 [0, 1] 范围 (B, 3, H, W)
            infer_steps: 去噪步数
            do_entropy_coding: 是否使用熵编码
        
        Returns:
            包含重建结果和压缩信息的字典
        """
        z_clean = self.encode_images(x01)
        compressed = self.compress(z_clean, do_entropy_coding=do_entropy_coding)
        x_hat01 = self.decompress(
            compressed["strings"],
            compressed["shapes"],
            infer_steps=infer_steps,
        )
        
        return {
            "x_hat": x_hat01,
            "bytes": compressed["bytes"],
        }