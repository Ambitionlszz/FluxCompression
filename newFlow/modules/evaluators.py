"""
评估器模块 - 用于 newFlow 模型训练过程中的评估和可视化

提供多步评估模式,验证真实推理性能(使用 DiT-IC Codec + Flux Transformer 去噪)。

注意: newFlow 使用 ELIC Aux Encoder,评估时需要提供 z_aux。
"""

import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .utils import AverageMeter, ensure_dir


class SimpleNamespace:
    """简单的命名空间类,用于模拟 Accelerator"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def is_main_process(self):
        return getattr(self, '_is_main', True)


class Stage1Evaluator:
    """
    newFlow Stage1 训练评估器
    
    评估 FlowCompression 模型的多步去噪重建质量,包括:
    - BPP (码率)
    - MSE (像素级重建)
    - LPIPS (感知质量)
    - CLIP L2 (语义一致性)
    
    Args:
        output_dir: 输出目录
        eval_batches: 评估的 batch 数量
        accelerator: Accelerate Accelerator 对象(可选)
    """
    
    def __init__(self, output_dir: str, eval_batches: int = 5, accelerator=None):
        self.output_dir = output_dir
        self.eval_batches = eval_batches
        self.accelerator = accelerator if accelerator is not None else SimpleNamespace(_is_main=True)
        
        # 延迟初始化的评估指标
        self._lpips_metric = None
        self._clip_metric = None

    def _init_metrics(self, clip_ckpt: str, device: torch.device):
        """
        延迟初始化评估指标
        
        Args:
            clip_ckpt: CLIP 模型路径
            device: 计算设备
        """
        if self._lpips_metric is None:
            import lpips
            from .losses import CLIPL2Loss
            
            self._lpips_metric = lpips.LPIPS(net="vgg").to(device).eval()
            self._clip_metric = CLIPL2Loss(clip_ckpt).to(device).eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        pipeline,
        criterion,
        val_loader: DataLoader,
        global_step: int,
        clip_ckpt: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        执行模型评估（4 步多步去噪 + 熵编码）
        
        Args:
            pipeline: FlowCompression 模型实例（兼容 FlowTCMStage1Pipeline 接口）
            criterion: 损失函数（未使用，保留接口兼容性）
            val_loader: 验证集 DataLoader
            global_step: 当前训练步数
            clip_ckpt: CLIP 模型路径
        
        Returns:
            评估指标字典 {"loss": ..., "bpp": ..., "mse": ..., "lpips": ..., "clip_l2": ...}
        """
        # 兼容两种架构：newFlow (model) 和 Flow (pipeline)
        if hasattr(pipeline, 'codec'):
            # newFlow 架构：FlowCompression 模型
            model = pipeline
            model.codec.eval()
            model.flux.eval()
        else:
            # Flow 架构：FlowTCMStage1Pipeline
            model = pipeline
            if hasattr(model, 'tcm'):
                model.tcm.eval()
            if hasattr(model, 'flux'):
                model.flux.eval()

        meters = {
            "loss": AverageMeter(),
            "bpp": AverageMeter(),
            "mse": AverageMeter(),
            "psnr": AverageMeter(),
            "lpips": AverageMeter(),
            "clip_l2": AverageMeter(),
        }

        # 创建评估图像保存目录
        if hasattr(self.accelerator, 'is_main_process') and self.accelerator.is_main_process:
            eval_image_dir = self._get_eval_image_dir(global_step)
            print(f"Saving multistep evaluation images to: {eval_image_dir}")
            ensure_dir(eval_image_dir)

        # 随机选择要保存的 batch 索引（增加多样性）
        save_batch_indices = self._sample_batch_indices(len(val_loader))

        # 初始化 metrics
        if clip_ckpt:
            self._init_metrics(clip_ckpt, next(model.parameters()).device)

        for i, batch in enumerate(val_loader):
            if i >= self.eval_batches:
                break
            
            x01 = batch.to(next(model.parameters()).device)
            
            # 根据模型架构选择不同的推理方式
            if hasattr(model, 'codec'):
                # newFlow 架构：使用真正的推理逻辑（熵编码 + 多步去噪）
                with torch.no_grad():
                    # 调用真正的推理方法（ELIC编码器已集成到模型内部）
                    out = model.forward_stage1_infer(
                        x01, 
                        infer_steps=4, 
                        do_entropy_coding=True
                    )
                    x_hat = out["x_hat"]
                    
                    # 计算评估指标
                    loss_dict = self._compute_multistep_metrics(batch, out)
            else:
                # Flow 架构：使用 pipeline 的推理方法
                with self.accelerator.autocast():
                    # 多步推理模式（真实重建效果 + 熵编码）
                    out = model.forward_stage1_infer(batch, infer_steps=4, do_entropy_coding=True)
                    loss_dict = self._compute_multistep_metrics(batch, out)
                    x_hat = out["x_hat"]

            bs = batch.shape[0]
            for k, meter in meters.items():
                meter.update(loss_dict[k].item(), bs)
            
            # 保存随机选中的 batch 的重建对比图
            if hasattr(self.accelerator, 'is_main_process') and self.accelerator.is_main_process and i in save_batch_indices:
                self._save_comparison_images(
                    x01, x_hat, 
                    eval_image_dir, i
                )

        # 恢复训练模式
        if hasattr(model, 'codec'):
            model.codec.train()
            model.flux.train()
        else:
            if hasattr(model, 'tcm'):
                model.tcm.train()
            if hasattr(model, 'flux'):
                model.flux.train()

        # 汇总并返回结果
        return self._reduce_metrics(meters)
    
    def _get_eval_image_dir(self, global_step: int) -> str:
        """获取评估图像保存目录"""
        subdir = f"eval_images/step_{global_step:08d}"
        return os.path.join(self.output_dir, subdir)
    
    def _sample_batch_indices(self, total_batches: int, n_samples: int = 5) -> list:
        """
        随机采样 batch 索引
        
        Args:
            total_batches: 总 batch 数量
            n_samples: 采样数量
        
        Returns:
            排序后的索引列表
        """
        indices = set()
        max_idx = min(total_batches, self.eval_batches) - 1
        while len(indices) < min(n_samples, max_idx + 1):
            idx = random.randint(0, max_idx)
            indices.add(idx)
        return sorted(list(indices))
    
    def _compute_metrics(self, batch: torch.Tensor, x_hat: torch.Tensor, likelihoods: dict) -> Dict[str, torch.Tensor]:
        """
        计算评估指标（训练模式，使用 likelihoods）
        
        Args:
            batch: 原始图像 (B, 3, H, W)
            x_hat: 重建图像 (B, 3, H, W)
            likelihoods: 熵模型似然字典
        
        Returns:
            指标字典
        """
        n, _, h, w = batch.shape
        num_pixels = n * h * w
        
        # 计算 BPP
        bpp = self._compute_bpp(likelihoods, num_pixels)
        
        # 计算 MSE
        mse = F.mse_loss(x_hat, batch)
        
        # 计算 PSNR
        if mse > 0:
            psnr = 10 * torch.log10(1.0 / mse)
        else:
            psnr = torch.tensor(float('inf'), device=batch.device)
        
        # 计算 LPIPS (需要 [-1, 1] 范围)
        lpips_loss = self._lpips_metric(x_hat * 2.0 - 1.0, batch * 2.0 - 1.0).mean()
        
        # 计算 CLIP L2
        clip_l2 = self._clip_metric(batch, x_hat)
        
        # 返回 loss（用于训练监控）
        loss = mse
        
        return {
            "loss": loss,
            "bpp": bpp,
            "mse": mse,
            "psnr": psnr,
            "lpips": lpips_loss,
            "clip_l2": clip_l2,
        }
    
    def _compute_multistep_metrics(self, batch: torch.Tensor, out: dict) -> Dict[str, torch.Tensor]:
        """
        计算多步模式的评估指标（推理模式，使用 bytes）
        
        多步模式下如果执行了熵编码，可以计算真实的 bpp 和总损失
        
        Args:
            batch: 原始图像 (B, 3, H, W)
            out: forward_stage1_infer 的输出字典
        
        Returns:
            指标字典
        """
        n, _, h, w = batch.shape
        num_pixels = n * h * w
        
        # 计算 bpp（如果有 bytes 信息）
        if "bytes" in out and out["bytes"]:
            total_bytes = sum(out["bytes"])
            bpp = torch.tensor(total_bytes * 8.0 / num_pixels, device=batch.device)
        else:
            bpp = torch.tensor(0.0, device=batch.device)
        
        # 计算其他质量指标
        x_hat = out["x_hat"]
        mse = F.mse_loss(x_hat, batch)
        
        # 计算 PSNR
        if mse > 0:
            psnr = 10 * torch.log10(1.0 / mse)
        else:
            psnr = torch.tensor(float('inf'), device=batch.device)
        
        lpips_loss = self._lpips_metric(x_hat * 2.0 - 1.0, batch * 2.0 - 1.0).mean()
        clip_l2 = self._clip_metric(batch, x_hat)
        
        # 评估器只返回独立的质量指标，不计算加权损失
        # 损失函数的权重配置属于训练逻辑，不应出现在评估器中
        loss = mse  # 使用 MSE 作为主要的评估损失指标
        
        return {
            "loss": loss,
            "bpp": bpp,
            "mse": mse,
            "psnr": psnr,
            "lpips": lpips_loss,
            "clip_l2": clip_l2,
        }
    
    def _compute_bpp(self, likelihoods: dict, num_pixels: int) -> torch.Tensor:
        """
        计算 BPP (Bits Per Pixel)
        
        Args:
            likelihoods: 熵模型输出的似然字典
            num_pixels: 总像素数
        
        Returns:
            BPP 值
        """
        import math
        total = 0.0
        eps = 1e-9
        for v in likelihoods.values():
            total = total + torch.log(v + eps).sum() / (-math.log(2) * num_pixels)
        return torch.tensor(total, device=next(iter(likelihoods.values())).device)
    
    def _save_comparison_images(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        save_dir: str,
        batch_idx: int,
        n_images: int = 4,
    ):
        """
        保存原图和重建图的对比网格
        
        Args:
            original: 原始图像 batch [B, C, H, W]
            reconstructed: 重建图像 batch [B, C, H, W]
            save_dir: 保存目录
            batch_idx: batch 索引
            n_images: 每个 batch 保存的图片数量
        """
        images = original[:n_images]
        reconstructions = reconstructed[:n_images]
        
        # 创建对比网格:上排原图,下排重建图
        comparison = torch.cat([images, reconstructions], dim=0)
        grid = make_grid(comparison, nrow=n_images, padding=2, pad_value=1.0)
        
        # 生成文件名
        filename = f"batch_{batch_idx:03d}_comparison.png"
        save_path = os.path.join(save_dir, filename)
        
        # 保存图片
        torchvision.utils.save_image(grid, save_path)
    
    def _reduce_metrics(self, meters: Dict[str, AverageMeter]) -> Dict[str, float]:
        """
        汇总多 GPU 的评估指标
        
        Returns:
            平均后的指标字典
        """
        reduced = {}
        for k, meter in meters.items():
            tensor = torch.tensor(
                [meter.sum, meter.count], 
                device=self.accelerator.device, 
                dtype=torch.float64
            )
            tensor = self.accelerator.reduce(tensor, reduction="sum")
            reduced[k] = (tensor[0] / max(tensor[1], 1)).item()
        return reduced
