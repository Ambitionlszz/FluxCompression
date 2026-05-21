"""
Evaluator Module - For evaluation and visualization during training.
"""

import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .utils import AverageMeter, ensure_dir


class Stage1Evaluator:
    """
    Stage1 Training Evaluator.
    """
    
    def __init__(self, output_dir: str, eval_batches: int = 5, accelerator=None):
        self.output_dir = output_dir
        self.eval_batches = eval_batches
        self.accelerator = accelerator if accelerator is not None else SimpleNamespace(is_main_process=True)
        
        # Lazy initialized metrics
        self._lpips_metric = None
        self._dists_metric = None
 
    def _init_metrics(self, device: torch.device):
        """Lazy initialize metrics."""
        if self._lpips_metric is None:
            import lpips
            from .losses import DISTSMetric
            
            self._lpips_metric = lpips.LPIPS(net="vgg").to(device).eval()
            self._dists_metric = DISTSMetric().to(device).eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        pipeline,
        criterion,
        val_loader: DataLoader,
        global_step: int,
    ) -> Dict[str, float]:
        """
        Perform model evaluation (4-step multistep denoise + entropy coding).
        """
        pipeline.codec.eval()
        pipeline.flux.eval()

        meters = {
            "loss": AverageMeter(),
            "bpp": AverageMeter(),
            "mse": AverageMeter(),
            "psnr": AverageMeter(),
            "lpips": AverageMeter(),
            "dists": AverageMeter(),
        }

        # Create evaluation image directory
        if self.accelerator.is_main_process:
            eval_image_dir = self._get_eval_image_dir(global_step)
            print(f"Saving multistep evaluation images to: {eval_image_dir}")
            ensure_dir(eval_image_dir)
 
        # Randomly select batch indices to save
        save_batch_indices = self._sample_batch_indices(len(val_loader))
 
        # Initialize metrics
        self._init_metrics(self.accelerator.device)

        for i, batch in enumerate(val_loader):
            if i >= self.eval_batches:
                break
            
            with self.accelerator.autocast():
                # Multistep inference mode
                out = pipeline.forward_stage1_infer(batch, infer_steps=4, do_entropy_coding=True)
                loss_dict = self._compute_multistep_metrics(batch, out)
 
            bs = batch.shape[0]
            for k, meter in meters.items():
                meter.update(loss_dict[k].item(), bs)
             
            # Save comparison grid for selected batches
            if self.accelerator.is_main_process and i in save_batch_indices:
                self._save_comparison_images(
                    batch, out["x_hat"], 
                    eval_image_dir, i
                )
 
        # Resume training mode
        pipeline.codec.train()
        pipeline.flux.train()
 
        # Aggregate and return results
        return self._reduce_metrics(meters)
    
    def _get_eval_image_dir(self, global_step: int) -> str:
        """Get evaluation image save directory."""
        subdir = f"eval_images_multistep/step_{global_step:08d}"
        return os.path.join(self.output_dir, subdir)
     
    def _sample_batch_indices(self, total_batches: int, n_samples: int = 5) -> set:
        """Sample random batch indices."""
        indices = set()
        max_idx = min(total_batches, self.eval_batches) - 1
        while len(indices) < min(n_samples, max_idx + 1):
            idx = random.randint(0, max_idx)
            indices.add(idx)
        return sorted(list(indices))
    
    def _compute_multistep_metrics(self, batch: torch.Tensor, out: dict) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics for multistep mode.
        """
        n, _, h, w = batch.shape
        num_pixels = n * h * w
         
        # Compute BPP
        if "bytes" in out and out["bytes"]:
            total_bytes = sum(out["bytes"])
            bpp = torch.tensor(total_bytes * 8.0 / num_pixels, device=batch.device)
        else:
            bpp = torch.tensor(0.0, device=batch.device)
         
        # Compute other quality metrics
        mse = F.mse_loss(out["x_hat"], batch)
        psnr = -10 * torch.log10(torch.clamp(mse, min=1e-8))
        lpips_loss = self._lpips_metric(out["x_hat"] * 2.0 - 1.0, batch * 2.0 - 1.0).mean()
        dists_loss = self._dists_metric(batch, out["x_hat"])
         
        # Return individual metrics, no weighted loss
        loss = mse  # Use MSE as the primary evaluation loss metric
        
        return {
            "loss": loss,
            "bpp": bpp,
            "mse": mse,
            "psnr": psnr,
            "lpips": lpips_loss,
            "dists": dists_loss,
        }
    
    def _save_comparison_images(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        save_dir: str,
        batch_idx: int,
        n_images: int = 4,
    ):
        """
        Save original vs reconstructed comparison grid.
        """
        images = original[:n_images]
        reconstructions = reconstructed[:n_images]
         
        # Comparison grid: top row original, bottom row reconstructed
        comparison = torch.cat([images, reconstructions], dim=0)
        grid = make_grid(comparison, nrow=n_images, padding=2, pad_value=1.0)
         
        # Generate filename
        filename = f"batch_{batch_idx:03d}_multistep_comparison.png"
        save_path = os.path.join(save_dir, filename)
         
        # Save image
        torchvision.utils.save_image(grid, save_path)
     
    def _reduce_metrics(self, meters: Dict[str, AverageMeter]) -> Dict[str, float]:
        """
        Aggregate evaluation metrics across GPUs.
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
