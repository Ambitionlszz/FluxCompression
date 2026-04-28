"""Training script for TCMLatent with FLUX.2 VAE pipeline.

Supports distributed multi-GPU training via HuggingFace Accelerate.

Usage:
    # Single GPU
    python train_tcm.py

    # Multi-GPU (e.g., 4 GPUs)
    accelerate launch --num_processes=4 train_tcm.py

    # Multi-GPU with config
    accelerate launch --config_file accelerate_config.yaml train_tcm.py
"""

import datetime
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import lpips
import logging

# ============================================================
# PATH CONFIGURATION (modify these directly)
# ============================================================
DATASET_PATH = "/data2/luosheng/code/flux2/datasets/train"
TEST_DATASET_PATH = "/data2/luosheng/data/Datasets/Kodak"
SAVE_PATH = "./results/checkpoints/tcm_latent/"
TENSORBOARD_PATH = "./results/tensorboard_logs/tcm/"
VAE_CHECKPOINT = "/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors"
TCM_CHECKPOINT = None                          # Resume from TCM checkpoint, None = train from scratch

# ============================================================
# MODEL CONFIGURATION
# ============================================================
VAE_MODEL_NAME = "flux.2-klein-4b"     # FLUX.2 variant
IN_CHANNELS = 128                       # VAE latent channels
OUT_CHANNELS = 128                      # TCM output channels (= VAE latent channels)
N = 128                                 # TCM intermediate feature channels
M = 320                                 # TCM bottleneck channels
NUM_SLICES = 5                          # Entropy model slice count
MAX_SUPPORT_SLICES = 5                  # Max context slices
GA_CONFIG = [2]                         # g_a ConvTransBlock counts per stage
GS_CONFIG = [2]                         # g_s ConvTransBlock counts per stage
HA_CONFIG = [2]                         # h_a ConvTransBlock counts
HS_CONFIG = [2]                         # h_s ConvTransBlock counts
GA_HEAD_DIM = [8]                       # g_a attention head dimensions
GS_HEAD_DIM = [8]                       # g_s attention head dimensions
WINDOW_SIZE = 8                         # Swin Transformer window size for g_a/g_s
ATTEN_WINDOW_SIZE = 4                   # Window size for SWAtten (must <= y spatial dims)
DROP_PATH_RATE = 0.0                    # DropPath rate

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
EPOCHS = 100                             # Total training epochs
BATCH_SIZE = 8                          # Training batch size (per GPU)
TEST_BATCH_SIZE = 8                     # Test batch size (per GPU)
LEARNING_RATE = 1e-4                    # Main optimizer LR
AUX_LEARNING_RATE = 1e-4                # Auxiliary loss LR
LAMBDA = 0.001                            # Rate-distortion trade-off (lambda)
LOSS_TYPE = "mse"                       # "mse" or "ms-ssim"
LATENT_LOSS_WEIGHT = 0.0                # Latent-domain MSE aux loss weight (0 = disabled)
LPIPS_WEIGHT = 0.5                      # LPIPS loss weight (adjust as needed, 0 = disabled)
PIXEL_LOSS = True                       # Compute distortion in pixel domain (requires VAE decode)
CLIP_MAX_NORM = 1.0                     # Gradient clipping max norm
PATCH_SIZE = (256, 256)                 # Training crop size (must be multiple of 256)
NUM_WORKERS = 4                         # DataLoader workers
SEED = 100                              # Random seed
CONTINUE_TRAIN = True                   # Continue from checkpoint epoch
SKIP_EPOCH = 0                          # Skip N epochs at start

# ============================================================
# ACCELERATE CONFIGURATION
# ============================================================
MIXED_PRECISION = "no"                  # "no", "fp16", or "bf16" (for TCM; VAE always bf16)
GRADIENT_ACCUMULATION_STEPS = 1         # Accumulate gradients over N steps

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
LOG_IMAGE_INTERVAL = 5                  # Log images to TB every N epochs
LOG_TRAIN_INTERVAL = 500                # Print train stats every N iterations

# ============================================================
# End of configuration
# ============================================================

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models import TCMLatent
from tcm_pipeline import LatentCompressionPipeline

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class RecursiveImageFolder(Dataset):
    """Image dataset that recursively searches subdirectories (follows symlinks)."""

    def __init__(self, root, split=None, transform=None):
        split_dir = Path(root)
        if split is not None:
            split_dir = split_dir / split
         
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples = []
        for dirpath, _, filenames in os.walk(str(split_dir), followlinks=True):
            for f in filenames:
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append(os.path.join(dirpath, f))
        
        self.samples.sort()
        self.transform = transform

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir} (searched recursively with symlinks)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.0)


def compute_psnr(mse_val):
    if mse_val == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse_val)


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss for latent compression pipeline."""

    def __init__(self, lmbda=1e-2, loss_type="mse", latent_loss_weight=0.0, lpips_weight=0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.loss_type = loss_type
        self.latent_loss_weight = latent_loss_weight
        self.lpips_weight = lpips_weight
        
        if self.lpips_weight > 0:
            self.lpips_vgg = lpips.LPIPS(net='vgg').eval()
            self.lpips_vgg.requires_grad_(False)

    def forward(self, pipeline_out, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}

        # BPP loss
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in pipeline_out["likelihoods"].values()
        )

        # Pixel-domain distortion
        if pipeline_out["x_hat"] is not None:
            if self.loss_type == "mse":
                out["mse_loss"] = self.mse(pipeline_out["x_hat"], target)
                out["psnr"] = compute_psnr(out["mse_loss"].item())
                distortion = self.lmbda * 255 ** 2 * out["mse_loss"]
            else:
                out["ms_ssim_loss"] = compute_msssim(pipeline_out["x_hat"], target)
                out["ms_ssim"] = out["ms_ssim_loss"].item()
                distortion = self.lmbda * (1 - out["ms_ssim_loss"])
                
            if self.lpips_weight > 0:
                # lpips expects inputs in [-1, 1], our images are [0, 1]
                x_hat_lpips = pipeline_out["x_hat"] * 2.0 - 1.0
                target_lpips = target * 2.0 - 1.0
                # Move to correct device dynamically
                if not getattr(self, "lpips_device_set", False):
                    self.lpips_vgg.to(target.device)
                    self.lpips_device_set = True
                    
                out["lpips_loss"] = self.lpips_vgg(x_hat_lpips, target_lpips).mean()
                distortion = distortion + self.lpips_weight * out["lpips_loss"]
        else:
            distortion = torch.tensor(0.0, device=target.device)

        # Latent-domain auxiliary loss
        if self.latent_loss_weight > 0:
            out["latent_mse"] = self.mse(pipeline_out["latent_hat"], pipeline_out["latent"])
            distortion = distortion + self.latent_loss_weight * out["latent_mse"]

        out["loss"] = distortion + out["bpp_loss"]
        return out


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build_tcm(device):
    """Build TCMLatent model (without pipeline - pipeline created separately)."""
    tcm = TCMLatent(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        N=N, M=M,
        num_slices=NUM_SLICES,
        max_support_slices=MAX_SUPPORT_SLICES,
        ga_config=GA_CONFIG,
        gs_config=GS_CONFIG,
        ha_config=HA_CONFIG,
        hs_config=HS_CONFIG,
        ga_head_dim=GA_HEAD_DIM,
        gs_head_dim=GS_HEAD_DIM,
        window_size=WINDOW_SIZE,
        atten_window_size=ATTEN_WINDOW_SIZE,
        drop_path_rate=DROP_PATH_RATE,
    )
    return tcm


def configure_optimizers(tcm_model, lr, aux_lr):
    """Separate main and auxiliary optimizers (TCM only)."""
    parameters = {
        n for n, p in tcm_model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in tcm_model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(tcm_model.named_parameters())
    assert len(parameters & aux_parameters) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_lr,
    )
    return optimizer, aux_optimizer


def train_one_epoch(pipeline, criterion, train_dataloader, optimizer, aux_optimizer,
                    epoch, clip_max_norm, writer, global_step, accelerator):
    """Train for one epoch with accelerate."""
    pipeline.train()

    # Wrap dataloader in tqdm only on the main process for clean output
    if accelerator.is_main_process:
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=True)
        iterator = enumerate(pbar)
    else:
        iterator = enumerate(train_dataloader)

    for i, d in iterator:
        with accelerator.accumulate(pipeline.tcm):
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            out = pipeline(d, decode_pixel=PIXEL_LOSS)
            out_criterion = criterion(out, d)

            accelerator.backward(out_criterion["loss"])
            if clip_max_norm > 0:
                accelerator.clip_grad_norm_(pipeline.tcm.parameters(), clip_max_norm)
            optimizer.step()

            aux_loss = pipeline.tcm.aux_loss()
            accelerator.backward(aux_loss)
            aux_optimizer.step()

        # TensorBoard scalar logging (main process only)
        if accelerator.is_main_process and global_step % 100 == 0:
            writer.add_scalar("train/loss", out_criterion["loss"].item(), global_step)
            writer.add_scalar("train/bpp_loss", out_criterion["bpp_loss"].item(), global_step)
            writer.add_scalar("train/aux_loss", aux_loss.item(), global_step)
            if "mse_loss" in out_criterion:
                writer.add_scalar("train/mse_loss", out_criterion["mse_loss"].item(), global_step)
                writer.add_scalar("train/psnr", out_criterion["psnr"], global_step)
            if "ms_ssim_loss" in out_criterion:
                writer.add_scalar("train/ms_ssim", out_criterion["ms_ssim"], global_step)
            if "lpips_loss" in out_criterion:
                writer.add_scalar("train/lpips_loss", out_criterion["lpips_loss"].item(), global_step)
            if "latent_mse" in out_criterion:
                writer.add_scalar("train/latent_mse", out_criterion["latent_mse"].item(), global_step)

        # Update progress bar prefix (main process only)
        if accelerator.is_main_process:
            metrics_dict = {
                "Loss": f'{out_criterion["loss"].item():.3f}',
                "BPP": f'{out_criterion["bpp_loss"].item():.3f}',
                "Aux": f'{aux_loss.item():.2f}'
            }
            if "mse_loss" in out_criterion:
                metrics_dict["PSNR"] = f'{out_criterion["psnr"]:.2f}'
            if "lpips_loss" in out_criterion:
                metrics_dict["LPIPS"] = f'{out_criterion["lpips_loss"].item():.3f}'
            
            pbar.set_postfix(metrics_dict)

        global_step += 1

    return global_step


def test_epoch(epoch, test_dataloader, pipeline, criterion, writer, accelerator, logger):
    """Evaluate on test set."""
    pipeline.eval()

    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    aux_meter = AverageMeter()
    lpips_meter = AverageMeter()

    if accelerator.is_main_process:
        pbar = tqdm(test_dataloader, desc=f"Test Epoch {epoch}", leave=False)
        iterator = enumerate(pbar)
    else:
        iterator = enumerate(test_dataloader)

    sample_images = None

    with torch.no_grad():
        for i, d in iterator:
            out = pipeline(d, decode_pixel=True)
            out_criterion = criterion(out, d)

            loss_meter.update(out_criterion["loss"].item())
            bpp_meter.update(out_criterion["bpp_loss"].item())
            aux_meter.update(pipeline.tcm.aux_loss().item())

            if "mse_loss" in out_criterion:
                mse_meter.update(out_criterion["mse_loss"].item())
                psnr_meter.update(out_criterion["psnr"])
            if "lpips_loss" in out_criterion:
                lpips_meter.update(out_criterion["lpips_loss"].item())

            # Capture first batch for image logging (main process only)
            if accelerator.is_main_process and sample_images is None and out["x_hat"] is not None:
                n_show = min(4, d.size(0))
                sample_images = {
                    "input": d[:n_show].cpu(),
                    "recon": out["x_hat"][:n_show].cpu(),
                    "latent": out["latent"][:n_show].cpu(),
                }

    # TensorBoard (main process only)
    if accelerator.is_main_process:
        writer.add_scalar("test/loss", loss_meter.avg, epoch)
        writer.add_scalar("test/bpp_loss", bpp_meter.avg, epoch)
        writer.add_scalar("test/aux_loss", aux_meter.avg, epoch)
        if mse_meter.count > 0:
            writer.add_scalar("test/mse_loss", mse_meter.avg, epoch)
            writer.add_scalar("test/psnr", psnr_meter.avg, epoch)
        if lpips_meter.count > 0:
            writer.add_scalar("test/lpips_loss", lpips_meter.avg, epoch)

        if epoch % LOG_IMAGE_INTERVAL == 0 and sample_images is not None:
            # Individual grids
            input_grid = make_grid(sample_images["input"], nrow=2, normalize=True)
            recon_grid = make_grid(sample_images["recon"], nrow=2, normalize=True)
            writer.add_image("test/input", input_grid, epoch)
            writer.add_image("test/reconstruction", recon_grid, epoch)

            # Side-by-side comparison (Orig | Recon)
            comparison = torch.cat([sample_images["input"], sample_images["recon"]], dim=3)
            comp_grid = make_grid(comparison, nrow=1, normalize=True)
            writer.add_image("test/comparison", comp_grid, epoch)

            latent_vis = sample_images["latent"][:, :3, :, :]
            latent_grid = make_grid(latent_vis, nrow=2, normalize=True)
            writer.add_image("test/latent_vis", latent_grid, epoch)
            writer.add_histogram("test/latent_distribution",
                                 sample_images["latent"].flatten(), epoch)

        logger.info(
            f"Test epoch {epoch}: "
            f"Loss: {loss_meter.avg:.4f} | "
            f"MSE: {mse_meter.avg:.6f} | "
            f"PSNR: {psnr_meter.avg:.2f}dB | "
            f"LPIPS: {lpips_meter.avg:.4f} | "
            f"BPP: {bpp_meter.avg:.4f} | "
            f"Aux: {aux_meter.avg:.2f}"
        )

    return loss_meter.avg


def save_checkpoint(state, is_best, epoch, save_path):
    os.makedirs(save_path, exist_ok=True)
    torch.save(state, os.path.join(save_path, "checkpoint_latest.pth.tar"))
    if epoch % 5 == 0:
        torch.save(state, os.path.join(save_path, f"checkpoint_epoch{epoch}.pth.tar"))
    if is_best:
        torch.save(state, os.path.join(save_path, "checkpoint_best.pth.tar"))


def main():
    # ---- Accelerator ----
    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    # Generate timestamp for this run (main process only, then broadcasted if needed, but we can just generate it on main)
    # Actually, simpler: just generate string, all processes get nearly same time, but to be 100% safe, only main uses it for saving
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_save_path = os.path.join(SAVE_PATH, timestamp)
    run_tb_path = os.path.join(TENSORBOARD_PATH, timestamp)
    
    # We must ensure save_path exists before configuring logger for it
    if accelerator.is_main_process:
        os.makedirs(run_save_path, exist_ok=True)
        os.makedirs(run_tb_path, exist_ok=True)

    # ---- Setup Logging ----
    logger = logging.getLogger("train_tcm")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        # Only log from main process to avoid duplicate lines
        if accelerator.is_main_process:
            # File Handler
            fh = logging.FileHandler(os.path.join(run_save_path, "training_log.txt"))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            # Stream Handler (Console)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    # Print config (main process only)
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info(f"TCMLatent Training (Accelerate) - Run: {timestamp}")
        logger.info("=" * 60)
        config_vars = {
            "DATASET_PATH": DATASET_PATH, "SAVE_PATH": run_save_path,
            "VAE_MODEL_NAME": VAE_MODEL_NAME, "N": N, "M": M,
            "LAMBDA": LAMBDA, "LOSS_TYPE": LOSS_TYPE, "EPOCHS": EPOCHS,
            "BATCH_SIZE (per GPU)": BATCH_SIZE, "LR": LEARNING_RATE,
            "PATCH_SIZE": PATCH_SIZE, "PIXEL_LOSS": PIXEL_LOSS,
            "LATENT_LOSS_WEIGHT": LATENT_LOSS_WEIGHT,
            "LPIPS_WEIGHT": LPIPS_WEIGHT,
            "NUM_GPUS": accelerator.num_processes,
            "MIXED_PRECISION": MIXED_PRECISION,
            "GRAD_ACCUM_STEPS": GRADIENT_ACCUMULATION_STEPS,
        }
        for k, v in config_vars.items():
            logger.info(f"  {k}: {v}")
        logger.info("=" * 60)

    # Seed
    if SEED is not None:
        set_seed(SEED)

    # TensorBoard (main process only)
    writer = None
    if accelerator.is_main_process:
        writer = SummaryWriter(run_tb_path)

    # ---- Data ----
    train_transforms = transforms.Compose([
        transforms.RandomCrop(PATCH_SIZE),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.CenterCrop(PATCH_SIZE),
        transforms.ToTensor(),
    ])

    train_dataset = RecursiveImageFolder(DATASET_PATH, split=None, transform=train_transforms)
    test_dataset = RecursiveImageFolder(TEST_DATASET_PATH, split=None, transform=test_transforms)

    if accelerator.is_main_process:
        logger.info(f"  Train: {len(train_dataset)} images | Test: {len(test_dataset)} images")

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=True, pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS,
        shuffle=False, pin_memory=True,
    )

    # ---- Model ----
    # 1. Build TCM model
    tcm = build_tcm(accelerator.device)

    # 2. Optimizers (before accelerator.prepare)
    optimizer, aux_optimizer = configure_optimizers(tcm, LEARNING_RATE, AUX_LEARNING_RATE)

    # 3. Prepare TCM + optimizers + dataloaders with accelerate
    tcm, optimizer, aux_optimizer, train_dataloader, test_dataloader = (
        accelerator.prepare(
            tcm, optimizer, aux_optimizer, train_dataloader, test_dataloader
        )
    )

    # 4. Build pipeline with the accelerate-wrapped TCM
    #    VAE is loaded separately on each process (frozen, not wrapped by accelerate)
    pipeline = LatentCompressionPipeline(
        tcm=tcm,
        vae_model_name=VAE_MODEL_NAME,
        vae_checkpoint=VAE_CHECKPOINT,
        vae_device=accelerator.device,
        freeze_vae=True,
    )

    if accelerator.is_main_process:
        tcm_params = sum(p.numel() for p in tcm.parameters() if p.requires_grad)
        logger.info(f"Trainable TCM parameters: {tcm_params:,}")
        logger.info(f"Using {accelerator.num_processes} GPU(s)")

    # Loss
    criterion = RateDistortionLoss(
        lmbda=LAMBDA, loss_type=LOSS_TYPE, latent_loss_weight=LATENT_LOSS_WEIGHT,
        lpips_weight=LPIPS_WEIGHT
    )
    # Move criterion to device for LPIPS networks
    criterion.to(accelerator.device)

    # ---- Resume ----
    last_epoch = 0
    global_step = 0
    if TCM_CHECKPOINT:
        if accelerator.is_main_process:
            logger.info(f"Loading checkpoint: {TCM_CHECKPOINT}")
        ckpt = torch.load(TCM_CHECKPOINT, map_location=accelerator.device)
        # Load into unwrapped model
        unwrapped = accelerator.unwrap_model(tcm)
        unwrapped.load_state_dict(ckpt["state_dict"])
        if CONTINUE_TRAIN:
            last_epoch = ckpt["epoch"] + 1
            optimizer.load_state_dict(ckpt["optimizer"])
            aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
            global_step = ckpt.get("global_step", 0)

    # ---- Training loop ----
    best_loss = float("inf")
    for epoch in range(last_epoch, EPOCHS):
        if epoch < SKIP_EPOCH:
            continue

        if accelerator.is_main_process:
            logger.info(f"\nEpoch {epoch}/{EPOCHS-1} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            if writer:
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        global_step = train_one_epoch(
            pipeline, criterion, train_dataloader,
            optimizer, aux_optimizer, epoch,
            CLIP_MAX_NORM, writer, global_step, accelerator,
        )

        loss = test_epoch(epoch, test_dataloader, pipeline, criterion, writer, accelerator, logger)

        # Save checkpoint (main process only)
        if accelerator.is_main_process:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            unwrapped = accelerator.unwrap_model(tcm)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": unwrapped.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "global_step": global_step,
                },
                is_best, epoch, run_save_path,
            )

        # Sync all processes before next epoch
        accelerator.wait_for_everyone()

    if accelerator.is_main_process and writer:
        writer.close()
        logger.info(f"\nTraining complete. Best loss: {best_loss:.4f}")
        logger.info(f"Checkpoints saved to: {run_save_path}")
        logger.info(f"TensorBoard logs: {run_tb_path}")


if __name__ == "__main__":
    main()
