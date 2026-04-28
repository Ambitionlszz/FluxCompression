"""Inference script for TCMLatent with FLUX.2 VAE pipeline.

Supports single-GPU inference. For multi-GPU batch inference,
use accelerate launch (optional).

Usage:
    # Single GPU
    python infer_tcm.py

    # Multi-GPU (for large batch inference)
    accelerate launch --num_processes=2 infer_tcm.py
"""

import csv
import datetime
import math
import os
import sys
import time
import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
from torchvision.utils import save_image
import lpips

# ============================================================
# PATH CONFIGURATION (modify these directly)
# ============================================================
DATASETS = {
    "Kodak": "/data2/luosheng/data/Datasets/Kodak",
    "CLIC2020": "/data2/luosheng/data/Datasets/CLIC2020_test",
}
OUTPUT_PATH = "./results/inference_output/"      # Base output dir for reconstructed images
TCM_CHECKPOINT = "./results/checkpoints/tcm_latent/20260309_040128/checkpoint_best.pth.tar" # Trained TCM checkpoint
VAE_CHECKPOINT = "/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors"

# ============================================================
# MODEL CONFIGURATION (must match training)
# ============================================================
VAE_MODEL_NAME = "flux.2-klein-4b"
IN_CHANNELS = 128
OUT_CHANNELS = 128
N = 128
M = 320
NUM_SLICES = 5
MAX_SUPPORT_SLICES = 5
GA_CONFIG = [2]
GS_CONFIG = [2]
HA_CONFIG = [2]
HS_CONFIG = [2]
GA_HEAD_DIM = [8]
GS_HEAD_DIM = [8]
WINDOW_SIZE = 8
ATTEN_WINDOW_SIZE = 4

# ============================================================
# INFERENCE CONFIGURATION
# ============================================================
SAVE_COMPARISON = True          # Save side-by-side input vs output comparison
COMPUTE_METRICS = True          # Compute PSNR / MS-SSIM / BPP
SAVE_METRICS_CSV = True         # Save metrics to CSV file
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# ============================================================
# End of configuration
# ============================================================

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models import TCMLatent
from tcm_pipeline import LatentCompressionPipeline


def compute_psnr(mse_val):
    if mse_val == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse_val)


def pad_to_multiple(img_tensor, multiple=16):
    """Pad image to be a multiple of `multiple` in both spatial dims."""
    _, _, h, w = img_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return img_tensor, h, w


def load_model(device):
    """Build model and load checkpoint."""
    tcm = TCMLatent(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
        N=N, M=M, num_slices=NUM_SLICES, max_support_slices=MAX_SUPPORT_SLICES,
        ga_config=GA_CONFIG, gs_config=GS_CONFIG,
        ha_config=HA_CONFIG, hs_config=HS_CONFIG,
        ga_head_dim=GA_HEAD_DIM, gs_head_dim=GS_HEAD_DIM,
        window_size=WINDOW_SIZE, atten_window_size=ATTEN_WINDOW_SIZE,
    )

    # Load checkpoint - print handled by invoker via logger now
    ckpt = torch.load(TCM_CHECKPOINT, map_location=device, weights_only=False)
    tcm.load_state_dict(ckpt["state_dict"])
    tcm = tcm.to(device)

    pipeline = LatentCompressionPipeline(
        tcm=tcm, vae_model_name=VAE_MODEL_NAME,
        vae_checkpoint=VAE_CHECKPOINT, vae_device=device, freeze_vae=True,
    )

    pipeline.eval()
    
    # Initialize the CDFs for CompressAI's entropy models
    # This must be called after loading weights and before running .compress()
    pipeline.tcm.update(force=True)
    
    return pipeline


def get_image_paths(input_path):
    """Get list of image paths from file or directory."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        paths = sorted([f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS])
        return paths
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


@torch.no_grad()
def infer_single(pipeline, img_path, output_dir, device, lpips_vgg, dataset_name):
    """Run inference on a single image. Returns metrics dict."""
    img = Image.open(img_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).to(device)

    # Pad to multiple of 256 (= 16 * 16, for VAE + TCM window alignment)
    img_padded, orig_h, orig_w = pad_to_multiple(img_tensor, multiple=256)

    # ---------------------------------------------------------
    # ENCODING STAGE (VAE Encode -> TCM Compress)
    # ---------------------------------------------------------
    
    # 1. VAE Encode
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.time()
    latent = pipeline.encode_to_latent(img_padded)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.time()
    vae_enc_time = t1 - t0

    # 2. TCM Compress (Entropy Encode)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t2 = time.time()
    # Use compress instead of forward to get actual bitstreams
    tcm_compressed = pipeline.tcm.compress(latent)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t3 = time.time()
    tcm_enc_time = t3 - t2
    
    total_enc_time = vae_enc_time + tcm_enc_time

    # Compute Bitrate
    # tcm_compressed["strings"] contains [y_strings, z_strings]
    # where y_strings is a list of byte strings (one per slice), z_strings is a list of byte strings
    num_pixels = orig_h * orig_w
    
    # tcm_compressed["strings"][0] is y_strings: a list of byte strings (one per slice)
    # tcm_compressed["strings"][1] is z_strings: a list containing a single byte string
    
    bytes_y = sum(len(s) for s in tcm_compressed["strings"][0])
    bytes_z = sum(len(s) for s in tcm_compressed["strings"][1])
    total_bytes = bytes_y + bytes_z
    bpp = (total_bytes * 8.0) / num_pixels

    # ---------------------------------------------------------
    # DECODING STAGE (TCM Decompress -> VAE Decode)
    # ---------------------------------------------------------

    # 3. TCM Decompress (Entropy Decode)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t4 = time.time()
    # Decode from bitstreams back to latent
    tcm_decompressed = pipeline.tcm.decompress(tcm_compressed["strings"], tcm_compressed["shape"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t5 = time.time()
    tcm_dec_time = t5 - t4

    # 4. VAE Decode
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t6 = time.time()
    recon = pipeline.decode_from_latent(tcm_decompressed["x_hat"], no_grad=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t7 = time.time()
    vae_dec_time = t7 - t6
    
    total_dec_time = tcm_dec_time + vae_dec_time

    # Crop to original shape
    recon = recon[:, :, :orig_h, :orig_w]

    # Save reconstructed image
    stem = Path(img_path).stem
    recon_path = os.path.join(output_dir, f"{stem}_recon.png")
    save_image(recon, recon_path)

    # Save comparison
    if SAVE_COMPARISON:
        comparison = torch.cat([img_tensor[:, :, :orig_h, :orig_w], recon], dim=3)
        comp_path = os.path.join(output_dir, f"{stem}_compare.png")
        save_image(comparison, comp_path)

    # Compute metrics
    metrics = {
        "dataset": dataset_name,
        "filename": Path(img_path).name,
        "vae_enc_time": vae_enc_time,
        "tcm_enc_time": tcm_enc_time,
        "tcm_dec_time": tcm_dec_time,
        "vae_dec_time": vae_dec_time,
        "enc_time": total_enc_time,
        "dec_time": total_dec_time,
        "bpp": bpp,
    }
    if COMPUTE_METRICS:
        mse_val = torch.nn.functional.mse_loss(recon, img_tensor[:, :, :orig_h, :orig_w]).item()
        metrics["mse"] = mse_val
        metrics["psnr"] = compute_psnr(mse_val)

        if orig_h >= 160 and orig_w >= 160:
            metrics["ms_ssim"] = ms_ssim(
                recon, img_tensor[:, :, :orig_h, :orig_w], data_range=1.0
            ).item()
        else:
            metrics["ms_ssim"] = float("nan")

        # LPIPS computation
        orig_lpips = img_tensor[:, :, :orig_h, :orig_w] * 2.0 - 1.0
        recon_lpips = recon * 2.0 - 1.0
        metrics["lpips"] = lpips_vgg(recon_lpips, orig_lpips).item()

    return metrics


def main():
    accelerator = Accelerator()
    device = accelerator.device

    # Generate timestamp for this inference run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_path = os.path.join(OUTPUT_PATH, timestamp)

    if accelerator.is_main_process:
        os.makedirs(run_output_path, exist_ok=True)

    # ---- Setup Logging ----
    logger = logging.getLogger("infer_tcm")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(message)s') # Keep simple for inference
        if accelerator.is_main_process:
            # File Handler
            fh = logging.FileHandler(os.path.join(run_output_path, "inference_log.txt"))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            # Stream Handler (Console)
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info(f"TCMLatent Inference - Run: {timestamp}")
        logger.info("=" * 60)

    # Load model
    if accelerator.is_main_process:
        logger.info(f"Loading TCM checkpoint: {TCM_CHECKPOINT}")
    pipeline = load_model(device)

    # Load LPIPS vgg
    if COMPUTE_METRICS:
        lpips_vgg = lpips.LPIPS(net='vgg').eval().to(device)
        for p in lpips_vgg.parameters():
            p.requires_grad = False
    else:
        lpips_vgg = None

    all_dataset_metrics = []

    for dataset_name, dataset_path in DATASETS.items():
        if not os.path.exists(dataset_path):
            if accelerator.is_main_process:
                logger.info(f"Skipping {dataset_name}: Path not found ({dataset_path})")
            continue

        if accelerator.is_main_process:
            logger.info(f"\nEvaluating dataset: {dataset_name}")
            dataset_output_dir = os.path.join(run_output_path, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
        else:
            dataset_output_dir = os.path.join(run_output_path, dataset_name)

        # Get image paths
        image_paths = get_image_paths(dataset_path)

        # Distribute images across processes
        local_paths = image_paths[accelerator.process_index::accelerator.num_processes]

        if accelerator.is_main_process:
            logger.info(f"Total: {len(image_paths)} images | Per GPU: ~{len(local_paths)}")

        # Warmup (Process a dummy tensor once to initialize CUDA kernels/caches)
        if len(local_paths) > 0:
            if accelerator.is_main_process:
                logger.info("Performing GPU warmup pass...")
            # We just need any input tensor with the right shape, e.g. 1x3x256x256
            dummy_input = torch.zeros(1, 3, 256, 256, device=device)
            with torch.no_grad():
                dummy_latent = pipeline.encode_to_latent(dummy_input)
                dummy_tcm_out = pipeline.tcm.compress(dummy_latent)
                _ = pipeline.tcm.decompress(dummy_tcm_out["strings"], dummy_tcm_out["shape"])
                _ = pipeline.decode_from_latent(dummy_latent, no_grad=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        # Process images
        local_metrics = []
        for i, img_path in enumerate(local_paths):
            if accelerator.is_main_process:
                total_done = i * accelerator.num_processes
                # We use end="" to keep the log on the same line, loggers don't handle this well
                # So we combine the printing logic for inference lines
                prefix_msg = f"[{total_done:03d}/{len(image_paths):03d}] Processing: {img_path.name} ... "
            else:
                prefix_msg = ""
            
            metrics = infer_single(pipeline, img_path, dataset_output_dir, device, lpips_vgg, dataset_name)
            local_metrics.append(metrics)

            if accelerator.is_main_process and COMPUTE_METRICS:
                result_msg = (f"BPP: {metrics['bpp']:5.3f} | "
                              f"PSNR: {metrics['psnr']:5.2f}dB | "
                              f"SSIM: {metrics['ms_ssim']:6.4f} | "
                              f"LPIPS: {metrics['lpips']:6.4f} | "
                              f"vE:{metrics['vae_enc_time']*1000:4.0f} / "
                              f"TCM_E:{metrics['tcm_enc_time']*1000:4.0f} / "
                              f"TCM_D:{metrics['tcm_dec_time']*1000:4.0f} / "
                              f"vD:{metrics['vae_dec_time']*1000:4.0f} (ms)")
                logger.info(prefix_msg + result_msg)

        # Gather metrics from all processes
        accelerator.wait_for_everyone()
        
        # Collect all metrics on main process
        all_local = accelerator.gather_for_metrics(local_metrics) if hasattr(accelerator, 'gather_for_metrics') else local_metrics
        # Fallback: just use local metrics (single GPU or simple gather)
        ds_metrics = all_local if isinstance(all_local, list) and len(all_local) > 0 and isinstance(all_local[0], dict) else local_metrics
        
        all_dataset_metrics.extend(ds_metrics)

        if accelerator.is_main_process:
            # Summary for this dataset
            if COMPUTE_METRICS and len(ds_metrics) > 0:
                avg_psnr = sum(m["psnr"] for m in ds_metrics) / len(ds_metrics)
                avg_bpp = sum(m["bpp"] for m in ds_metrics) / len(ds_metrics)
                avg_lpips = sum(m["lpips"] for m in ds_metrics) / len(ds_metrics)
                avg_vae_enc = sum(m["vae_enc_time"] for m in ds_metrics) / len(ds_metrics)
                avg_tcm_enc = sum(m["tcm_enc_time"] for m in ds_metrics) / len(ds_metrics)
                avg_tcm_dec = sum(m["tcm_dec_time"] for m in ds_metrics) / len(ds_metrics)
                avg_vae_dec = sum(m["vae_dec_time"] for m in ds_metrics) / len(ds_metrics)
                avg_enc = sum(m["enc_time"] for m in ds_metrics) / len(ds_metrics)
                avg_dec = sum(m["dec_time"] for m in ds_metrics) / len(ds_metrics)
                
                valid_ssim = [m["ms_ssim"] for m in ds_metrics if not math.isnan(m["ms_ssim"])]
                avg_ssim = sum(valid_ssim) / len(valid_ssim) if valid_ssim else float("nan")

                logger.info(f"\n{'=' * 60}")
                logger.info(f"{dataset_name} Average Results ({len(ds_metrics)} images):")
                logger.info(f"  BPP:          {avg_bpp:.4f}")
                logger.info(f"  PSNR:         {avg_psnr:.2f} dB")
                logger.info(f"  MS-SSIM:      {avg_ssim:.4f}")
                logger.info(f"  LPIPS:        {avg_lpips:.4f}")
                logger.info(f"  -----------------------")
                logger.info(f"                 [Detailed Timing]")
                logger.info(f"  VAE Encode:   {avg_vae_enc*1000:.1f} ms")
                logger.info(f"  TCM Encode:   {avg_tcm_enc*1000:.1f} ms")
                logger.info(f"  TCM Decode:   {avg_tcm_dec*1000:.1f} ms")
                logger.info(f"  VAE Decode:   {avg_vae_dec*1000:.1f} ms")
                logger.info(f"  -----------------------")
                logger.info(f"                 [Traditional Total]")
                logger.info(f"  Total Encode: {avg_enc*1000:.1f} ms  (VAE Enc + TCM Enc)")
                logger.info(f"  Total Decode: {avg_dec*1000:.1f} ms  (TCM Dec + VAE Dec)")
                logger.info(f"{'=' * 60}")

    if accelerator.is_main_process:
        # Save comprehensive CSV
        if SAVE_METRICS_CSV and COMPUTE_METRICS and len(all_dataset_metrics) > 0:
            csv_path = os.path.join(run_output_path, "all_metrics.csv")
            with open(csv_path, "w", newline="") as f:
                fieldnames = ["dataset", "filename", "bpp", "psnr", "ms_ssim", "lpips", "mse", 
                              "vae_enc_time", "tcm_enc_time", "tcm_dec_time", "vae_dec_time", 
                              "enc_time", "dec_time"]
                csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
                csv_writer.writeheader()
                for m in all_dataset_metrics:
                    csv_writer.writerow(m)
            logger.info(f"\nComplete metrics saved to: {csv_path}")

        logger.info(f"All images and logs saved to: {run_output_path}")

if __name__ == "__main__":
    main()
