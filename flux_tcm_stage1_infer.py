import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import lpips
from tqdm import tqdm  # 添加进度条支持

# 导入 MSSSIM 和 DISTS
try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False
    print("Warning: pytorch_msssim not installed. Install with: pip install pytorch-msssim")

try:
    import pyiqa
    HAS_DISTS = True
except ImportError:
    HAS_DISTS = False
    print("Warning: pyiqa not installed. Install with: pip install pyiqa")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from LIC_TCM.models import TCMLatent
from Flow.modules.data import IMAGE_EXTENSIONS
from Flow.modules.lora import inject_lora, load_lora_state_dict
from Flow.modules.pipeline import FlowTCMStage1Pipeline
from Flow.modules.utils import ensure_dir, save_json, write_csv


class ImagePathDataset(Dataset):
    def __init__(self, root: str):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Input dir not found: {root}")
        self.paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        self.paths.sort()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return {
                "image": self.to_tensor(image),
                "path": str(path),
            }
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            # 返回一个占位符，后续过滤
            return {
                "image": torch.zeros(3, 512, 512),
                "path": str(path),
                "error": True
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1 inference for FLUX.2[Klein] + TCM")

    # 支持多个输入目录
    parser.add_argument("--input_dirs", type=str, nargs="+", required=True, 
                        help="One or more input directories (datasets)")
    parser.add_argument("--output_dir", type=str, default="./outputs/infer")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="flux.2-klein-4b")
    parser.add_argument("--flux_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors")
    parser.add_argument("--ae_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors")
    parser.add_argument("--qwen_ckpt", type=str, default="/data2/luosheng/hf_models/hub/Qwen3-4B-FP8")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference (larger = faster but more VRAM)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--infer_steps", type=int, default=4, help="Number of denoising steps (fewer = faster)")
    parser.add_argument("--do_entropy_coding", action="store_true", default=True, help="Use entropy coding for accurate BPP (slow)")
    parser.add_argument("--no_entropy_coding", action="store_true", help="Skip entropy coding for fast testing (BPP will be 0)")

    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_regex", type=str, default=r".*")

    parser.add_argument("--tcm_in_channels", type=int, default=128)
    parser.add_argument("--tcm_out_channels", type=int, default=128)
    parser.add_argument("--tcm_N", type=int, default=128)
    parser.add_argument("--tcm_M", type=int, default=320)
    parser.add_argument("--tcm_num_slices", type=int, default=5)
    parser.add_argument("--tcm_max_support_slices", type=int, default=5)
    parser.add_argument("--tcm_ga_config", type=int, nargs="+", default=[2])
    parser.add_argument("--tcm_gs_config", type=int, nargs="+", default=[2])
    parser.add_argument("--tcm_ha_config", type=int, nargs="+", default=[2])
    parser.add_argument("--tcm_hs_config", type=int, nargs="+", default=[2])
    parser.add_argument("--tcm_ga_head_dim", type=int, nargs="+", default=[8])
    parser.add_argument("--tcm_gs_head_dim", type=int, nargs="+", default=[8])
    parser.add_argument("--tcm_window_size", type=int, default=8)
    parser.add_argument("--tcm_atten_window_size", type=int, default=4)
    parser.add_argument("--tcm_drop_path_rate", type=float, default=0.0)

    # 新增：性能优化选项
    parser.add_argument("--skip_metrics", action="store_true", help="Skip LPIPS/DISTS calculation for faster inference")
    parser.add_argument("--metrics_batch_size", type=int, default=8, help="Batch size for metric calculation")
    parser.add_argument("--save_images", action="store_true", default=True, help="Save reconstructed images")
    parser.add_argument("--verbose", action="store_true", help="Print per-image metrics")

    return parser.parse_args()


def build_tcm(args):
    return TCMLatent(
        in_channels=args.tcm_in_channels,
        out_channels=args.tcm_out_channels,
        N=args.tcm_N,
        M=args.tcm_M,
        num_slices=args.tcm_num_slices,
        max_support_slices=args.tcm_max_support_slices,
        ga_config=args.tcm_ga_config,
        gs_config=args.tcm_gs_config,
        ha_config=args.tcm_ha_config,
        hs_config=args.tcm_hs_config,
        ga_head_dim=args.tcm_ga_head_dim,
        gs_head_dim=args.tcm_gs_head_dim,
        window_size=args.tcm_window_size,
        atten_window_size=args.tcm_atten_window_size,
        drop_path_rate=args.tcm_drop_path_rate,
    )


def collate_fn(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    paths = [x["path"] for x in batch]
    return {"image": images, "path": paths}


def psnr(x, y):
    mse = F.mse_loss(x, y).item()
    if mse == 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def calc_ms_ssim(rec, img):
    """计算 MS-SSIM"""
    if not HAS_MSSSIM:
        return 0.0
    try:
        # MS-SSIM 期望输入在 [0, 1] 范围
        ms_ssim_val = ms_ssim(rec, img, data_range=1.0).item()
        return ms_ssim_val
    except Exception as e:
        print(f"Warning: MS-SSIM calculation failed: {e}")
        return 0.0


def calculate_metrics_batch(lpips_metric, dists_metric, rec_batch, img_batch, device, skip_metrics=False):
    """
    批量计算评估指标，减少 GPU-CPU 数据传输
    
    Args:
        lpips_metric: LPIPS 模型
        dists_metric: DISTS 模型
        rec_batch: 重建图像批次 [B, C, H, W]
        img_batch: 原始图像批次 [B, C, H, W]
        device: 设备
        skip_metrics: 是否跳过深度指标计算
    
    Returns:
        dict: 包含所有指标的列表
    """
    batch_size = rec_batch.shape[0]
    
    # PSNR 和 MS-SSIM 可以快速批量计算
    psnr_values = []
    ms_ssim_values = []
    for i in range(batch_size):
        psnr_values.append(psnr(rec_batch[i:i+1], img_batch[i:i+1]))
        ms_ssim_values.append(calc_ms_ssim(rec_batch[i:i+1], img_batch[i:i+1]))
    
    # LPIPS 和 DISTS 可以批量计算
    if not skip_metrics:
        with torch.no_grad():
            # LPIPS 期望输入在 [-1, 1]
            lpips_values = lpips_metric(rec_batch * 2.0 - 1.0, img_batch * 2.0 - 1.0).squeeze().cpu().tolist()
            if not isinstance(lpips_values, list):
                lpips_values = [lpips_values]
            
            if dists_metric is not None:
                dists_values = dists_metric(rec_batch, img_batch).squeeze().cpu().tolist()
                if not isinstance(dists_values, list):
                    dists_values = [dists_values]
            else:
                dists_values = [0.0] * batch_size
    else:
        lpips_values = [0.0] * batch_size
        dists_values = [0.0] * batch_size
    
    return {
        'psnr': psnr_values,
        'ms_ssim': ms_ssim_values,
        'lpips': lpips_values,
        'dists': dists_values
    }


def main():
    args = parse_args()
    if args.no_entropy_coding:
        args.do_entropy_coding = False

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if accelerator.is_main_process:
        ensure_dir(args.output_dir)
        if args.save_images:
            ensure_dir(os.path.join(args.output_dir, "recon"))
        save_json(os.path.join(args.output_dir, "infer_config.json"), vars(args))

    # 处理多个输入目录
    all_rows = []
    dataset_summaries = {}
    
    for input_dir in args.input_dirs:
        dataset_name = Path(input_dir).name
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name} ({input_dir})")
        print(f"{'='*80}")
        
        dataset = ImagePathDataset(input_dir)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        tcm = build_tcm(args)
        pipeline = FlowTCMStage1Pipeline(
            model_name=args.model_name,
            flux_ckpt=args.flux_ckpt,
            ae_ckpt=args.ae_ckpt,
            qwen_ckpt=args.qwen_ckpt,
            tcm=tcm,
            device=accelerator.device,
            guidance=args.guidance,
        )

        inject_lora(
            pipeline.flux,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_regex=args.lora_target_regex,
        )

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        
        # 兼容两种 checkpoint 格式
        if "tcm" in ckpt and "flux_lora" in ckpt:
            # 格式 1: 最终保存的格式（包含 tcm 和 flux_lora）
            print(f"Loading checkpoint in final format (tcm + flux_lora)")
            pipeline.tcm.load_state_dict(ckpt["tcm"])
            load_lora_state_dict(pipeline.flux, ckpt["flux_lora"])
        elif "pipeline" in ckpt:
            # 格式 2: 定期保存的格式（包含整个 pipeline）
            print(f"Loading checkpoint in pipeline format")
            # 从 pipeline state_dict 中提取 tcm 和 flux 的状态
            pipeline_state = ckpt["pipeline"]
            
            # 提取 TCM 状态（键名以 "tcm." 开头）
            tcm_state = {k.replace("tcm.", ""): v for k, v in pipeline_state.items() if k.startswith("tcm.")}
            pipeline.tcm.load_state_dict(tcm_state)
            
            # 提取 Flux LoRA 状态（键名以 "flux." 开头）
            flux_state = {k.replace("flux.", ""): v for k, v in pipeline_state.items() if k.startswith("flux.")}
            load_lora_state_dict(pipeline.flux, flux_state)
        else:
            raise KeyError(f"Checkpoint format not recognized. Available keys: {list(ckpt.keys())}")
        
        print(f"✓ Checkpoint loaded successfully from {args.checkpoint}")
        pipeline.tcm, pipeline.flux, loader = accelerator.prepare(pipeline.tcm, pipeline.flux, loader)

        pipeline.tcm.eval()
        pipeline.flux.eval()

        # 初始化评估指标
        lpips_metric = lpips.LPIPS(net="vgg").to(accelerator.device).eval()
        if HAS_DISTS and not args.skip_metrics:
            dists_metric = pyiqa.create_metric('dists').to(accelerator.device).eval()
        else:
            dists_metric = None

        rows = []
        sum_psnr = 0.0
        sum_ms_ssim = 0.0
        sum_lpips = 0.0
        sum_dists = 0.0
        sum_bpp = 0.0
        count = 0

        # 用于批量收集指标
        rec_buffer = []
        img_buffer = []
        path_buffer = []
        bytes_buffer = []
        
        # 添加进度条
        progress_bar = tqdm(loader, desc=f"Inference [{dataset_name}]", disable=not accelerator.is_main_process)
        
        for batch in progress_bar:
            images = batch["image"].to(accelerator.device)
            paths = batch["path"]

            with torch.no_grad():
                with accelerator.autocast():
                    out = pipeline.forward_stage1_infer(
                        images,
                        infer_steps=args.infer_steps,
                        do_entropy_coding=args.do_entropy_coding,
                    )
            recon = out["x_hat"]

            # 收集数据到缓冲区
            for i in range(recon.shape[0]):
                rec_buffer.append(recon[i : i + 1])
                img_buffer.append(images[i : i + 1])
                path_buffer.append(paths[i])
                bytes_buffer.append(out["bytes"][i])
                
                # 如果需要保存图片
                if args.save_images and accelerator.is_main_process:
                    p = paths[i]
                    stem = Path(p).stem
                    save_path = os.path.join(args.output_dir, "recon", f"{stem}_recon.png")
                    save_image(recon[i : i + 1], save_path)
            
            # 当缓冲区达到指定大小或最后一个批次时，批量计算指标
            if len(rec_buffer) >= args.metrics_batch_size or (hasattr(progress_bar, 'n') and progress_bar.n == len(loader)):
                if rec_buffer:
                    # 拼接成批次
                    rec_batch = torch.cat(rec_buffer, dim=0)
                    img_batch = torch.cat(img_buffer, dim=0)
                    
                    # 批量计算指标
                    metrics = calculate_metrics_batch(
                        lpips_metric, dists_metric, rec_batch, img_batch, 
                        accelerator.device, skip_metrics=args.skip_metrics
                    )
                    
                    # 处理结果
                    for j in range(len(rec_buffer)):
                        h, w = img_buffer[j].shape[-2:]
                        curr_bpp = float(bytes_buffer[j]) * 8.0 / float(h * w) if args.do_entropy_coding else 0.0
                        
                        row = {
                            "dataset": dataset_name,
                            "path": path_buffer[j],
                            "psnr": metrics['psnr'][j],
                            "ms_ssim": metrics['ms_ssim'][j],
                            "lpips": metrics['lpips'][j],
                            "dists": metrics['dists'][j],
                            "bpp": curr_bpp,
                        }
                        rows.append(row)
                        all_rows.append(row)

                        sum_psnr += metrics['psnr'][j]
                        sum_ms_ssim += metrics['ms_ssim'][j]
                        sum_lpips += metrics['lpips'][j]
                        sum_dists += metrics['dists'][j]
                        sum_bpp += curr_bpp
                        count += 1
                        
                        # 打印每张图的指标（仅在 verbose 模式下）
                        if args.verbose and accelerator.is_main_process:
                            print(f"\n  [{count}] {Path(path_buffer[j]).name}")
                            print(f"      PSNR:    {metrics['psnr'][j]:7.4f} dB")
                            print(f"      MS-SSIM: {metrics['ms_ssim'][j]:7.6f}")
                            print(f"      LPIPS:   {metrics['lpips'][j]:7.6f}")
                            print(f"      DISTS:   {metrics['dists'][j]:7.6f}")
                            print(f"      BPP:     {curr_bpp:7.6f}")
                    
                    # 清空缓冲区
                    rec_buffer.clear()
                    img_buffer.clear()
                    path_buffer.clear()
                    bytes_buffer.clear()
                
                # 更新进度条显示当前平均值
                if accelerator.is_main_process and count > 0:
                    progress_bar.set_postfix({
                        'PSNR': f'{sum_psnr/count:.2f}',
                        'MS-SSIM': f'{sum_ms_ssim/count:.4f}',
                        'LPIPS': f'{sum_lpips/count:.4f}',
                        'DISTS': f'{sum_dists/count:.4f}',
                        'BPP': f'{sum_bpp/count:.4f}'
                    })

        # 处理剩余的缓冲区数据
        if rec_buffer:
            rec_batch = torch.cat(rec_buffer, dim=0)
            img_batch = torch.cat(img_buffer, dim=0)
            
            metrics = calculate_metrics_batch(
                lpips_metric, dists_metric, rec_batch, img_batch, 
                accelerator.device, skip_metrics=args.skip_metrics
            )
            
            for j in range(len(rec_buffer)):
                h, w = img_buffer[j].shape[-2:]
                curr_bpp = float(bytes_buffer[j]) * 8.0 / float(h * w) if args.do_entropy_coding else 0.0
                
                row = {
                    "dataset": dataset_name,
                    "path": path_buffer[j],
                    "psnr": metrics['psnr'][j],
                    "ms_ssim": metrics['ms_ssim'][j],
                    "lpips": metrics['lpips'][j],
                    "dists": metrics['dists'][j],
                    "bpp": curr_bpp,
                }
                rows.append(row)
                all_rows.append(row)

                sum_psnr += metrics['psnr'][j]
                sum_ms_ssim += metrics['ms_ssim'][j]
                sum_lpips += metrics['lpips'][j]
                sum_dists += metrics['dists'][j]
                sum_bpp += curr_bpp
                count += 1
                
                if args.verbose and accelerator.is_main_process:
                    print(f"\n  [{count}] {Path(path_buffer[j]).name}")
                    print(f"      PSNR:    {metrics['psnr'][j]:7.4f} dB")
                    print(f"      MS-SSIM: {metrics['ms_ssim'][j]:7.6f}")
                    print(f"      LPIPS:   {metrics['lpips'][j]:7.6f}")
                    print(f"      DISTS:   {metrics['dists'][j]:7.6f}")
                    print(f"      BPP:     {curr_bpp:7.6f}")

        # 计算当前数据集的平均值
        if accelerator.is_main_process and count > 0:
            dataset_summary = {
                "dataset": dataset_name,
                "num_images": count,
                "psnr": sum_psnr / count,
                "ms_ssim": sum_ms_ssim / count,
                "lpips": sum_lpips / count,
                "dists": sum_dists / count,
                "bpp": sum_bpp / count,
            }
            dataset_summaries[dataset_name] = dataset_summary
            
            print(f"\n{'='*80}")
            print(f"Dataset Summary: {dataset_name}")
            print(f"{'='*80}")
            print(f"  Images:    {count}")
            print(f"  Avg PSNR:    {dataset_summary['psnr']:.4f} dB")
            print(f"  Avg MS-SSIM: {dataset_summary['ms_ssim']:.6f}")
            print(f"  Avg LPIPS:   {dataset_summary['lpips']:.6f}")
            print(f"  Avg DISTS:   {dataset_summary['dists']:.6f}")
            print(f"  Avg BPP:     {dataset_summary['bpp']:.6f}")
            print(f"{'='*80}\n")
            
            # 保存单个数据集的指标
            dataset_output_dir = os.path.join(args.output_dir, dataset_name)
            ensure_dir(dataset_output_dir)
            write_csv(os.path.join(dataset_output_dir, "metrics.csv"), rows)
            save_json(os.path.join(dataset_output_dir, "summary.json"), dataset_summary)

    # 所有数据集处理完成后，计算总体平均
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and all_rows:
        # 保存所有数据的合并结果
        write_csv(os.path.join(args.output_dir, "metrics_all_datasets.csv"), all_rows)
        
        # 计算总体平均
        total_count = len(all_rows)
        overall_summary = {
            "total_datasets": len(args.input_dirs),
            "total_images": total_count,
            "datasets": dataset_summaries,
            "overall_average": {
                "psnr": sum(r["psnr"] for r in all_rows) / total_count,
                "ms_ssim": sum(r["ms_ssim"] for r in all_rows) / total_count,
                "lpips": sum(r["lpips"] for r in all_rows) / total_count,
                "dists": sum(r["dists"] for r in all_rows) / total_count,
                "bpp": sum(r["bpp"] for r in all_rows) / total_count,
            }
        }
        save_json(os.path.join(args.output_dir, "metrics_summary.json"), overall_summary)
        
        # 打印总体总结
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY - All Datasets")
        print(f"{'='*80}")
        print(f"Total datasets processed: {len(args.input_dirs)}")
        print(f"Total images processed:   {total_count}")
        print(f"\nOverall Averages:")
        print(f"  PSNR:    {overall_summary['overall_average']['psnr']:.4f} dB")
        print(f"  MS-SSIM: {overall_summary['overall_average']['ms_ssim']:.6f}")
        print(f"  LPIPS:   {overall_summary['overall_average']['lpips']:.6f}")
        print(f"  DISTS:   {overall_summary['overall_average']['dists']:.6f}")
        print(f"  BPP:     {overall_summary['overall_average']['bpp']:.6f}")
        print(f"\nPer-Dataset Summaries:")
        for ds_name, ds_summary in dataset_summaries.items():
            print(f"\n  {ds_name}:")
            print(f"    Images: {ds_summary['num_images']}")
            print(f"    PSNR:    {ds_summary['psnr']:.4f} dB")
            print(f"    MS-SSIM: {ds_summary['ms_ssim']:.6f}")
            print(f"    LPIPS:   {ds_summary['lpips']:.6f}")
            print(f"    DISTS:   {ds_summary['dists']:.6f}")
            print(f"    BPP:     {ds_summary['bpp']:.6f}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
