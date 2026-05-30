import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
import lpips
from tqdm import tqdm

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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modules.latent_codec import LatentCodec
from modules.data import IMAGE_EXTENSIONS, ResizeIfSmall
from modules.lora import inject_lora, inject_lora_conv, load_lora_state_dict
from modules.pipeline import FluxCodecPipeline
from modules.utils import ensure_dir, save_json, write_csv
from elic_aux_encoder import load_elic_encoder


class ImagePathDataset(Dataset):
    def __init__(self, root: str, image_size: int | None = None, center_crop: bool = False):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Input dir not found: {root}")
        self.paths = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        self.paths.sort()
        transform_list = []
        if center_crop:
            if image_size is None or image_size <= 0:
                raise ValueError("--image_size must be positive when --center_crop is enabled")
            transform_list.extend(
                [
                    ResizeIfSmall(image_size),
                    transforms.CenterCrop(image_size),
                ]
            )
        transform_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return {
                "image": self.transform(image),
                "path": str(path),
            }
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            return {
                "image": torch.zeros(3, 512, 512),
                "path": str(path),
                "error": True
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for FluxCodec")

    parser.add_argument("--input_dirs", type=str, nargs="+", required=True, 
                        help="One or more input directories (datasets)")
    parser.add_argument("--output_dir", type=str, default="./outputs/infer")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="flux.2-klein-4b")
    parser.add_argument("--flux_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors")
    parser.add_argument("--ae_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors")
    parser.add_argument("--qwen_ckpt", type=str, default="/data2/luosheng/hf_models/hub/Qwen3-4B-FP8")
    parser.add_argument("--elic_ckpt", type=str, default="/data2/luosheng/code/DiT-IC/checkpoints/elic_official.pth", help="Path to pretrained ELIC checkpoint")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--infer_steps", type=int, default=4)
    parser.add_argument("--do_entropy_coding", action="store_true", default=True)
    parser.add_argument("--no_entropy_coding", action="store_true")
    parser.add_argument("--image_size", type=int, default=256, help="Crop size used with --center_crop")
    parser.add_argument("--center_crop", action="store_true", help="Match training eval preprocessing instead of using original image sizes")

    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_regex", type=str, default=r".*")

    # LatentCodec args
    parser.add_argument("--codec_ch_emd", type=int, default=128, help="FLUX AE output channels")
    parser.add_argument("--codec_channel", type=int, default=320, help="Bottleneck channel")
    parser.add_argument("--codec_channel_out", type=int, default=128, help="Codec output channels")
    parser.add_argument("--codec_num_slices", type=int, default=5)
    parser.add_argument("--use_aux_encoder", type=int, default=1, help="1 to enable ELIC aux encoder, 0 to disable")
    parser.add_argument("--use_aux_decoder", type=int, default=1, help="1 to enable AuxDecoder residual, 0 to disable")
    parser.add_argument("--aux_decoder_zero_init", type=int, default=0, help="Only affects randomly initialized AuxDecoder before checkpoint loading")
    parser.add_argument("--elic_proj_channels", type=int, default=64, help="ELIC feature projection channels (320 -> elic_proj_channels)")

    parser.add_argument("--color_fix", action="store_true", help="Apply StableCodec color fix strategy")
    parser.add_argument("--use_ema_weights", type=int, default=1, help="1 to load EMA weights from checkpoint when available")
    parser.add_argument("--unload_text_encoder", action="store_true", default=True,
                        help="Cache fixed prompt embeddings and unload Qwen before inference")
    parser.add_argument("--keep_text_encoder", action="store_true",
                        help="Keep Qwen text encoder on GPU after prompt caching")
    parser.add_argument("--use_tile", action="store_true",
                        help="Enable latent-tiled FLUX denoise for large images")
    parser.add_argument("--tile_latent_size", type=int, default=64,
                        help="Latent tile size. 64 roughly equals 1024px image tiles.")
    parser.add_argument("--tile_latent_overlap", type=int, default=16,
                        help="Latent tile overlap used for blending")

    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--save_images", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")
    
    parser.add_argument("--use_gradient_checkpointing", action="store_true")

    return parser.parse_args()


def build_codec(args):
    return LatentCodec(
        ch_emd=args.codec_ch_emd,
        channel=args.codec_channel,
        channel_out=args.codec_channel_out,
        num_slices=args.codec_num_slices,
        use_aux_encoder=bool(args.use_aux_encoder),
        use_aux_decoder=bool(args.use_aux_decoder),
        aux_decoder_zero_init=bool(args.aux_decoder_zero_init),
        elic_proj_channels=args.elic_proj_channels,
    )


def infer_lora_rank(lora_sd: dict[str, torch.Tensor], name: str) -> int:
    for key, tensor in lora_sd.items():
        if key.endswith(".lora_A"):
            return int(tensor.shape[0])
    for key, tensor in lora_sd.items():
        if key.endswith(".lora_B"):
            return int(tensor.shape[1])
    raise KeyError(f"Could not infer {name} LoRA rank from keys: {list(lora_sd.keys())[:5]}")


def _ckpt_arg(ckpt_args: dict, key: str, default):
    return ckpt_args[key] if key in ckpt_args else default


def _infer_codec_channel(codec_sd: dict[str, torch.Tensor], default: int) -> int:
    weight = codec_sd.get("g_a.analysis_transform.2.down.weight")
    if weight is not None:
        return int(weight.shape[0])
    return int(default)


def _infer_elic_proj_channels(codec_sd: dict[str, torch.Tensor], default: int = 64) -> int:
    weight = codec_sd.get("g_a.aux_proj.weight")
    if weight is not None:
        return int(weight.shape[0])
    return int(default)


def _infer_codec_ch_emd(codec_sd: dict[str, torch.Tensor], use_aux_encoder: bool, default: int) -> int:
    weight = codec_sd.get("g_a.fuse.weight")
    if weight is not None:
        if use_aux_encoder:
            aux_proj_weight = codec_sd.get("g_a.aux_proj.weight")
            if aux_proj_weight is not None:
                aux_channels = int(aux_proj_weight.shape[0])
            else:
                aux_channels = 64  # fallback for legacy checkpoints
        else:
            aux_channels = 0
        return int(weight.shape[1] - aux_channels)
    return int(default)


def _infer_codec_channel_out(codec_sd: dict[str, torch.Tensor], default: int) -> int:
    weight = codec_sd.get("g_s.synthesis_transform.4.weight")
    if weight is not None:
        return int(weight.shape[0])
    return int(default)


def apply_checkpoint_config(args, ckpt: dict):
    codec_sd = ckpt["codec"]
    ckpt_args = ckpt.get("args", {})

    has_aux_encoder = bool(codec_sd.get("g_a.aux_proj.weight") is not None)
    has_aux_decoder = any(k.startswith("aux.") for k in codec_sd.keys())

    args.model_name = _ckpt_arg(ckpt_args, "model_name", args.model_name)
    args.guidance = float(_ckpt_arg(ckpt_args, "guidance", args.guidance))

    args.codec_channel = int(_ckpt_arg(ckpt_args, "codec_channel", _infer_codec_channel(codec_sd, args.codec_channel)))
    args.codec_ch_emd = int(
        _ckpt_arg(ckpt_args, "codec_ch_emd", _infer_codec_ch_emd(codec_sd, has_aux_encoder, args.codec_ch_emd))
    )
    args.codec_channel_out = int(
        _ckpt_arg(ckpt_args, "codec_channel_out", _infer_codec_channel_out(codec_sd, args.codec_channel_out))
    )
    args.codec_num_slices = int(_ckpt_arg(ckpt_args, "codec_num_slices", args.codec_num_slices))
    args.use_aux_encoder = int(has_aux_encoder)
    args.use_aux_decoder = int(has_aux_decoder)
    args.aux_decoder_zero_init = int(_ckpt_arg(ckpt_args, "aux_decoder_zero_init", args.aux_decoder_zero_init))
    args.elic_proj_channels = int(
        _ckpt_arg(ckpt_args, "elic_proj_channels", _infer_elic_proj_channels(codec_sd, args.elic_proj_channels))
    )

    args.lora_rank = int(_ckpt_arg(ckpt_args, "lora_rank", infer_lora_rank(ckpt["flux_lora"], "FLUX")))
    args.lora_alpha = float(_ckpt_arg(ckpt_args, "lora_alpha", args.lora_rank))
    args.lora_dropout = float(_ckpt_arg(ckpt_args, "lora_dropout", args.lora_dropout))
    args.lora_target_regex = _ckpt_arg(ckpt_args, "lora_target_regex", args.lora_target_regex)

    has_ae_decoder_lora = bool(ckpt.get("ae_decoder_lora"))
    ae_decoder_lora_rank = 32
    ae_decoder_lora_alpha = 32.0
    ae_decoder_lora_target_regex = r".*"
    if has_ae_decoder_lora:
        ae_decoder_lora_rank = int(
            _ckpt_arg(ckpt_args, "ae_lora_rank", infer_lora_rank(ckpt["ae_decoder_lora"], "AE decoder"))
        )
        ae_decoder_lora_alpha = float(_ckpt_arg(ckpt_args, "ae_lora_alpha", ae_decoder_lora_rank))
        ae_decoder_lora_target_regex = _ckpt_arg(ckpt_args, "ae_lora_target_regex", ae_decoder_lora_target_regex)

    has_ae_encoder_lora = bool(ckpt.get("ae_encoder_lora"))
    ae_encoder_lora_rank = 32
    ae_encoder_lora_alpha = 32.0
    ae_encoder_lora_target_regex = r".*"
    if has_ae_encoder_lora:
        ae_encoder_lora_rank = int(
            _ckpt_arg(
                ckpt_args,
                "ae_encoder_lora_rank",
                infer_lora_rank(ckpt["ae_encoder_lora"], "AE encoder"),
            )
        )
        ae_encoder_lora_alpha = float(_ckpt_arg(ckpt_args, "ae_encoder_lora_alpha", ae_encoder_lora_rank))
        ae_encoder_lora_target_regex = _ckpt_arg(
            ckpt_args, "ae_encoder_lora_target_regex", ae_encoder_lora_target_regex
        )

    args.use_ae_lora = int(has_ae_decoder_lora)
    args.use_ae_encoder_lora = int(has_ae_encoder_lora)
    args.ae_lora_rank = ae_decoder_lora_rank
    args.ae_lora_alpha = ae_decoder_lora_alpha
    args.ae_encoder_lora_rank = ae_encoder_lora_rank
    args.ae_encoder_lora_alpha = ae_encoder_lora_alpha

    return (
        has_ae_decoder_lora,
        ae_decoder_lora_rank,
        ae_decoder_lora_alpha,
        ae_decoder_lora_target_regex,
        has_ae_encoder_lora,
        ae_encoder_lora_rank,
        ae_encoder_lora_alpha,
        ae_encoder_lora_target_regex,
    )


def load_ema_parameters(model: torch.nn.Module, ema_state: dict[str, torch.Tensor], name: str) -> tuple[int, int]:
    params = dict(model.named_parameters())
    loaded = 0
    missing = 0
    for key, tensor in ema_state.items():
        param = params.get(key)
        if param is None:
            missing += 1
            continue
        param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
        loaded += 1
    print(f"[EMA] Loaded {name}: {loaded} tensors, missing: {missing}")
    return loaded, missing


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
    if not HAS_MSSSIM:
        return 0.0
    try:
        return ms_ssim(rec, img, data_range=1.0).item()
    except Exception as e:
        print(f"Warning: MS-SSIM calculation failed: {e}")
        return 0.0


def main():
    args = parse_args()
    if args.no_entropy_coding:
        args.do_entropy_coding = False

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    all_rows = []
    dataset_summaries = {}
    
    if accelerator.is_main_process:
        print(f"Loading checkpoint from {args.checkpoint} to auto-detect configuration...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "codec" not in ckpt or "flux_lora" not in ckpt:
        raise KeyError(f"Checkpoint format not recognized. Available keys: {list(ckpt.keys())}")

    (
        has_ae_decoder_lora,
        ae_decoder_lora_rank,
        ae_decoder_lora_alpha,
        ae_decoder_lora_target_regex,
        has_ae_encoder_lora,
        ae_encoder_lora_rank,
        ae_encoder_lora_alpha,
        ae_encoder_lora_target_regex,
    ) = apply_checkpoint_config(args, ckpt)
    use_ema_weights = bool(args.use_ema_weights)
    has_any_ema = any(ckpt.get(k) is not None for k in ("ema_codec", "ema_flux", "ema_ae_decoder", "ema_ae_encoder"))
    args.use_ema_weights_applied = int(use_ema_weights and has_any_ema)

    if accelerator.is_main_process:
        print(
            "Auto-detected checkpoint config: "
            f"model_name={args.model_name}, guidance={args.guidance}, "
            f"codec_ch_emd={args.codec_ch_emd}, codec_channel={args.codec_channel}, "
            f"codec_channel_out={args.codec_channel_out}, codec_num_slices={args.codec_num_slices}, "
            f"use_aux_encoder={bool(args.use_aux_encoder)}, use_aux_decoder={bool(args.use_aux_decoder)}, "
            f"use_ae_lora={has_ae_decoder_lora}, use_ae_encoder_lora={has_ae_encoder_lora}, "
            f"lora_rank={args.lora_rank}, lora_alpha={args.lora_alpha}, "
            f"ae_lora_rank={ae_decoder_lora_rank if has_ae_decoder_lora else 'none'}, "
            f"ae_encoder_lora_rank={ae_encoder_lora_rank if has_ae_encoder_lora else 'none'}, "
            f"use_ema_weights={args.use_ema_weights_applied}"
        )

    if accelerator.is_main_process:
        ensure_dir(args.output_dir)
        ensure_dir(os.path.join(args.output_dir, "recon"))
        save_json(os.path.join(args.output_dir, "infer_config.json"), vars(args))

    codec = build_codec(args)
    
    elic_aux = None
    if args.elic_ckpt and args.use_aux_encoder:
        elic_aux = load_elic_encoder(args.elic_ckpt, device=accelerator.device)

    pipeline = FluxCodecPipeline(
        model_name=args.model_name,
        flux_ckpt=args.flux_ckpt,
        ae_ckpt=args.ae_ckpt,
        qwen_ckpt=args.qwen_ckpt,
        codec=codec,
        elic_aux_encoder=elic_aux,
        device=accelerator.device,
        guidance=args.guidance,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    inject_lora(
        pipeline.flux,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_regex=args.lora_target_regex,
    )

    if has_ae_decoder_lora:
        inject_lora_conv(
            pipeline.ae.decoder,
            rank=ae_decoder_lora_rank,
            alpha=ae_decoder_lora_alpha,
            dropout=args.lora_dropout,
            target_regex=ae_decoder_lora_target_regex,
        )

    if has_ae_encoder_lora:
        inject_lora_conv(
            pipeline.ae.encoder,
            rank=ae_encoder_lora_rank,
            alpha=ae_encoder_lora_alpha,
            dropout=args.lora_dropout,
            target_regex=ae_encoder_lora_target_regex,
        )

    pipeline.codec.load_state_dict(ckpt["codec"])
    load_lora_state_dict(pipeline.flux, ckpt["flux_lora"])
    if has_ae_decoder_lora:
        load_lora_state_dict(pipeline.ae.decoder, ckpt["ae_decoder_lora"])
    if has_ae_encoder_lora:
        load_lora_state_dict(pipeline.ae.encoder, ckpt["ae_encoder_lora"])

    if use_ema_weights:
        if ckpt.get("ema_codec") is not None:
            load_ema_parameters(pipeline.codec, ckpt["ema_codec"], "codec")
        else:
            print("[EMA] WARNING: ema_codec not found; using raw codec weights")
        if ckpt.get("ema_flux") is not None:
            load_ema_parameters(pipeline.flux, ckpt["ema_flux"], "FLUX LoRA")
        else:
            print("[EMA] WARNING: ema_flux not found; using raw FLUX LoRA weights")
        if has_ae_decoder_lora:
            if ckpt.get("ema_ae_decoder") is not None:
                load_ema_parameters(pipeline.ae.decoder, ckpt["ema_ae_decoder"], "AE decoder LoRA")
            else:
                print("[EMA] WARNING: ema_ae_decoder not found; using raw AE decoder LoRA weights")
        if has_ae_encoder_lora:
            if ckpt.get("ema_ae_encoder") is not None:
                load_ema_parameters(pipeline.ae.encoder, ckpt["ema_ae_encoder"], "AE encoder LoRA")
            else:
                print("[EMA] WARNING: ema_ae_encoder not found; using raw AE encoder LoRA weights")
    pipeline.cache_text_contexts(
        max_batch_size=args.batch_size,
        device=accelerator.device,
        unload_text_encoder=args.unload_text_encoder and not args.keep_text_encoder,
    )
    
    if accelerator.is_main_process:
        print(f"✓ Checkpoint weights loaded successfully")
        print("Cleaning up memory after model loading...")
        
    del ckpt
    torch.cuda.empty_cache()
    gc.collect()
    
    if accelerator.is_main_process:
        print(f"GPU Memory after cleanup: {torch.cuda.memory_allocated(accelerator.device) / 1e9:.2f} GB allocated")

    pipeline.codec, pipeline.flux = accelerator.prepare(pipeline.codec, pipeline.flux)
    pipeline.codec.eval()
    pipeline.flux.eval()

    if not args.skip_metrics:
        lpips_metric = lpips.LPIPS(net="vgg").to(accelerator.device).eval()
        dists_metric = pyiqa.create_metric('dists').to(accelerator.device).eval() if HAS_DISTS else None
    else:
        lpips_metric = None
        dists_metric = None

    for input_dir in args.input_dirs:
        dataset_name = Path(input_dir).name
        if accelerator.is_main_process:
            print(f"\n{'='*80}")
            print(f"Processing dataset: {dataset_name} ({input_dir})")
            print(f"{'='*80}")
        
        dataset = ImagePathDataset(
            input_dir,
            image_size=args.image_size,
            center_crop=args.center_crop,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        loader = accelerator.prepare(loader)

        rows = []
        sum_psnr = 0.0
        sum_ms_ssim = 0.0
        sum_lpips = 0.0
        sum_dists = 0.0
        sum_bpp = 0.0
        count = 0
        
        progress_bar = tqdm(loader, desc=f"Inference [{dataset_name}]", disable=not accelerator.is_main_process)
        
        time_pipeline = 0
        time_metrics = 0
        
        for batch in progress_bar:
            batch_start = time.time()
            images = batch["image"].to(accelerator.device)
            paths = batch["path"]

            torch.cuda.synchronize()
            t_pipeline_start = time.time()
            
            with torch.no_grad():
                with accelerator.autocast():
                    out = pipeline.forward_stage1_infer(
                        images,
                        infer_steps=args.infer_steps,
                        do_entropy_coding=args.do_entropy_coding,
                        color_fix=args.color_fix,
                        tile_latent_size=args.tile_latent_size if args.use_tile else 0,
                        tile_latent_overlap=args.tile_latent_overlap,
                    )
            
            torch.cuda.synchronize()
            t_pipeline_end = time.time()
            recon = out["x_hat"]
            time_pipeline += (t_pipeline_end - t_pipeline_start)
            
            t_metrics_start = time.time()

            for i in range(recon.shape[0]):
                img = images[i : i + 1]
                rec = recon[i : i + 1]
                p = paths[i]
                
                if args.save_images and accelerator.is_main_process:
                    stem = Path(p).stem
                    save_path_recon = os.path.join(args.output_dir, "recon", f"{stem}_recon.png")
                    save_image(rec, save_path_recon)
                    save_path_gt = os.path.join(args.output_dir, "recon", f"{stem}_gt.png")
                    save_image(img, save_path_gt)
                    
                    try:
                        from torchvision.utils import make_grid
                        comparison = torch.cat([img, rec], dim=0)
                        grid = make_grid(comparison, nrow=2, padding=4, pad_value=1.0)
                        save_path_comparison = os.path.join(args.output_dir, "recon", f"{stem}_comparison.png")
                        save_image(grid, save_path_comparison)
                    except Exception as e:
                        pass
                
                curr_psnr = psnr(rec, img)
                curr_ms_ssim = calc_ms_ssim(rec, img)
                
                if not args.skip_metrics:
                    with torch.no_grad():
                        curr_lpips = lpips_metric(rec * 2.0 - 1.0, img * 2.0 - 1.0).mean().item()
                        if dists_metric is not None:
                            curr_dists = dists_metric(rec, img).item()
                        else:
                            curr_dists = 0.0
                else:
                    curr_lpips = 0.0
                    curr_dists = 0.0
                
                h, w = img.shape[-2:]
                curr_bpp = float(out["bytes"][i]) * 8.0 / float(h * w) if args.do_entropy_coding else 0.0

                row = {
                    "dataset": dataset_name,
                    "path": p,
                    "psnr": curr_psnr,
                    "ms_ssim": curr_ms_ssim,
                    "lpips": curr_lpips,
                    "dists": curr_dists,
                    "bpp": curr_bpp,
                }
                rows.append(row)
                all_rows.append(row)

                sum_psnr += curr_psnr
                sum_ms_ssim += curr_ms_ssim
                sum_lpips += curr_lpips
                sum_dists += curr_dists
                sum_bpp += curr_bpp
                count += 1
                
                if args.verbose and accelerator.is_main_process:
                    print(f"\n  [{count}] {Path(p).name}")
                    print(f"      PSNR:    {curr_psnr:7.4f} dB")
                    print(f"      MS-SSIM: {curr_ms_ssim:7.6f}")
                    print(f"      LPIPS:   {curr_lpips:7.6f}")
                    print(f"      DISTS:   {curr_dists:7.6f}")
                    print(f"      BPP:     {curr_bpp:7.6f}")
            
            torch.cuda.synchronize()
            t_metrics_end = time.time()
            time_metrics += (t_metrics_end - t_metrics_start)
            
            if accelerator.is_main_process and count % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if accelerator.is_main_process and count % 10 == 0:
                avg_per_image = count
                print(f"\n{'='*80}")
                print(f"Performance Breakdown (avg per image, after {count} images):")
                print(f"{'='*80}")
                print(f"  Pipeline Total:    {time_pipeline/avg_per_image:.4f}s")
                print(f"  Metrics:           {time_metrics/avg_per_image:.4f}s")
                print(f"{'='*80}\n")
                
                progress_bar.set_postfix({
                    'PSNR': f'{sum_psnr/count:.2f}',
                    'MS-SSIM': f'{sum_ms_ssim/count:.4f}',
                    'BPP': f'{sum_bpp/count:.4f}',
                })

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
            print(f"  Avg PSNR:    {dataset_summary['psnr']:.4f} dB")
            print(f"  Avg MS-SSIM: {dataset_summary['ms_ssim']:.6f}")
            print(f"  Avg LPIPS:   {dataset_summary['lpips']:.6f}")
            print(f"  Avg DISTS:   {dataset_summary['dists']:.6f}")
            print(f"  Avg BPP:     {dataset_summary['bpp']:.6f}")
            print(f"{'='*80}\n")
            
            dataset_output_dir = os.path.join(args.output_dir, dataset_name)
            ensure_dir(dataset_output_dir)
            write_csv(os.path.join(dataset_output_dir, "metrics.csv"), rows)
            save_json(os.path.join(dataset_output_dir, "summary.json"), dataset_summary)
            
            if accelerator.is_main_process:
                torch.cuda.empty_cache()
                gc.collect()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process and all_rows:
        write_csv(os.path.join(args.output_dir, "metrics_all_datasets.csv"), all_rows)
        
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
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
