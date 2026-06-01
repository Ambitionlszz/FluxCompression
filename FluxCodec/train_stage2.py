"""
FluxCodec Stage2 Training Script — GAN adversarial fine-tuning.

Follows StableCodec/src/finetune.py training pattern:
  - FluxCodec's local vision_aided_loss.Discriminator (DINOv2 backbone) for GAN
  - loss = R + λD,  where D = λ_l2*MSE + λ_lpips*LPIPS + λ_dists*DISTS + λ_gan*GAN
  - Generator step → Discriminator real step → Discriminator fake step
  - Manual LR decay at milestones
"""
import argparse
import glob
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modules.latent_codec import LatentCodec
from modules.data import (
    RecursiveImageDataset, build_dataloader,
    build_train_transform, build_val_transform, build_eval_transform,
)
from modules.discriminator import Discriminator
from modules.lora import inject_lora, inject_lora_conv, load_lora_state_dict, lora_state_dict
from modules.losses_stage2 import Stage2Loss
from modules.pipeline import FluxCodecPipeline
from modules.evaluators import Stage1Evaluator
from modules.utils import AverageMeter, EMA, ensure_dir, save_checkpoint, save_json
from elic_aux_encoder import load_elic_encoder


def parse_args():
    p = argparse.ArgumentParser(description="FluxCodec Stage2 Training (GAN)")

    # Paths
    p.add_argument("--train_root", type=str, default="/data2/luosheng/code/flux2/datasets/train")
    p.add_argument("--val_root", type=str, default="/data2/luosheng/data/Datasets/clic2024_test_image")
    p.add_argument("--output_dir", type=str, default="./outputs/fluxcodec_stage2")
    p.add_argument("--stage1_ckpt", type=str, required=True, help="Path to Stage1 checkpoint (.pt)")

    p.add_argument("--use_tensorboard", action="store_true", default=True)
    p.add_argument("--save_log_file", action="store_true", default=True)

    # Model paths
    p.add_argument("--model_name", type=str, default="flux.2-klein-4b")
    p.add_argument("--flux_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors")
    p.add_argument("--ae_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors")
    p.add_argument("--qwen_ckpt", type=str, default="/data2/luosheng/hf_models/hub/Qwen3-4B-FP8")
    p.add_argument("--elic_ckpt", type=str, default="/data2/luosheng/code/DiT-IC/checkpoints/elic_official.pth")

    # Data
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)

    # Training
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=5e-5, help="Generator LR")
    p.add_argument("--lr_disc", type=float, default=2e-5, help="Discriminator LR")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # LR decay milestones (StableCodec style)
    p.add_argument("--lr_decay_steps", type=str, default="10000,20000,35000",
                   help="Comma-separated steps for LR decay")
    p.add_argument("--lr_decay_values", type=str, default="2e-5,1e-5,1e-6",
                   help="Comma-separated LR values at each milestone")

    # FLUX denoise
    p.add_argument("--train_schedule_steps", type=int, default=100)
    p.add_argument("--train_timestep_mode", type=str, default="infer_schedule",
                   choices=["infer_schedule", "fixed", "random"],
                   help="Training timestep sampling. infer_schedule samples only timesteps used by inference.")
    p.add_argument("--train_infer_steps", type=int, default=4,
                   help="Inference-step count whose schedule is used when train_timestep_mode is infer_schedule/fixed.")
    p.add_argument("--fixed_timestep_index", type=int, default=0,
                   help="Index into inference schedule[:-1] when train_timestep_mode=fixed.")
    p.add_argument("--guidance", type=float, default=1.0)

    # Loss weights (StableCodec structure: R + λD)
    p.add_argument("--lambda_rate", type=float, default=0.5)
    p.add_argument("--lambda_l2", type=float, default=2.0)
    p.add_argument("--lambda_lpips", type=float, default=1.0)
    p.add_argument("--lambda_dists", type=float, default=1.0)
    p.add_argument("--lambda_gan", type=float, default=0.1)

    # GAN discriminator (local vision_aided_loss)
    p.add_argument("--gan_loss_type", type=str, default="multilevel_sigmoid_s")
    p.add_argument("--disc_cv_type", type=str, default="dinov2_reg")
    p.add_argument("--dinov2_repo", type=str, default=None, help="Local DINOv2 repo directory for torch.hub")
    p.add_argument("--dinov2_weights", type=str, default=None, help="Local dinov2_vitb14_reg checkpoint path")

    # Disc LR scheduler
    p.add_argument("--lr_scheduler_disc", type=str, default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=500)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_regex", type=str, default=r".*")

    # AE Decoder LoRA
    p.add_argument("--use_ae_lora", type=int, default=1, help="1 to inject LoRA into AE decoder, 0 to disable")
    p.add_argument("--ae_lora_rank", type=int, default=32)
    p.add_argument("--ae_lora_alpha", type=float, default=32.0)
    p.add_argument("--ae_lora_target_regex", type=str, default=r".*")
    p.add_argument("--use_ae_encoder_lora", type=int, default=0, help="1 to inject LoRA into AE encoder, 0 to disable")
    p.add_argument("--ae_encoder_lora_rank", type=int, default=32)
    p.add_argument("--ae_encoder_lora_alpha", type=float, default=32.0)
    p.add_argument("--ae_encoder_lora_target_regex", type=str, default=r".*")

    # EMA
    p.add_argument("--use_ema", type=int, default=1, help="1 to enable EMA, 0 to disable")
    p.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")

    # LatentCodec
    p.add_argument("--codec_ch_emd", type=int, default=128)
    p.add_argument("--codec_channel", type=int, default=320)
    p.add_argument("--codec_channel_out", type=int, default=128)
    p.add_argument("--codec_num_slices", type=int, default=5)
    p.add_argument("--use_aux_encoder", type=int, default=1)
    p.add_argument("--use_aux_decoder", type=int, default=1)
    p.add_argument("--aux_decoder_zero_init", type=int, default=0)
    p.add_argument("--elic_proj_channels", type=int, default=64, help="ELIC feature projection channels (320 -> elic_proj_channels)")

    # Logging / Eval / Save
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--eval_batches", type=int, default=24)
    p.add_argument("--eval_center_crop", action="store_true", help="Use 256 crop eval instead of full-image inference-style eval")
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--resume_ckpt", type=str, default="", help="Explicit Stage2 checkpoint to resume from")

    return p.parse_args()


def build_codec(args):
    return LatentCodec(
        ch_emd=args.codec_ch_emd, channel=args.codec_channel,
        channel_out=args.codec_channel_out, num_slices=args.codec_num_slices,
        use_aux_encoder=bool(args.use_aux_encoder),
        use_aux_decoder=bool(args.use_aux_decoder),
        aux_decoder_zero_init=bool(args.aux_decoder_zero_init),
        elic_proj_channels=args.elic_proj_channels,
    )


def find_latest_checkpoint(output_dir: str, pattern: str) -> str | None:
    ckpts = sorted(glob.glob(os.path.join(output_dir, "run_*", "checkpoints", pattern)))
    return ckpts[-1] if ckpts else None


def load_stage1_checkpoint(args, pipeline, accelerator):
    """Load Stage1 checkpoint (codec + LoRA weights)."""
    ckpt_path = args.stage1_ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Stage1 checkpoint not found: {ckpt_path}")
    if accelerator.is_main_process:
        print(f"Loading Stage1 checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")

    codec_state = state.get("codec", state.get("state_dict", {}))
    missing, unexpected = pipeline.codec.load_state_dict(codec_state, strict=False)
    if accelerator.is_main_process:
        if missing: print(f"  Codec missing keys: {len(missing)}")
        if unexpected: print(f"  Codec unexpected keys: {len(unexpected)}")

    lora_dict = state.get("flux_lora", {})
    if lora_dict:
        missing_lora = load_lora_state_dict(pipeline.flux, lora_dict)
        if accelerator.is_main_process:
            print(f"  FLUX LoRA loaded: {len(lora_dict)} tensors, missing: {len(missing_lora)}")

    if args.use_ae_lora and state.get("ae_decoder_lora"):
        missing_ae = load_lora_state_dict(pipeline.ae.decoder, state["ae_decoder_lora"])
        if accelerator.is_main_process:
            print(f"  AE Decoder LoRA loaded: {len(state['ae_decoder_lora'])} tensors, missing: {len(missing_ae)}")
    if args.use_ae_encoder_lora and state.get("ae_encoder_lora"):
        missing_ae_enc = load_lora_state_dict(pipeline.ae.encoder, state["ae_encoder_lora"])
        if accelerator.is_main_process:
            print(f"  AE Encoder LoRA loaded: {len(state['ae_encoder_lora'])} tensors, missing: {len(missing_ae_enc)}")

    return state


def build_checkpoint_state(
    global_step,
    current_epoch,
    pipeline,
    net_disc,
    optimizer,
    optimizer_disc,
    aux_optimizer,
    args,
    ema_codec,
    ema_flux,
    ema_ae_decoder,
    ema_ae_encoder,
    accelerator,
    best_metric=None,
    best_step=0,
):
    return {
        "global_step": global_step,
        "epoch": current_epoch,
        "codec": accelerator.unwrap_model(pipeline.codec).state_dict(),
        "flux_lora": lora_state_dict(accelerator.unwrap_model(pipeline.flux)),
        "ae_decoder_lora": lora_state_dict(accelerator.unwrap_model(pipeline.ae.decoder)) if args.use_ae_lora else None,
        "ae_encoder_lora": lora_state_dict(accelerator.unwrap_model(pipeline.ae.encoder)) if args.use_ae_encoder_lora else None,
        "discriminator": accelerator.unwrap_model(net_disc).state_dict(),
        "ema_codec": ema_codec.state_dict() if ema_codec is not None else None,
        "ema_flux": ema_flux.state_dict() if ema_flux is not None else None,
        "ema_ae_decoder": ema_ae_decoder.state_dict() if ema_ae_decoder is not None else None,
        "ema_ae_encoder": ema_ae_encoder.state_dict() if ema_ae_encoder is not None else None,
        "optimizer": optimizer.state_dict(),
        "optimizer_disc": optimizer_disc.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer else None,
        "best_metric": best_metric,
        "best_step": best_step,
        "args": vars(args),
    }


def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Parse LR milestones
    lr_milestones = [int(x) for x in args.lr_decay_steps.split(",")]
    lr_values = [float(x) for x in args.lr_decay_values.split(",")]

    # Setup output dirs
    run_dir = ckpt_dir = log_dir = writer = log_file_handle = None
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lr_str = str(args.lr).replace("e-0", "e-")
        lrd_str = str(args.lr_disc).replace("e-0", "e-")
        name_parts = [f"lr{lr_str}", f"ld{args.lambda_rate}", f"gan{args.lambda_gan}", f"lrd{lrd_str}"]
        if args.use_ae_lora:
            name_parts.append("ael")
        else:
            name_parts.append("noael")
        if args.use_ae_encoder_lora:
            name_parts.append("aee")
        if args.use_ema:
            name_parts.append("ema")
        else:
            name_parts.append("noema")
        name_parts.append(f"t{args.train_timestep_mode}")
        if args.train_timestep_mode in {"infer_schedule", "fixed"}:
            name_parts.append(f"i{args.train_infer_steps}")
        if args.train_timestep_mode == "fixed":
            name_parts.append(f"ti{args.fixed_timestep_index}")
        if not args.use_aux_encoder:
            name_parts.append("noauxe")
        if not args.use_aux_decoder:
            name_parts.append("noauxd")
        elif args.aux_decoder_zero_init:
            name_parts.append("auxdzero")
        run_name = f"run_{timestamp}_{'_'.join(name_parts)}"
        run_dir = os.path.join(args.output_dir, run_name)
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        log_dir = os.path.join(run_dir, "logs")
        for d in [ckpt_dir, log_dir, os.path.join(run_dir, "eval_images_multistep")]:
            os.makedirs(d, exist_ok=True)
        save_json(os.path.join(run_dir, "train_config.json"), vars(args))

        if args.save_log_file:
            log_file = os.path.join(log_dir, "train_stage2.log")
            log_file_handle = open(log_file, "w", encoding="utf-8")
            class TeeLogger:
                def __init__(self, fh, orig):
                    self.fh, self.orig = fh, orig
                def write(self, msg):
                    self.fh.write(msg); self.fh.flush(); self.orig.write(msg)
                def flush(self):
                    self.fh.flush(); self.orig.flush()
            sys.stdout = TeeLogger(log_file_handle, sys.stdout)
            sys.stderr = sys.stdout
            print(f"Run directory: {run_dir}")

        if args.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)

    set_seed(args.seed)

    # ========== Data ==========
    train_dataset = RecursiveImageDataset(args.train_root, transform=build_train_transform(args.image_size))
    val_transform = build_val_transform(args.image_size) if args.eval_center_crop else build_eval_transform()
    val_batch_size = args.batch_size if args.eval_center_crop else 1
    val_dataset = RecursiveImageDataset(args.val_root, transform=val_transform)
    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True, drop_last=True)
    val_loader = build_dataloader(val_dataset, val_batch_size, args.num_workers, shuffle=False, drop_last=False)

    # ========== Model ==========
    codec = build_codec(args)
    elic_aux = None
    if args.elic_ckpt and args.use_aux_encoder:
        elic_aux = load_elic_encoder(args.elic_ckpt, device=accelerator.device)

    pipeline = FluxCodecPipeline(
        model_name=args.model_name, flux_ckpt=args.flux_ckpt, ae_ckpt=args.ae_ckpt,
        qwen_ckpt=args.qwen_ckpt, codec=codec, elic_aux_encoder=elic_aux,
        device=accelerator.device, guidance=args.guidance,
    )

    # LoRA injection (before loading Stage1)
    lora_stats = inject_lora(
        pipeline.flux, rank=args.lora_rank, alpha=args.lora_alpha,
        dropout=args.lora_dropout, target_regex=args.lora_target_regex,
    )

    # AE Decoder LoRA
    ae_lora_stats = None
    if args.use_ae_lora:
        ae_lora_stats = inject_lora_conv(
            pipeline.ae.decoder,
            rank=args.ae_lora_rank,
            alpha=args.ae_lora_alpha,
            dropout=args.lora_dropout,
            target_regex=args.ae_lora_target_regex,
        )
    ae_encoder_lora_stats = None
    if args.use_ae_encoder_lora:
        ae_encoder_lora_stats = inject_lora_conv(
            pipeline.ae.encoder,
            rank=args.ae_encoder_lora_rank,
            alpha=args.ae_encoder_lora_alpha,
            dropout=args.lora_dropout,
            target_regex=args.ae_encoder_lora_target_regex,
        )

    # Load Stage1 weights
    load_stage1_checkpoint(args, pipeline, accelerator)

    # ========== Loss (frozen perceptual nets) ==========
    criterion = Stage2Loss(
        lambda_rate=args.lambda_rate, lambda_l2=args.lambda_l2,
        lambda_lpips=args.lambda_lpips, lambda_dists=args.lambda_dists,
        lambda_gan=args.lambda_gan,
    )

    # ========== Discriminator (local vision_aided_loss) ==========
    net_disc = Discriminator(
        cv_type=args.disc_cv_type,
        output_type="conv_multi_level",
        loss_type=args.gan_loss_type,
        device="cuda",
        dinov2_repo=args.dinov2_repo,
        dinov2_weights=args.dinov2_weights,
    )
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)  # Freeze DINOv2 backbone
    net_disc.train()

    # ========== Evaluator ==========
    evaluator = Stage1Evaluator(
        accelerator=accelerator,
        output_dir=run_dir if run_dir else args.output_dir,
        eval_batches=args.eval_batches,
        infer_steps=args.train_infer_steps,
    )

    # ========== Optimizers ==========
    # Generator: codec + FLUX LoRA + AE LoRA
    codec_params = [p for n, p in pipeline.codec.named_parameters()
                    if not n.endswith("quantiles") and p.requires_grad]
    aux_params = [p for n, p in pipeline.codec.named_parameters()
                  if n.endswith("quantiles") and p.requires_grad]
    flux_params = [p for p in pipeline.flux.parameters() if p.requires_grad]
    ae_decoder_params = [p for p in pipeline.ae.decoder.parameters() if p.requires_grad] if args.use_ae_lora else []
    ae_encoder_params = [p for p in pipeline.ae.encoder.parameters() if p.requires_grad] if args.use_ae_encoder_lora else []

    gen_params = codec_params + flux_params + ae_decoder_params + ae_encoder_params
    optimizer = torch.optim.AdamW(gen_params, lr=args.lr, weight_decay=0.0)

    # Discriminator optimizer + LR scheduler (StableCodec pattern)
    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.lr_disc, weight_decay=0.0)
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler_disc, optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_steps * accelerator.num_processes,
    )

    # Auxiliary optimizer (entropy bottleneck CDF)
    aux_optimizer = None
    if aux_params:
        aux_optimizer = torch.optim.AdamW(aux_params, lr=1e-5, weight_decay=0.0)

    # ========== Accelerate prepare ==========
    prep = [pipeline.codec, pipeline.flux, criterion, net_disc,
            optimizer, optimizer_disc, train_loader, val_loader]
    if aux_optimizer:
        prep.insert(6, aux_optimizer)
    prepared = accelerator.prepare(*prep)

    if aux_optimizer:
        (pipeline.codec, pipeline.flux, criterion, net_disc,
         optimizer, optimizer_disc, aux_optimizer, train_loader, val_loader) = prepared
    else:
        (pipeline.codec, pipeline.flux, criterion, net_disc,
         optimizer, optimizer_disc, train_loader, val_loader) = prepared

    # EMA — created after prepare so shadows are on the correct device
    ema_codec = EMA(accelerator.unwrap_model(pipeline.codec), decay=args.ema_decay) if args.use_ema else None
    ema_flux = EMA(accelerator.unwrap_model(pipeline.flux), decay=args.ema_decay) if args.use_ema else None
    ema_ae_decoder = EMA(pipeline.ae.decoder, decay=args.ema_decay) if (args.use_ema and args.use_ae_lora) else None
    ema_ae_encoder = EMA(pipeline.ae.encoder, decay=args.ema_decay) if (args.use_ema and args.use_ae_encoder_lora) else None

    pipeline.flux.train()
    pipeline.codec.train()

    # Disable fused_attn in disc (StableCodec compatibility)
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    global_step = 0
    start_epoch = 0
    best_metric = None
    best_step = 0

    # ========== Resume from Stage2 checkpoint ==========
    if args.resume:
        latest_ckpt = args.resume_ckpt or find_latest_checkpoint(args.output_dir, "stage2_step_*.pt")
        if latest_ckpt:
            if accelerator.is_main_process:
                print(f"Resuming Stage2 from: {latest_ckpt}")
            state = torch.load(latest_ckpt, map_location="cpu")
            accelerator.unwrap_model(pipeline.codec).load_state_dict(state["codec"])
            load_lora_state_dict(accelerator.unwrap_model(pipeline.flux), state["flux_lora"])
            if args.use_ae_lora and state.get("ae_decoder_lora"):
                load_lora_state_dict(accelerator.unwrap_model(pipeline.ae.decoder), state["ae_decoder_lora"])
            if args.use_ae_encoder_lora and state.get("ae_encoder_lora"):
                load_lora_state_dict(accelerator.unwrap_model(pipeline.ae.encoder), state["ae_encoder_lora"])
            optimizer.load_state_dict(state["optimizer"])
            optimizer_disc.load_state_dict(state["optimizer_disc"])
            if "discriminator" in state:
                accelerator.unwrap_model(net_disc).load_state_dict(state["discriminator"])
            if "aux_optimizer" in state and aux_optimizer:
                aux_optimizer.load_state_dict(state["aux_optimizer"])
            if args.use_ema and "ema_codec" in state:
                ema_codec.load_state_dict(state["ema_codec"])
                ema_flux.load_state_dict(state["ema_flux"])
                if ema_ae_decoder is not None and state.get("ema_ae_decoder"):
                    ema_ae_decoder.load_state_dict(state["ema_ae_decoder"])
                if ema_ae_encoder is not None and state.get("ema_ae_encoder"):
                    ema_ae_encoder.load_state_dict(state["ema_ae_encoder"])
            global_step = int(state.get("global_step", 0))
            start_epoch = int(state.get("epoch", 0))
            best_metric = state.get("best_metric", None)
            best_step = int(state.get("best_step", 0))
        elif accelerator.is_main_process:
            print(f"[resume] No Stage2 checkpoint found under {args.output_dir}")

    if accelerator.is_main_process:
        total_codec = sum(p.numel() for p in accelerator.unwrap_model(pipeline.codec).parameters() if p.requires_grad)
        disc_params = sum(p.numel() for p in net_disc.parameters() if p.requires_grad)
        print(f"=== Stage2 GAN Training (StableCodec pattern) ===")
        print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
        print(f"Eval mode: {'center crop' if args.eval_center_crop else 'full image'} (batch_size={val_batch_size})")
        print(f"FLUX LoRA: {lora_stats.injected_layers} layers, {lora_stats.trainable_params} params")
        if ae_lora_stats is not None:
            print(f"AE Decoder LoRA: {ae_lora_stats.injected_layers} layers, {ae_lora_stats.trainable_params} params")
        if ae_encoder_lora_stats is not None:
            print(f"AE Encoder LoRA: {ae_encoder_lora_stats.injected_layers} layers, {ae_encoder_lora_stats.trainable_params} params")
        print(f"Codec trainable: {total_codec} | Disc trainable: {disc_params}")
        print(
            f"Train timestep mode: {args.train_timestep_mode} "
            f"(train_schedule_steps={args.train_schedule_steps}, train_infer_steps={args.train_infer_steps}, "
            f"fixed_timestep_index={args.fixed_timestep_index})"
        )
        print(f"Gen LR: {args.lr} | Disc LR: {args.lr_disc}")
        print(f"Loss: R*{args.lambda_rate} + L2*{args.lambda_l2} + LPIPS*{args.lambda_lpips} "
              f"+ DISTS*{args.lambda_dists} + GAN*{args.lambda_gan}")

    # ========== Training loop (StableCodec finetune.py pattern) ==========
    meter_keys = ["loss", "bpp", "mse", "psnr", "lpips", "dists", "g_loss", "d_real", "d_fake"]
    meters = {k: AverageMeter() for k in meter_keys}

    stop = False
    current_epoch = start_epoch
    while not stop:
        for batch in train_loader:
            # --- Manual LR decay (StableCodec style) ---
            for ms, lrv in zip(lr_milestones, lr_values):
                if global_step == ms:
                    for pg in optimizer.param_groups:
                        pg["lr"] = lrv
                    if accelerator.is_main_process:
                        print(f"[step {global_step}] Generator LR → {lrv}")

            l_acc = [pipeline.codec, pipeline.flux, net_disc]

            with accelerator.accumulate(*l_acc):
                with accelerator.autocast():
                    # Forward pass
                    out = pipeline.forward_stage1_train(
                        batch,
                        train_schedule_steps=args.train_schedule_steps,
                        timestep_mode=args.train_timestep_mode,
                        train_infer_steps=args.train_infer_steps,
                        fixed_timestep_index=args.fixed_timestep_index,
                    )
                    x = batch.detach().float()
                    x_hat = out["x_hat"].float()

                    # --- Generator step ---
                    # GAN generator loss: fool discriminator
                    loss_adv = net_disc(x_hat, for_G=True).mean()

                    # R + λD (including GAN)
                    loss_dict = accelerator.unwrap_model(criterion)(
                        x, x_hat, out["likelihoods"], loss_adv=loss_adv,
                    )
                    gen_loss = loss_dict["loss"]

                accelerator.backward(gen_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(gen_params, args.grad_clip)
                optimizer.step()

                # Update EMA
                if args.use_ema:
                    ema_codec.update(pipeline.codec)
                    ema_flux.update(pipeline.flux)
                    if ema_ae_decoder is not None:
                        ema_ae_decoder.update(pipeline.ae.decoder)
                    if ema_ae_encoder is not None:
                        ema_ae_encoder.update(pipeline.ae.encoder)

                optimizer.zero_grad(set_to_none=True)

                # Auxiliary loss (entropy bottleneck CDF)
                if aux_optimizer is not None:
                    aux_loss = accelerator.unwrap_model(pipeline.codec).aux_loss()
                    aux_optimizer.zero_grad()
                    accelerator.backward(aux_loss)
                    aux_optimizer.step()

                # --- Discriminator real step (StableCodec pattern) ---
                loss_real = net_disc(x.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(loss_real)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.grad_clip)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=True)

                # --- Discriminator fake step (StableCodec pattern) ---
                loss_fake = net_disc(x_hat.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(loss_fake)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.grad_clip)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=True)

            # Update meters
            bs = batch.shape[0]
            for k in loss_dict:
                if k in meters:
                    meters[k].update(loss_dict[k].item(), bs)
            meters["g_loss"].update(loss_adv.item(), bs)
            meters["d_real"].update(loss_real.item(), bs)
            meters["d_fake"].update(loss_fake.item(), bs)

            global_step += 1

            # === Logging ===
            if global_step % args.log_every == 0:
                log_vals = {}
                for k, meter in meters.items():
                    t = torch.tensor([meter.sum, meter.count], device=accelerator.device, dtype=torch.float64)
                    t = accelerator.reduce(t, reduction="sum")
                    log_vals[k] = (t[0] / max(t[1], 1)).item()
                    meter.reset()
                if accelerator.is_main_process:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"[step {global_step}] lr={cur_lr:.2e} "
                        f"loss={log_vals['loss']:.5f} bpp={log_vals['bpp']:.5f} "
                        f"psnr={log_vals['psnr']:.2f} mse={log_vals['mse']:.6f} "
                        f"lpips={log_vals['lpips']:.5f} dists={log_vals['dists']:.5f} "
                        f"g={log_vals['g_loss']:.5f} d_r={log_vals['d_real']:.5f} d_f={log_vals['d_fake']:.5f}"
                    )
                    if writer:
                        for k, v in log_vals.items():
                            writer.add_scalar(f"train/{k}", v, global_step)

            # === Evaluation ===
            if global_step % args.eval_every == 0:
                if args.use_ema:
                    ema_codec.apply_shadow(pipeline.codec)
                    ema_flux.apply_shadow(pipeline.flux)
                    if ema_ae_decoder is not None:
                        ema_ae_decoder.apply_shadow(pipeline.ae.decoder)
                    if ema_ae_encoder is not None:
                        ema_ae_encoder.apply_shadow(pipeline.ae.encoder)

                metrics = evaluator.evaluate(
                    pipeline=pipeline, criterion=criterion,
                    val_loader=val_loader, global_step=global_step,
                )

                if args.use_ema:
                    ema_codec.restore(pipeline.codec)
                    ema_flux.restore(pipeline.flux)
                    if ema_ae_decoder is not None:
                        ema_ae_decoder.restore(pipeline.ae.decoder)
                    if ema_ae_encoder is not None:
                        ema_ae_encoder.restore(pipeline.ae.encoder)

                if accelerator.is_main_process:
                    print(
                        f"[eval {global_step}] "
                        f"loss={metrics['loss']:.5f} bpp={metrics['bpp']:.5f} "
                        f"psnr={metrics['psnr']:.2f} lpips={metrics['lpips']:.5f} dists={metrics['dists']:.5f}"
                    )
                    if writer:
                        for k, v in metrics.items():
                            writer.add_scalar(f"val/{k}", v, global_step)
                    eval_metric = float(metrics["loss"])
                    if best_metric is None or eval_metric < best_metric:
                        best_metric = eval_metric
                        best_step = global_step
                        best_path = os.path.join(ckpt_dir, "stage2_best.pt")
                        state = build_checkpoint_state(
                            global_step,
                            current_epoch,
                            pipeline,
                            net_disc,
                            optimizer,
                            optimizer_disc,
                            aux_optimizer,
                            args,
                            ema_codec,
                            ema_flux,
                            ema_ae_decoder,
                            ema_ae_encoder,
                            accelerator,
                            best_metric=best_metric,
                            best_step=best_step,
                        )
                        save_checkpoint(best_path, state)
                        print(f"Saved best checkpoint to {best_path} (eval_loss={best_metric:.6f}, step={best_step})")

            # === Save ===
            if global_step % args.save_every == 0 and accelerator.is_main_process:
                state = build_checkpoint_state(
                    global_step,
                    current_epoch,
                    pipeline,
                    net_disc,
                    optimizer,
                    optimizer_disc,
                    aux_optimizer,
                    args,
                    ema_codec,
                    ema_flux,
                    ema_ae_decoder,
                    ema_ae_encoder,
                    accelerator,
                    best_metric=best_metric,
                    best_step=best_step,
                )
                path = os.path.join(ckpt_dir, f"stage2_step_{global_step:08d}.pt")
                save_checkpoint(path, state)
                print(f"Saved checkpoint: {path}")

            if global_step >= args.max_steps:
                stop = True
                break
        current_epoch += 1

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final = os.path.join(ckpt_dir, "stage2_last.pt") if ckpt_dir else os.path.join(args.output_dir, "stage2_last.pt")
        state = build_checkpoint_state(
            global_step,
            current_epoch,
            pipeline,
            net_disc,
            optimizer,
            optimizer_disc,
            aux_optimizer,
            args,
            ema_codec,
            ema_flux,
            ema_ae_decoder,
            ema_ae_encoder,
            accelerator,
            best_metric=best_metric,
            best_step=best_step,
        )
        save_checkpoint(final, state)
        if best_metric is not None:
            print(f"Best checkpoint: {os.path.join(ckpt_dir, 'stage2_best.pt')} (eval_loss={best_metric:.6f}, step={best_step})")
        else:
            best = os.path.join(ckpt_dir, "stage2_best.pt") if ckpt_dir else os.path.join(args.output_dir, "stage2_best.pt")
            save_checkpoint(best, state)
            print(f"No eval was run; saved final weights as best checkpoint: {best}")
        print("Stage2 training completed.")
        if writer: writer.close()
        if log_file_handle:
            tee_logger = sys.stdout
            if hasattr(tee_logger, "orig"):
                sys.stdout = tee_logger.orig
                sys.stderr = tee_logger.orig
            log_file_handle.close()


if __name__ == "__main__":
    main()
