"""
FluxCodec Stage1 Training Script.
"""
import argparse
import glob
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torchvision
from accelerate import Accelerator
from accelerate.utils import set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
# Support import from modules.xxx
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modules.latent_codec import LatentCodec
from modules.data import (
    RecursiveImageDataset, build_dataloader,
    build_train_transform, build_val_transform, build_eval_transform,
)
from modules.lora import inject_lora, inject_lora_conv, load_lora_state_dict, lora_state_dict
from modules.losses import Stage1Loss
from modules.pipeline import FluxCodecPipeline
from modules.evaluators import Stage1Evaluator
from modules.utils import AverageMeter, EMA, ensure_dir, save_checkpoint, save_json
from elic_aux_encoder import load_elic_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="FluxCodec Stage1 Training")

    parser.add_argument("--train_root", type=str, default="/data2/luosheng/code/flux2/datasets/train")
    parser.add_argument("--val_root", type=str, default="/data2/luosheng/data/Datasets/clic2024_test_image")
    parser.add_argument("--output_dir", type=str, default="./outputs/fluxcodec_stage1")

    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--log_dir", type=str, default="./outputs/logs")
    parser.add_argument("--save_log_file", action="store_true", default=True)

    parser.add_argument("--model_name", type=str, default="flux.2-klein-4b")
    parser.add_argument("--flux_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors")
    parser.add_argument("--ae_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors")
    parser.add_argument("--qwen_ckpt", type=str, default="/data2/luosheng/hf_models/hub/Qwen3-4B-FP8")
    parser.add_argument("--elic_ckpt", type=str, default="/data2/luosheng/code/DiT-IC/checkpoints/elic_official.pth", help="Path to pretrained ELIC checkpoint")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--train_schedule_steps", type=int, default=100)
    parser.add_argument("--train_timestep_mode", type=str, default="infer_schedule",
                        choices=["infer_schedule", "fixed", "random"],
                        help="Training timestep sampling. infer_schedule samples only timesteps used by inference.")
    parser.add_argument("--train_infer_steps", type=int, default=4,
                        help="Inference-step count whose schedule is used when train_timestep_mode is infer_schedule/fixed.")
    parser.add_argument("--fixed_timestep_index", type=int, default=0,
                        help="Index into inference schedule[:-1] when train_timestep_mode=fixed.")
    parser.add_argument("--guidance", type=float, default=1.0)

    # Loss weights
    parser.add_argument("--lambda_rate", type=float, default=0.5)
    parser.add_argument("--d1_mse", type=float, default=2.0)
    parser.add_argument("--d2_lpips", type=float, default=1.0)
    parser.add_argument("--d3_dists", type=float, default=1.0)
    parser.add_argument("--d3_clip", type=float, default=0.1)
    parser.add_argument("--d3_metric", type=str, default="dists", choices=["dists", "clip"])
    parser.add_argument("--clip_ckpt", type=str, default="/data2/luosheng/hf_models/hub/clip-vit-base-patch32")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_regex", type=str, default=r".*")

    # AE Decoder LoRA
    parser.add_argument("--use_ae_lora", type=int, default=1, help="1 to inject LoRA into AE decoder, 0 to disable")
    parser.add_argument("--ae_lora_rank", type=int, default=32)
    parser.add_argument("--ae_lora_alpha", type=float, default=32.0)
    parser.add_argument("--ae_lora_target_regex", type=str, default=r".*")
    parser.add_argument("--use_ae_encoder_lora", type=int, default=0, help="1 to inject LoRA into AE encoder, 0 to disable")
    parser.add_argument("--ae_encoder_lora_rank", type=int, default=32)
    parser.add_argument("--ae_encoder_lora_alpha", type=float, default=32.0)
    parser.add_argument("--ae_encoder_lora_target_regex", type=str, default=r".*")

    # EMA
    parser.add_argument("--use_ema", type=int, default=1, help="1 to enable EMA, 0 to disable")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")

    # LatentCodec 参数
    parser.add_argument("--codec_ch_emd", type=int, default=128, help="FLUX AE output channels")
    parser.add_argument("--codec_channel", type=int, default=320, help="Bottleneck channel")
    parser.add_argument("--codec_channel_out", type=int, default=128, help="Codec output channels (match FLUX AE)")
    parser.add_argument("--codec_num_slices", type=int, default=5)
    parser.add_argument("--codec_ckpt", type=str, default="", help="Pretrained codec checkpoint")

    # Aux encoder/decoder flags
    parser.add_argument("--use_aux_encoder", type=int, default=1, help="1 to enable ELIC aux encoder, 0 to disable")
    parser.add_argument("--use_aux_decoder", type=int, default=1, help="1 to enable AuxDecoder residual, 0 to disable")
    parser.add_argument("--aux_decoder_zero_init", type=int, default=0, help="1 to zero-init AuxDecoder final conv, 0 for default Conv2d init")
    parser.add_argument("--elic_proj_channels", type=int, default=64, help="ELIC feature projection channels (320 -> elic_proj_channels)")

    # Logging/Evaluation/Saving
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--eval_batches", type=int, default=24)
    parser.add_argument("--eval_center_crop", action="store_true", help="Use 256 crop eval instead of full-image inference-style eval")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_ckpt", type=str, default="", help="Explicit Stage1 checkpoint to resume from")

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


def find_latest_checkpoint(output_dir: str, pattern: str) -> str | None:
    ckpts = sorted(glob.glob(os.path.join(output_dir, "run_*", "checkpoints", pattern)))
    return ckpts[-1] if ckpts else None


def _extract_codec_state(ckpt: dict) -> dict:
    if isinstance(ckpt.get("codec"), dict):
        return ckpt["codec"]
    if isinstance(ckpt.get("state_dict"), dict):
        return ckpt["state_dict"]
    return ckpt


def build_checkpoint_state(
    global_step,
    current_epoch,
    pipeline,
    optimizer,
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
        "ema_codec": ema_codec.state_dict() if ema_codec is not None else None,
        "ema_flux": ema_flux.state_dict() if ema_flux is not None else None,
        "ema_ae_decoder": ema_ae_decoder.state_dict() if ema_ae_decoder is not None else None,
        "ema_ae_encoder": ema_ae_encoder.state_dict() if ema_ae_encoder is not None else None,
        "optimizer": optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
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

    # Setup output directories
    run_dir = None
    ckpt_dir = None
    log_dir = None
    writer = None
    log_file_handle = None

    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lr_str = str(args.lr).replace("e-0", "e-")
        name_parts = [f"lr{lr_str}", f"ld{args.lambda_rate}"]
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
        os.makedirs(run_dir, exist_ok=True)

        ckpt_dir = os.path.join(run_dir, "checkpoints")
        log_dir = os.path.join(run_dir, "logs")
        eval_image_dir = os.path.join(run_dir, "eval_images_multistep")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(eval_image_dir, exist_ok=True)

        save_json(os.path.join(run_dir, "train_config.json"), vars(args))

        if args.save_log_file:
            log_file = os.path.join(log_dir, "train.log")
            log_file_handle = open(log_file, "w", encoding="utf-8")

            class TeeLogger:
                def __init__(self, fh, orig):
                    self.file_handle = fh
                    self.original_stdout = orig
                def write(self, msg):
                    self.file_handle.write(msg)
                    self.file_handle.flush()
                    self.original_stdout.write(msg)
                def flush(self):
                    self.file_handle.flush()
                    self.original_stdout.flush()

            sys.stdout = TeeLogger(log_file_handle, sys.stdout)
            sys.stderr = sys.stdout
            print(f"Run directory: {run_dir}")
            print(f"Terminal output will be saved to: {log_file}")
        else:
            print(f"Run directory: {run_dir}")

        if args.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs will be saved to: {log_dir}")

    set_seed(args.seed)

    # Datasets and Dataloaders
    train_dataset = RecursiveImageDataset(args.train_root, transform=build_train_transform(args.image_size))
    val_transform = build_val_transform(args.image_size) if args.eval_center_crop else build_eval_transform()
    val_batch_size = args.batch_size if args.eval_center_crop else 1
    val_dataset = RecursiveImageDataset(args.val_root, transform=val_transform)
    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True, drop_last=True)
    val_loader = build_dataloader(val_dataset, val_batch_size, args.num_workers, shuffle=False, drop_last=False)

    # Model
    codec = build_codec(args)
    if args.codec_ckpt:
        ckpt = torch.load(args.codec_ckpt, map_location="cpu")
        state = _extract_codec_state(ckpt)
        codec.load_state_dict(state, strict=False)

    # ELIC Auxiliary Encoder
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
    )

    # LoRA
    lora_stats = inject_lora(
        pipeline.flux,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_regex=args.lora_target_regex,
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

    # Loss Function
    criterion = Stage1Loss(
        lambda_rate=args.lambda_rate,
        d1_mse=args.d1_mse,
        d2_lpips=args.d2_lpips,
        d3_dists=args.d3_dists,
        d3_clip=args.d3_clip,
        d3_metric=args.d3_metric,
        clip_path=args.clip_ckpt,
    )

    # Evaluator
    evaluator = Stage1Evaluator(
        accelerator=accelerator,
        output_dir=run_dir if run_dir else args.output_dir,
        eval_batches=args.eval_batches,
        infer_steps=args.train_infer_steps,
    )

    # Trainable parameters: codec + FLUX LoRA + AE LoRA
    # Separate aux_parameters for separate auxiliary optimizer (entropy CDF estimation)
    codec_params = [p for n, p in pipeline.codec.named_parameters() if not n.endswith("quantiles") and p.requires_grad]
    aux_params = [p for n, p in pipeline.codec.named_parameters() if n.endswith("quantiles") and p.requires_grad]
    flux_params = [p for p in pipeline.flux.parameters() if p.requires_grad]
    ae_decoder_params = [p for p in pipeline.ae.decoder.parameters() if p.requires_grad] if args.use_ae_lora else []
    ae_encoder_params = [p for p in pipeline.ae.encoder.parameters() if p.requires_grad] if args.use_ae_encoder_lora else []

    trainable_params = codec_params + flux_params + ae_decoder_params + ae_encoder_params
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    aux_optimizer = None
    if aux_params:
        aux_optimizer = torch.optim.AdamW(aux_params, lr=1e-5, weight_decay=0.0)

    # Accelerate prepare
    if aux_optimizer is not None:
        prepared = accelerator.prepare(
            pipeline.codec, pipeline.flux, criterion, optimizer, aux_optimizer, train_loader, val_loader,
        )
        pipeline.codec, pipeline.flux, criterion, optimizer, aux_optimizer, train_loader, val_loader = prepared
    else:
        prepared = accelerator.prepare(
            pipeline.codec, pipeline.flux, criterion, optimizer, train_loader, val_loader,
        )
        pipeline.codec, pipeline.flux, criterion, optimizer, train_loader, val_loader = prepared

    # EMA — created after prepare so shadows are on the correct device
    ema_codec = EMA(accelerator.unwrap_model(pipeline.codec), decay=args.ema_decay) if args.use_ema else None
    ema_flux = EMA(accelerator.unwrap_model(pipeline.flux), decay=args.ema_decay) if args.use_ema else None
    ema_ae_decoder = EMA(pipeline.ae.decoder, decay=args.ema_decay) if (args.use_ema and args.use_ae_lora) else None
    ema_ae_encoder = EMA(pipeline.ae.encoder, decay=args.ema_decay) if (args.use_ema and args.use_ae_encoder_lora) else None

    pipeline.flux.train()
    pipeline.codec.train()

    global_step = 0
    start_epoch = 0
    best_metric = None
    best_step = 0

    # Resume
    if args.resume:
        latest_ckpt = args.resume_ckpt or find_latest_checkpoint(args.output_dir, "stage1_step_*.pt")
        if latest_ckpt:
            if accelerator.is_main_process:
                print(f"Resuming from checkpoint: {latest_ckpt}")
            state = torch.load(latest_ckpt, map_location="cpu")
            accelerator.unwrap_model(pipeline.codec).load_state_dict(state["codec"], strict=True)
            missing = load_lora_state_dict(accelerator.unwrap_model(pipeline.flux), state["flux_lora"])
            if accelerator.is_main_process and missing:
                print(f"[resume] missing FLUX LoRA modules: {len(missing)}")
            if args.use_ae_lora and state.get("ae_decoder_lora"):
                missing_ae = load_lora_state_dict(accelerator.unwrap_model(pipeline.ae.decoder), state["ae_decoder_lora"])
                if accelerator.is_main_process and missing_ae:
                    print(f"[resume] missing AE decoder LoRA modules: {len(missing_ae)}")
            if args.use_ae_encoder_lora and state.get("ae_encoder_lora"):
                missing_ae_enc = load_lora_state_dict(accelerator.unwrap_model(pipeline.ae.encoder), state["ae_encoder_lora"])
                if accelerator.is_main_process and missing_ae_enc:
                    print(f"[resume] missing AE encoder LoRA modules: {len(missing_ae_enc)}")
            optimizer.load_state_dict(state["optimizer"])
            if "aux_optimizer" in state and aux_optimizer is not None:
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
            print(f"[resume] No Stage1 checkpoint found under {args.output_dir}")

    if accelerator.is_main_process:
        total_codec = sum(p.numel() for p in accelerator.unwrap_model(pipeline.codec).parameters() if p.requires_grad)
        print(f"Train images: {len(train_dataset)}")
        print(f"Val images: {len(val_dataset)}")
        print(f"Eval mode: {'center crop' if args.eval_center_crop else 'full image'} (batch_size={val_batch_size})")
        print(f"FLUX LoRA injected layers: {lora_stats.injected_layers}")
        print(f"FLUX LoRA trainable params: {lora_stats.trainable_params}")
        if ae_lora_stats is not None:
            print(f"AE Decoder LoRA injected layers: {ae_lora_stats.injected_layers}")
            print(f"AE Decoder LoRA trainable params: {ae_lora_stats.trainable_params}")
        if ae_encoder_lora_stats is not None:
            print(f"AE Encoder LoRA injected layers: {ae_encoder_lora_stats.injected_layers}")
            print(f"AE Encoder LoRA trainable params: {ae_encoder_lora_stats.trainable_params}")
        print(f"Codec trainable params: {total_codec}")
        print(
            f"Train timestep mode: {args.train_timestep_mode} "
            f"(train_schedule_steps={args.train_schedule_steps}, train_infer_steps={args.train_infer_steps}, "
            f"fixed_timestep_index={args.fixed_timestep_index})"
        )
        d3_weight = args.d3_dists if args.d3_metric == "dists" else args.d3_clip
        print(f"Stage1 d3 metric: {args.d3_metric} (weight={d3_weight})")

    meters = {k: AverageMeter() for k in ["loss", "bpp", "mse", "psnr", "lpips", "d3_loss"]}

    stop = False
    current_epoch = start_epoch
    while not stop:
        for batch in train_loader:
            with accelerator.accumulate(pipeline.flux):
                with accelerator.autocast():
                    out = pipeline.forward_stage1_train(
                        batch,
                        train_schedule_steps=args.train_schedule_steps,
                        timestep_mode=args.train_timestep_mode,
                        train_infer_steps=args.train_infer_steps,
                        fixed_timestep_index=args.fixed_timestep_index,
                    )
                    loss_dict = criterion(batch, out["x_hat"], out["likelihoods"])
                    loss = loss_dict["loss"]

                optimizer.zero_grad()
                accelerator.backward(loss)
                if args.grad_clip > 0:
                    params = [p for p in pipeline.flux.parameters() if p.requires_grad]
                    params += [p for n, p in pipeline.codec.named_parameters() if p.requires_grad and not n.endswith("quantiles")]
                    if args.use_ae_lora:
                        params += [p for p in pipeline.ae.decoder.parameters() if p.requires_grad]
                    if args.use_ae_encoder_lora:
                        params += [p for p in pipeline.ae.encoder.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()

                # Update EMA
                if args.use_ema:
                    ema_codec.update(pipeline.codec)
                    ema_flux.update(pipeline.flux)
                    if ema_ae_decoder is not None:
                        ema_ae_decoder.update(pipeline.ae.decoder)
                    if ema_ae_encoder is not None:
                        ema_ae_encoder.update(pipeline.ae.encoder)

                # Optimize auxiliary loss (entropy bottleneck CDF estimation)
                if aux_optimizer is not None:
                    aux_loss = accelerator.unwrap_model(pipeline.codec).aux_loss()
                    aux_optimizer.zero_grad()
                    accelerator.backward(aux_loss)
                    aux_optimizer.step()

            bs = batch.shape[0]
            for k in meters:
                meters[k].update(loss_dict[k].item(), bs)

            global_step += 1

            if global_step % args.log_every == 0:
                log_vals = {}
                for k, meter in meters.items():
                    tensor = torch.tensor([meter.sum, meter.count], device=accelerator.device, dtype=torch.float64)
                    tensor = accelerator.reduce(tensor, reduction="sum")
                    log_vals[k] = (tensor[0] / max(tensor[1], 1)).item()
                    meter.reset()
                if accelerator.is_main_process:
                    print(
                        f"[step {global_step}] "
                        f"loss={log_vals['loss']:.5f} bpp={log_vals['bpp']:.5f} "
                        f"psnr={log_vals['psnr']:.2f} mse={log_vals['mse']:.6f} "
                        f"lpips={log_vals['lpips']:.5f} {args.d3_metric}={log_vals['d3_loss']:.5f}"
                    )
                    if args.use_tensorboard:
                        for k, v in log_vals.items():
                            writer.add_scalar(f"train/{k}", v, global_step)

            if global_step % args.eval_every == 0:
                if args.use_ema:
                    ema_codec.apply_shadow(pipeline.codec)
                    ema_flux.apply_shadow(pipeline.flux)
                    if ema_ae_decoder is not None:
                        ema_ae_decoder.apply_shadow(pipeline.ae.decoder)
                    if ema_ae_encoder is not None:
                        ema_ae_encoder.apply_shadow(pipeline.ae.encoder)

                metrics = evaluator.evaluate(
                    pipeline=pipeline,
                    criterion=criterion,
                    val_loader=val_loader,
                    global_step=global_step,
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
                        f"psnr={metrics['psnr']:.2f} mse={metrics['mse']:.6f} "
                        f"lpips={metrics['lpips']:.5f} dists={metrics['dists']:.5f}"
                    )
                    if args.use_tensorboard:
                        for k, v in metrics.items():
                            writer.add_scalar(f"val/{k}", v, global_step)
                    eval_metric = float(metrics["loss"])
                    if best_metric is None or eval_metric < best_metric:
                        best_metric = eval_metric
                        best_step = global_step
                        best_path = os.path.join(ckpt_dir, "stage1_best.pt")
                        state = build_checkpoint_state(
                            global_step,
                            current_epoch,
                            pipeline,
                            optimizer,
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

            if global_step % args.save_every == 0 and accelerator.is_main_process:
                state = build_checkpoint_state(
                    global_step,
                    current_epoch,
                    pipeline,
                    optimizer,
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
                ckpt_path = os.path.join(ckpt_dir, f"stage1_step_{global_step:08d}.pt")
                save_checkpoint(ckpt_path, state)
                print(f"Saved checkpoint to {ckpt_path}")

            if global_step >= args.max_steps:
                stop = True
                break

        current_epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(ckpt_dir, "stage1_last.pt") if ckpt_dir else os.path.join(args.output_dir, "stage1_last.pt")
        state = build_checkpoint_state(
            global_step,
            current_epoch,
            pipeline,
            optimizer,
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
        save_checkpoint(final_path, state)
        if best_metric is not None:
            print(f"Best checkpoint: {os.path.join(ckpt_dir, 'stage1_best.pt')} (eval_loss={best_metric:.6f}, step={best_step})")
        else:
            best_path = os.path.join(ckpt_dir, "stage1_best.pt") if ckpt_dir else os.path.join(args.output_dir, "stage1_best.pt")
            save_checkpoint(best_path, state)
            print(f"No eval was run; saved final weights as best checkpoint: {best_path}")
        print("Training completed.")

        if args.use_tensorboard and writer:
            writer.close()
        if log_file_handle:
            tee_logger = sys.stdout
            if hasattr(tee_logger, "original_stdout"):
                sys.stdout = tee_logger.original_stdout
                sys.stderr = tee_logger.original_stdout
            log_file_handle.close()


if __name__ == "__main__":
    main()
