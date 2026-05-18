"""
FluxCodec Stage1 训练脚本 - 基于 Flow/flux_tcm_stage1_train.py 改造。
将 TCM 替换为 DiT-IC 风格的 LatentCodec + ELIC 辅助编码器。
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
# 确保当前脚本所在目录也在路径中（支持 modules.xxx 导入）
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modules.latent_codec import LatentCodec
from modules.data import (
    RecursiveImageDataset, build_dataloader,
    build_train_transform, build_val_transform,
)
from modules.lora import inject_lora, load_lora_state_dict, lora_state_dict
from modules.losses import Stage1Loss
from modules.pipeline import FluxCodecPipeline
from modules.evaluators import Stage1Evaluator
from modules.utils import AverageMeter, ensure_dir, save_checkpoint, save_json
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
    parser.add_argument("--clip_ckpt", type=str, default="/data2/luosheng/hf_models/hub/clip-vit-base-patch32")
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
    parser.add_argument("--guidance", type=float, default=1.0)

    # 损失权重
    parser.add_argument("--lambda_rate", type=float, default=0.5)
    parser.add_argument("--d1_mse", type=float, default=2.0)
    parser.add_argument("--d2_lpips", type=float, default=1.0)
    parser.add_argument("--d3_clip", type=float, default=0.1)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_regex", type=str, default=r".*")

    # LatentCodec 参数
    parser.add_argument("--codec_ch_emd", type=int, default=128, help="FLUX AE output channels")
    parser.add_argument("--codec_channel", type=int, default=320, help="Bottleneck channel")
    parser.add_argument("--codec_channel_out", type=int, default=128, help="Codec output channels (match FLUX AE)")
    parser.add_argument("--codec_num_slices", type=int, default=5)
    parser.add_argument("--codec_ckpt", type=str, default="", help="Pretrained codec checkpoint")

    # 辅助编码器/解码器开关
    parser.add_argument("--use_aux_encoder", type=int, default=1, help="1 to enable ELIC aux encoder, 0 to disable")
    parser.add_argument("--use_aux_decoder", type=int, default=1, help="1 to enable AuxDecoder residual, 0 to disable")

    # 日志/评估/保存
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--resume", action="store_true", default=False)

    return parser.parse_args()


def build_codec(args):
    return LatentCodec(
        ch_emd=args.codec_ch_emd,
        channel=args.codec_channel,
        channel_out=args.codec_channel_out,
        num_slices=args.codec_num_slices,
        use_aux_encoder=bool(args.use_aux_encoder),
        use_aux_decoder=bool(args.use_aux_decoder),
    )


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
        run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
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

    # 数据
    train_dataset = RecursiveImageDataset(args.train_root, transform=build_train_transform(args.image_size))
    val_dataset = RecursiveImageDataset(args.val_root, transform=build_val_transform(args.image_size))
    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True, drop_last=True)
    val_loader = build_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False, drop_last=False)

    # 模型
    codec = build_codec(args)
    if args.codec_ckpt:
        ckpt = torch.load(args.codec_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        codec.load_state_dict(state, strict=False)

    # ELIC 辅助编码器
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

    # 损失函数
    criterion = Stage1Loss(
        clip_path=args.clip_ckpt,
        lambda_rate=args.lambda_rate,
        d1_mse=args.d1_mse,
        d2_lpips=args.d2_lpips,
        d3_clip=args.d3_clip,
    )

    # 评估器
    evaluator = Stage1Evaluator(
        accelerator=accelerator,
        output_dir=run_dir if run_dir else args.output_dir,
        eval_batches=args.eval_batches,
    )

    # 可训练参数: codec + flux LoRA
    # 分离 aux_parameters 以用于单独的辅助优化器（用于熵模型 CDF 估计）
    codec_params = [p for n, p in pipeline.codec.named_parameters() if not n.endswith("quantiles") and p.requires_grad]
    aux_params = [p for n, p in pipeline.codec.named_parameters() if n.endswith("quantiles") and p.requires_grad]
    flux_params = [p for p in pipeline.flux.parameters() if p.requires_grad]

    trainable_params = codec_params + flux_params
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

    pipeline.flux.train()
    pipeline.codec.train()

    global_step = 0
    start_epoch = 0

    # Resume
    if args.resume:
        all_runs = sorted(glob.glob(os.path.join(args.output_dir, "run_*")))
        if all_runs:
            latest_run = all_runs[-1]
            ckpts = sorted(glob.glob(os.path.join(latest_run, "checkpoints", "stage1_step_*.pt")))
            if ckpts:
                latest_ckpt = ckpts[-1]
                if accelerator.is_main_process:
                    print(f"Resuming from checkpoint: {latest_ckpt}")
                state = torch.load(latest_ckpt, map_location="cpu")
                accelerator.unwrap_model(pipeline.codec).load_state_dict(state["codec"], strict=True)
                missing = load_lora_state_dict(accelerator.unwrap_model(pipeline.flux), state["flux_lora"])
                if accelerator.is_main_process and missing:
                    print(f"[resume] missing LoRA modules: {len(missing)}")
                optimizer.load_state_dict(state["optimizer"])
                if "aux_optimizer" in state and aux_optimizer is not None:
                    aux_optimizer.load_state_dict(state["aux_optimizer"])
                global_step = int(state.get("global_step", 0))
                start_epoch = int(state.get("epoch", 0))

    if accelerator.is_main_process:
        total_codec = sum(p.numel() for p in accelerator.unwrap_model(pipeline.codec).parameters() if p.requires_grad)
        print(f"Train images: {len(train_dataset)}")
        print(f"Val images: {len(val_dataset)}")
        print(f"LoRA injected layers: {lora_stats.injected_layers}")
        print(f"LoRA trainable params: {lora_stats.trainable_params}")
        print(f"Codec trainable params: {total_codec}")

    meters = {k: AverageMeter() for k in ["loss", "bpp", "mse", "psnr", "lpips", "clip_l2"]}

    stop = False
    current_epoch = start_epoch
    while not stop:
        for batch in train_loader:
            with accelerator.accumulate(pipeline.flux):
                with accelerator.autocast():
                    out = pipeline.forward_stage1_train(batch, train_schedule_steps=args.train_schedule_steps)
                    loss_dict = criterion(batch, out["x_hat"], out["likelihoods"])
                    loss = loss_dict["loss"]

                optimizer.zero_grad()
                accelerator.backward(loss)
                if args.grad_clip > 0:
                    params = [p for p in pipeline.flux.parameters() if p.requires_grad]
                    params += [p for p in pipeline.codec.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()

                # 优化辅助损失 (熵瓶颈 CDF 估计)
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
                        f"lpips={log_vals['lpips']:.5f} clip_l2={log_vals['clip_l2']:.5f}"
                    )
                    if args.use_tensorboard:
                        for k, v in log_vals.items():
                            writer.add_scalar(f"train/{k}", v, global_step)

            if global_step % args.eval_every == 0:
                metrics = evaluator.evaluate(
                    pipeline=pipeline,
                    criterion=criterion,
                    val_loader=val_loader,
                    global_step=global_step,
                    clip_ckpt=args.clip_ckpt,
                )
                if accelerator.is_main_process:
                    print(
                        f"[eval {global_step}] "
                        f"loss={metrics['loss']:.5f} bpp={metrics['bpp']:.5f} "
                        f"psnr={metrics['psnr']:.2f} mse={metrics['mse']:.6f} "
                        f"lpips={metrics['lpips']:.5f} clip_l2={metrics['clip_l2']:.5f}"
                    )
                    if args.use_tensorboard:
                        for k, v in metrics.items():
                            writer.add_scalar(f"val/{k}", v, global_step)

            if global_step % args.save_every == 0 and accelerator.is_main_process:
                state = {
                    "global_step": global_step,
                    "epoch": current_epoch,
                    "codec": accelerator.unwrap_model(pipeline.codec).state_dict(),
                    "flux_lora": lora_state_dict(accelerator.unwrap_model(pipeline.flux)),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    "args": vars(args),
                }
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
        state = {
            "global_step": global_step,
            "epoch": current_epoch,
            "codec": accelerator.unwrap_model(pipeline.codec).state_dict(),
            "flux_lora": lora_state_dict(accelerator.unwrap_model(pipeline.flux)),
            "optimizer": optimizer.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
            "args": vars(args),
        }
        save_checkpoint(final_path, state)
        print("Training completed.")

        if args.use_tensorboard and writer:
            writer.close()
        if log_file_handle:
            log_file_handle.close()


if __name__ == "__main__":
    main()
