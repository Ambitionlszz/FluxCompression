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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from LIC_TCM.models import TCMLatent
from Flow.modules.data import (
    RecursiveImageDataset,
    build_dataloader,
    build_train_transform,
    build_val_transform,
)
from Flow.modules.lora import inject_lora, load_lora_state_dict, lora_state_dict
from Flow.modules.losses import Stage1Loss
from Flow.modules.pipeline import FlowTCMStage1Pipeline
from Flow.modules.evaluators import Stage1Evaluator
from Flow.modules.utils import AverageMeter, ensure_dir, save_checkpoint, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1 training for FLUX.2[Klein] + TCM with Accelerate")

    parser.add_argument("--train_root", type=str, default="/data2/luosheng/code/flux2/datasets/train")
    parser.add_argument("--val_root", type=str, default="/data2/luosheng/data/Datasets/clic2024_test_image")
    parser.add_argument("--output_dir", type=str, default="./outputs/stage1")

    # 新增参数：tensorboard 和日志
    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--log_dir", type=str, default="./outputs/logs")
    parser.add_argument("--save_log_file", action="store_true", default=True)

    parser.add_argument("--model_name", type=str, default="flux.2-klein-4b")
    parser.add_argument("--flux_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors")
    parser.add_argument("--ae_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors")
    parser.add_argument("--qwen_ckpt", type=str, default="/data2/luosheng/hf_models/hub/Qwen3-4B-FP8")
    parser.add_argument("--clip_ckpt", type=str, default="/data2/luosheng/hf_models/hub/clip-vit-base-patch32")

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

    parser.add_argument("--lambda_rate", type=float, default=0.5)
    parser.add_argument("--d1_mse", type=float, default=2.0)
    parser.add_argument("--d2_lpips", type=float, default=1.0)
    parser.add_argument("--d3_clip", type=float, default=0.1)

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
    parser.add_argument("--tcm_ckpt", type=str, default="")
    
    # DiT-IC 风格架构选项
    parser.add_argument("--use_ditic_style", action="store_true", 
                        help="Use DiT-IC style architecture (LatentCodecDiTIC) instead of original TCMLatent")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--resume", action="store_true", default=False)
    
    return parser.parse_args()


def build_tcm(args):
    tcm = TCMLatent(
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
    return tcm


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
    original_write = None
    original_flush = None
    log_file_handle = None

    if accelerator.is_main_process:
        # Create a timestamped run directory inside output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Subdirectories for checkpoints, logs, and eval images
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        log_dir = os.path.join(run_dir, "logs")
        eval_image_dir = os.path.join(run_dir, "eval_images_multistep")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(eval_image_dir, exist_ok=True)
        
        # Save config
        save_json(os.path.join(run_dir, "train_config.json"), vars(args))

        # Setup logging
        if args.save_log_file:
            log_file = os.path.join(log_dir, "train.log")
            original_write = sys.stdout.write
            original_flush = sys.stdout.flush
            log_file_handle = open(log_file, "w", encoding="utf-8")
            
            class TeeLogger:
                def __init__(self, file_handle, original_stdout):
                    self.file_handle = file_handle
                    self.original_stdout = original_stdout
                
                def write(self, message):
                    self.file_handle.write(message)
                    self.file_handle.flush()
                    self.original_stdout.write(message)
                
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

    train_dataset = RecursiveImageDataset(args.train_root, transform=build_train_transform(args.image_size))
    val_dataset = RecursiveImageDataset(args.val_root, transform=build_val_transform(args.image_size))
    train_loader = build_dataloader(train_dataset, args.batch_size, args.num_workers, shuffle=True, drop_last=True)
    val_loader = build_dataloader(val_dataset, args.batch_size, args.num_workers, shuffle=False, drop_last=False)

    tcm = build_tcm(args)
    if args.tcm_ckpt:
        ckpt = torch.load(args.tcm_ckpt, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        tcm.load_state_dict(state, strict=False)

    pipeline = FlowTCMStage1Pipeline(
        model_name=args.model_name,
        flux_ckpt=args.flux_ckpt,
        ae_ckpt=args.ae_ckpt,
        qwen_ckpt=args.qwen_ckpt,
        tcm=tcm,
        device=accelerator.device,
        guidance=args.guidance,
    )

    lora_stats = inject_lora(
        pipeline.flux,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_regex=args.lora_target_regex,
    )

    criterion = Stage1Loss(
        clip_path=args.clip_ckpt,
        lambda_rate=args.lambda_rate,
        d1_mse=args.d1_mse,
        d2_lpips=args.d2_lpips,
        d3_clip=args.d3_clip,
    )

    # Initialize evaluator with the correct run_dir
    evaluator = Stage1Evaluator(
        accelerator=accelerator,
        output_dir=run_dir if run_dir else args.output_dir,
        eval_batches=args.eval_batches,
    )

    trainable_params = [p for p in pipeline.tcm.parameters() if p.requires_grad]
    trainable_params += [p for p in pipeline.flux.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    pipeline.tcm, pipeline.flux, criterion, optimizer, train_loader, val_loader = accelerator.prepare(
        pipeline.tcm,
        pipeline.flux,
        criterion,
        optimizer,
        train_loader,
        val_loader,
    )

    pipeline.flux.train()
    pipeline.tcm.train()

    global_step = 0
    start_epoch = 0
    
    # Resume logic
    if args.resume:
        # Find the latest checkpoint in the output_dir (searching all run_* folders)
        all_runs = sorted(glob.glob(os.path.join(args.output_dir, "run_*")))
        if all_runs:
            latest_run = all_runs[-1]
            latest_ckpt_dir = os.path.join(latest_run, "checkpoints")
            ckpts = sorted(glob.glob(os.path.join(latest_ckpt_dir, "stage1_step_*.pt")))
            if ckpts:
                latest_ckpt = ckpts[-1]
                if accelerator.is_main_process:
                    print(f"Resuming from checkpoint: {latest_ckpt}")
                state = torch.load(latest_ckpt, map_location="cpu")
                
                accelerator.unwrap_model(pipeline.tcm).load_state_dict(state["tcm"], strict=True)
                missing = load_lora_state_dict(accelerator.unwrap_model(pipeline.flux), state["flux_lora"])
                if accelerator.is_main_process and missing:
                    print(f"[resume] missing LoRA modules: {len(missing)}")
                    
                optimizer.load_state_dict(state["optimizer"])
                global_step = int(state.get("global_step", 0))
                start_epoch = int(state.get("epoch", 0))
            else:
                if accelerator.is_main_process:
                    print("No checkpoint found in the latest run. Starting from scratch.")
        else:
            if accelerator.is_main_process:
                print("No previous runs found. Starting from scratch.")
    else:
        if accelerator.is_main_process:
            print("Starting training from scratch.")

    if accelerator.is_main_process:
        total_tcm = sum(p.numel() for p in accelerator.unwrap_model(pipeline.tcm).parameters() if p.requires_grad)
        print(f"Train images: {len(train_dataset)}")
        print(f"Val images: {len(val_dataset)}")
        print(f"LoRA injected layers: {lora_stats.injected_layers}")
        print(f"LoRA trainable params: {lora_stats.trainable_params}")
        print(f"TCM trainable params: {total_tcm}")

    meters = {k: AverageMeter() for k in ["loss", "bpp", "mse", "lpips", "clip_l2"]}

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
                    params += [p for p in pipeline.tcm.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()

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
                        f"mse={log_vals['mse']:.6f} lpips={log_vals['lpips']:.5f} clip_l2={log_vals['clip_l2']:.5f}"
                    )
                    # 写入 tensorboard
                    if args.use_tensorboard:
                        for k, v in log_vals.items():
                            writer.add_scalar(f"train/{k}", v, global_step)

            if global_step % args.eval_every == 0:
                # 多步评估（真实推理效果 + 熵编码）- 每 500 步执行一次
                metrics_multistep = evaluator.evaluate(
                    pipeline=pipeline,
                    criterion=criterion,
                    val_loader=val_loader,
                    global_step=global_step,
                    clip_ckpt=args.clip_ckpt,
                )
                if accelerator.is_main_process:
                    print(
                        f"[eval multistep {global_step}] "
                        f"loss={metrics_multistep['loss']:.5f} bpp={metrics_multistep['bpp']:.5f} "
                        f"mse={metrics_multistep['mse']:.6f} lpips={metrics_multistep['lpips']:.5f} clip_l2={metrics_multistep['clip_l2']:.5f}"
                    )
                    # 写入 tensorboard
                    if args.use_tensorboard:
                        for k, v in metrics_multistep.items():
                            writer.add_scalar(f"val_multistep/{k}", v, global_step)

            if global_step % args.save_every == 0 and accelerator.is_main_process:
                state = {
                    "global_step": global_step,
                    "pipeline": pipeline.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{global_step:08d}.pt")
                torch.save(state, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

            if global_step >= args.max_steps:
                stop = True
                break
        
        current_epoch += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_ckpt_path = os.path.join(ckpt_dir, "stage1_last.pt") if ckpt_dir else os.path.join(args.output_dir, "stage1_last.pt")
        state = {
            "global_step": global_step,
            "epoch": current_epoch,
            "tcm": accelerator.unwrap_model(pipeline.tcm).state_dict(),
            "flux_lora": lora_state_dict(accelerator.unwrap_model(pipeline.flux)),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        save_checkpoint(final_ckpt_path, state)
        print("Training completed.")
        
        # 关闭 tensorboard writer
        if args.use_tensorboard and writer:
            writer.close()
            print(f"TensorBoard logs saved to: {log_dir}")
        
        # 恢复原始 stdout
        if args.save_log_file and log_file_handle:
            sys.stdout = original_write
            sys.stderr = original_write
            log_file_handle.close()
            print(f"Training log saved to: {os.path.join(log_dir, 'train.log') if log_dir else 'N/A'}")


if __name__ == "__main__":
    main()
