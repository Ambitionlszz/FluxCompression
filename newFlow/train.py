import os
import sys
import argparse
import glob
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# 确保项目根目录在路径中
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from accelerate import Accelerator
from accelerate.utils import set_seed

from newFlow.model import FlowCompression
from newFlow.modules import (
    RecursiveImageDataset,
    build_train_transform,
    build_val_transform,
    build_dataloader,
    Stage1Loss,
    Stage1Evaluator,
    inject_lora,
    lora_state_dict,
    load_lora_state_dict,
    AverageMeter,
    ensure_dir,
)


def print_model_parameters(net, model_name="FlowCompression", accelerator=None):
    """
    打印模型各模块的参数量统计信息
    
    Args:
        net: 模型实例
        model_name: 模型名称（用于显示）
        accelerator: Accelerator 实例（用于多卡同步打印）
    """
    # 只在主进程打印
    is_main = accelerator.is_main_process if accelerator else True
    
    if not is_main:
        return
    
    print(f"\n{'='*80}")
    print(f"Model Parameters Statistics - {model_name}")
    print(f"{'='*80}\n")
    
    # 总参数量
    total_params = sum(p.numel() for p in net.parameters())
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_params_m = total_params / 1e6
    total_trainable_params_m = total_trainable_params / 1e6
    
    print(f"Total Parameters:     {total_params_m:.2f}M ({total_params:,})")
    print(f"Trainable Parameters: {total_trainable_params_m:.2f}M ({total_trainable_params:,})")
    print(f"{'-'*80}\n")
    
    # 按主要模块分组统计
    module_stats = {}
    
    # 遍历顶层模块
    for name, module in net.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_stats[name] = {
            'params': params,
            'trainable_params': trainable_params,
            'module': module
        }
    
    # 按参数量排序
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['params'], reverse=True)
    
    print(f"Module-wise Breakdown:")
    print(f"{'Module Name':<40} {'Params (M)':>12} {'Trainable (M)':>15} {'Percentage':>12}")
    print(f"{'-'*80}")
    
    for name, stats in sorted_modules:
        params_m = stats['params'] / 1e6
        trainable_m = stats['trainable_params'] / 1e6
        percentage = (stats['params'] / total_params * 100) if total_params > 0 else 0
        
        print(f"{name:<40} {params_m:>12.2f} {trainable_m:>15.2f} {percentage:>11.2f}%")
    
    print(f"{'-'*80}")
    print(f"{'Total':<40} {total_params_m:>12.2f} {total_trainable_params_m:>15.2f} {'100.00%':>12}")
    print(f"{'='*80}\n")
    
    # 如果有codec属性，进一步详细统计codec内部模块
    if hasattr(net, 'codec') and net.codec is not None:
        print(f"Codec Internal Modules Details:")
        print(f"{'='*80}\n")
        
        codec = net.codec
        codec_module_stats = {}
        
        for name, module in codec.named_children():
            params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            codec_module_stats[name] = {
                'params': params,
                'trainable_params': trainable_params
            }
        
        # 按参数量排序
        sorted_codec_modules = sorted(codec_module_stats.items(), key=lambda x: x[1]['params'], reverse=True)
        
        print(f"{'Module Name':<40} {'Params (M)':>12} {'Trainable (M)':>15} {'Percentage':>12}")
        print(f"{'-'*80}")
        
        for name, stats in sorted_codec_modules:
            params_m = stats['params'] / 1e6
            trainable_m = stats['trainable_params'] / 1e6
            percentage = (stats['params'] / total_params * 100) if total_params > 0 else 0
            
            print(f"{name:<40} {params_m:>12.2f} {trainable_m:>15.2f} {percentage:>11.2f}%")
        
        print(f"{'='*80}\n")
    
    # 如果有flux模块，单独统计
    if hasattr(net, 'flux') and net.flux is not None:
        print(f"Flux/DiT Module Details:")
        print(f"{'='*80}\n")
        
        flux = net.flux
        flux_params = sum(p.numel() for p in flux.parameters())
        flux_trainable_params = sum(p.numel() for p in flux.parameters() if p.requires_grad)
        flux_params_m = flux_params / 1e6
        flux_trainable_params_m = flux_trainable_params / 1e6
        percentage = (flux_params / total_params * 100) if total_params > 0 else 0
        
        print(f"Flux Total Params:          {flux_params_m:.2f}M ({flux_params:,})")
        print(f"Flux Trainable Params:      {flux_trainable_params_m:.2f}M ({flux_trainable_params:,})")
        print(f"Percentage of Total:        {percentage:.2f}%")
        print(f"{'='*80}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FlowCompression (DiT-IC Style)")
    
    # 配置文件
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    
    # 命令行参数（会覆盖配置文件中的值）
    parser.add_argument("--train_root", type=str, default=None)
    parser.add_argument("--val_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--flux_ckpt", type=str, default=None)
    parser.add_argument("--ae_ckpt", type=str, default=None)
    parser.add_argument("--elic_ckpt", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_path", type=str, default=None)
    
    return parser.parse_args()


def load_config(args):
    """
    加载 YAML 配置文件（唯一配置源），并用命令行参数覆盖
    
    设计理念：
    - YAML 文件是唯一的配置来源，避免配置冗余和不同步问题
    - 如果未提供 --config 参数，自动使用默认的 train_config.yaml
    - 命令行参数具有最高优先级，可覆盖 YAML 中的值
    """
    # 确定配置文件路径
    config_path = args.config
    if not config_path:
        # 默认使用脚本同目录下的 config/train_config.yaml
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config', 'train_config.yaml')
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please provide a valid config file via --config argument."
        )
    
    # 加载 YAML 配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {config_path}")
    
    # 命令行参数覆盖（优先级最高）
    if args.train_root:
        config['paths']['train_root'] = args.train_root
    if args.val_root:
        config['paths']['val_root'] = args.val_root
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.flux_ckpt:
        config['paths']['flux_ckpt'] = args.flux_ckpt
    if args.ae_ckpt:
        config['paths']['ae_ckpt'] = args.ae_ckpt
    if args.elic_ckpt:
        config['paths']['elic_ckpt'] = args.elic_ckpt
    if args.resume:
        config['resume']['enabled'] = True
    if args.resume_path:
        config['resume']['checkpoint_path'] = args.resume_path
    
    return config


def setup_directories(config, accelerator):
    """设置输出目录和日志目录"""
    run_dir = None
    ckpt_dir = None
    log_dir = None
    writer = None
    original_stdout = None
    log_file_handle = None
    
    if accelerator.is_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config['paths']['output_dir'], f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        log_dir = os.path.join(run_dir, "logs")
        eval_image_dir = os.path.join(run_dir, "eval_images")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(eval_image_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(run_dir, "train_config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        # 设置日志文件
        if config['logging']['save_log_file']:
            log_file = os.path.join(log_dir, "train.log")
            original_stdout = sys.stdout
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
        
        # TensorBoard
        if config['logging']['use_tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            # 确保 log_dir 存在（SummaryWriter 有时不会自动创建）
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs will be saved to: {log_dir}")
    
    return run_dir, ckpt_dir, log_dir, writer, original_stdout, log_file_handle


def find_latest_checkpoint(output_dir):
    """查找最新的 checkpoint"""
    all_runs = sorted(glob.glob(os.path.join(output_dir, "run_*")))
    if not all_runs:
        return None
    
    latest_run = all_runs[-1]
    latest_ckpt_dir = os.path.join(latest_run, "checkpoints")
    ckpts = sorted(glob.glob(os.path.join(latest_ckpt_dir, "checkpoint_step_*.pt")))
    
    if ckpts:
        return ckpts[-1], latest_run
    return None, None


def main():
    args = parse_args()
    config = load_config(args)
    
    # 初始化 Accelerate
    accelerator = Accelerator(
        mixed_precision=config['gpu']['mixed_precision'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    )
    
    # 设置随机种子
    set_seed(config['training']['seed'])
    
    # 设置目录
    run_dir, ckpt_dir, log_dir, writer, original_stdout, log_file_handle = setup_directories(config, accelerator)
    
    device = accelerator.device
    
    # ==================== 1. 数据加载 ====================
    if accelerator.is_main_process:
        print("Loading datasets...")
    
    train_transform = build_train_transform(config['data']['image_size'])
    val_transform = build_val_transform(config['data']['image_size'])
    
    train_dataset = RecursiveImageDataset(root=config['paths']['train_root'], transform=train_transform)
    val_dataset = RecursiveImageDataset(root=config['paths']['val_root'], transform=val_transform)
    
    train_loader = build_dataloader(
        train_dataset, 
        batch_size=config['data']['batch_size'], 
        num_workers=config['data']['num_workers'], 
        shuffle=True, 
        drop_last=True
    )
    val_loader = build_dataloader(
        val_dataset, 
        batch_size=config['data']['batch_size'], 
        num_workers=config['data']['num_workers'], 
        shuffle=False, 
        drop_last=False
    )
    
    if accelerator.is_main_process:
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # ==================== 2. 模型初始化 ====================
    if accelerator.is_main_process:
        print("Initializing FlowCompression model...")
    
    codec_config = {
        "ch_emd": config['codec']['ch_emd'],
        "channel": config['codec']['channel'],
        "channel_out": config['codec']['channel_out'],
        "num_slices": config['codec']['num_slices'],
    }
    
    model = FlowCompression(
        model_name=config['paths']['model_name'],
        flux_ckpt=config['paths']['flux_ckpt'],
        ae_ckpt=config['paths']['ae_ckpt'],
        qwen_ckpt=config['paths']['qwen_ckpt'],  # Tokenizer 路径（use_text_condition=False时不会加载）
        codec_config=codec_config,
        device=device,
        use_text_condition=config.get('model', {}).get('use_text_condition', False),  # ← 从配置文件读取
        qwen_model_path=config['paths'].get('qwen_model_path'),  # Qwen 模型权重路径
        elic_ckpt=config['paths']['elic_ckpt'],  # ELIC 辅助编码器权重路径
    )
    
    # 启用梯度检查点（如果配置）
    if config['training'].get('gradient_checkpointing', False):
        if accelerator.is_main_process:
            print("\nEnabling Gradient Checkpointing...")
        model.enable_gradient_checkpointing(enabled=True)
    
    if accelerator.is_main_process:
        print(f"ELIC checkpoint path passed to FlowCompression model.")
    
    # ==================== 3. LoRA 注入 ====================
    if config['lora']['enabled']:
        if accelerator.is_main_process:
            print(f"Injecting LoRA (rank={config['lora']['rank']}, alpha={config['lora']['alpha']})...")
        
        lora_stats = inject_lora(
            model.flux,
            rank=config['lora']['rank'],
            alpha=config['lora']['alpha'],
            dropout=config['lora']['dropout'],
            target_regex=config['lora']['target_regex'],
        )
        
        if accelerator.is_main_process:
            print(f"LoRA injected layers: {lora_stats.injected_layers}")
            print(f"LoRA trainable params: {lora_stats.trainable_params:,}")
    else:
        lora_stats = None
    
    # ==================== 3.5 打印模型参数统计 ====================
    if accelerator.is_main_process:
        print_model_parameters(model, model_name="FlowCompression", accelerator=accelerator)
    
    # ==================== 4. 损失函数 ====================
    loss_fn = Stage1Loss(
        clip_path=config['paths']['clip_ckpt'],
        lambda_rate=config['loss']['lambda_rate'],
        d1_mse=config['loss']['d1_mse'],
        d2_lpips=config['loss']['d2_lpips'],
        d3_clip=config['loss']['d3_clip'],
    ).to(device)
    
    # ==================== 5. 评估器 ====================
    evaluator = Stage1Evaluator(
        output_dir=run_dir if run_dir else config['paths']['output_dir'],
        eval_batches=config['logging']['eval_batches'],
        accelerator=accelerator,
    )
    
    # ==================== 6. 优化器 ====================
    # 只优化 Codec + LoRA 参数（对齐 Flow 项目的单一优化器策略）
    trainable_params = list(model.codec.parameters())
    if config['lora']['enabled']:
        lora_params = [p for n, p in model.flux.named_parameters() if "lora" in n.lower()]
        trainable_params += lora_params
    
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    if accelerator.is_main_process:
        total_codec = sum(p.numel() for p in model.codec.parameters() if p.requires_grad)
        print(f"Codec trainable params: {total_codec:,}")
        print(f"Total trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    # ==================== 7. Accelerate 准备 ====================
    model, loss_fn, optimizer, train_loader, val_loader = accelerator.prepare(
        model, loss_fn, optimizer, train_loader, val_loader
    )
    
    # 设置训练模式
    model.train()
    
    # ==================== 8. 恢复训练 ====================
    global_step = 0
    start_epoch = 0
    
    if config['resume']['enabled']:
        if config['resume']['checkpoint_path']:
            resume_ckpt = config['resume']['checkpoint_path']
        else:
            resume_ckpt, _ = find_latest_checkpoint(config['paths']['output_dir'])
        
        if resume_ckpt and os.path.exists(resume_ckpt):
            if accelerator.is_main_process:
                print(f"Resuming from checkpoint: {resume_ckpt}")
            
            state = torch.load(resume_ckpt, map_location="cpu")
            
            # 加载模型状态
            accelerator.unwrap_model(model).load_state_dict(state["model"], strict=False)
            
            # 加载 LoRA 状态
            if config['lora']['enabled'] and "flux_lora" in state:
                missing = load_lora_state_dict(accelerator.unwrap_model(model).flux, state["flux_lora"])
                if accelerator.is_main_process and missing:
                    print(f"[resume] missing LoRA modules: {len(missing)}")
            
            # 加载优化器状态
            optimizer.load_state_dict(state["optimizer"])
            global_step = int(state.get("global_step", 0))
            start_epoch = int(state.get("epoch", 0))
            
            if accelerator.is_main_process:
                print(f"Resumed at global_step={global_step}, epoch={start_epoch}")
        else:
            if accelerator.is_main_process:
                print("No checkpoint found. Starting from scratch.")
    else:
        if accelerator.is_main_process:
            print("Starting training from scratch.")
    
    # ==================== 9. 训练循环 ====================
    if accelerator.is_main_process:
        print("\nStarting training...")
        print(f"Max steps: {config['training']['max_steps']}")
        print(f"Mixed precision: {config['gpu']['mixed_precision']}")
        print(f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
        print(f"Gradient checkpointing: {'Enabled' if config['training'].get('gradient_checkpointing', False) else 'Disabled'}")
        print(f"Effective batch size: {config['data']['batch_size'] * config['training']['gradient_accumulation_steps'] * accelerator.num_processes}")
        print("=" * 80)
    
    meters = {k: AverageMeter() for k in ["loss", "bpp", "mse", "psnr", "lpips", "clip_l2"]}
    
    stop = False
    current_epoch = start_epoch
    
    # 获取 unwrapped model 用于访问内部方法
    unwrapped_model = accelerator.unwrap_model(model)
    
    print(f"[Training Started] Using device: {accelerator.device}, Process Rank: {accelerator.process_index}")
    
    while not stop:
        for batch_data in train_loader:
            x01 = batch_data
            
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    # 使用 ELIC 生成 z_aux（ELIC已集成到模型内部）
                    with torch.no_grad():
                        if unwrapped_model.elic_aux_encoder is not None:
                            z_aux = unwrapped_model.elic_aux_encoder(x01)  # (B, 320, H/16, W/16)
                        else:
                            # 如果没有ELIC编码器，使用dummy z_aux
                            z_main, _ = unwrapped_model.encode_images(x01)
                            z_aux = torch.zeros(
                                z_main.shape[0], 320, 
                                z_main.shape[2], z_main.shape[3],
                                device=z_main.device, dtype=z_main.dtype
                            )
                    
                    # ⭐ 关键诊断：检查z_aux和z_main的维度是否匹配
                    if global_step < 5 and accelerator.is_main_process:
                        z_main_debug, pad_info = unwrapped_model.encode_images(x01)
                        print(f"\n[Step {global_step}] Dimension Check:")
                        print(f"  Input x01 shape: {x01.shape}")
                        print(f"  z_main shape (after padding): {z_main_debug.shape}")
                        print(f"  z_aux shape: {z_aux.shape}")
                        print(f"  Padding info: pad_h={pad_info['pad_h']}, pad_w={pad_info['pad_w']}")
                        if z_main_debug.shape[-2:] != z_aux.shape[-2:]:
                            print(f"  ⚠️ WARNING: Dimension mismatch! z_main={z_main_debug.shape[-2:]}, z_aux={z_aux.shape[-2:]}")
                        else:
                            print(f"  ✓ Dimensions match\n")
                    
                    # 前向传播
                    output = unwrapped_model.forward(
                        x01, 
                        z_aux, 
                        train_schedule_steps=config['training']['train_schedule_steps'],
                        global_step=global_step  # ← 传递global_step用于调试日志控制
                    )
                    
                    # 计算损失
                    losses = loss_fn(x01, output["x_hat"], output["likelihoods"])
                    total_loss = losses["loss"]
                    
                    # ⚠️ 移除KL损失累加，对齐TCMLatent方案
                    # if "kl_loss" in output and output["kl_loss"] is not None:
                    #     total_loss = total_loss + 0.05 * output["kl_loss"]
                    
                    # ⭐ 诊断：监控likelihoods范围（仅前5步）
                    if global_step < 5 and accelerator.is_main_process:
                        z_lik = output["likelihoods"]["z"]
                        y_lik = output["likelihoods"]["y"]
                        print(f"\n[Step {global_step}] Likelihoods stats:")
                        print(f"  z: min={z_lik.min():.6f}, max={z_lik.max():.6f}, mean={z_lik.mean():.6f}")
                        print(f"  y: min={y_lik.min():.6f}, max={y_lik.max():.6f}, mean={y_lik.mean():.6f}")
                        print(f"  BPP: {losses['bpp'].item():.6f}, MSE: {losses['mse'].item():.6f}\n")
                
                # 反向传播
                optimizer.zero_grad()
                accelerator.backward(total_loss)
                
                # 梯度裁剪
                if config['training']['grad_clip'] > 0:
                    accelerator.clip_grad_norm_(trainable_params, config['training']['grad_clip'])
                
                optimizer.step()

            # 更新统计
            bs = x01.shape[0]
            for k in meters:
                meters[k].update(losses[k].item(), bs)
            
            global_step += 1
            
            # ==================== 日志记录 ====================
            if global_step % config['logging']['log_every'] == 0:
                log_vals = {}
                for k, meter in meters.items():
                    tensor = torch.tensor(
                        [meter.sum, meter.count], 
                        device=accelerator.device, 
                        dtype=torch.float64
                    )
                    tensor = accelerator.reduce(tensor, reduction="sum")
                    log_vals[k] = (tensor[0] / max(tensor[1], 1)).item()
                    meter.reset()
                
                if accelerator.is_main_process:
                    # 诊断信息：检查BPP是否异常低
                    bpp_warning = ""
                    if log_vals['bpp'] < 0.05:
                        bpp_warning = " ⚠️  WARNING: BPP异常低，熵模型可能训练失败！"
                    elif log_vals['bpp'] < 0.1:
                        bpp_warning = " ⚠️  CAUTION: BPP偏低，请监控熵模型状态"
                    
                    print(
                        f"[step {global_step}/{config['training']['max_steps']}] "
                        f"loss={log_vals['loss']:.5f} bpp={log_vals['bpp']:.5f} "
                        f"mse={log_vals['mse']:.6f} psnr={log_vals['psnr']:.2f}dB "
                        f"lpips={log_vals['lpips']:.5f} clip_l2={log_vals['clip_l2']:.5f}"
                    )
                    
                    # TensorBoard
                    if config['logging']['use_tensorboard'] and writer:
                        for k, v in log_vals.items():
                            writer.add_scalar(f"train/{k}", v, global_step)
                        writer.add_scalar("train/lr", config['training']['lr'], global_step)
            
            # ==================== 验证评估 ====================
            if global_step % config['logging']['eval_every'] == 0:
                if accelerator.is_main_process:
                    print(f"\nRunning evaluation at step {global_step}...")
                
                metrics = evaluator.evaluate(
                    pipeline=accelerator.unwrap_model(model),
                    criterion=loss_fn,
                    val_loader=val_loader,
                    global_step=global_step,
                    clip_ckpt=config['paths']['clip_ckpt'],
                )
                
                if accelerator.is_main_process:
                    print(
                        f"[eval {global_step}] "
                        f"loss={metrics['loss']:.5f} bpp={metrics['bpp']:.5f} "
                        f"mse={metrics['mse']:.6f} psnr={metrics['psnr']:.2f}dB "
                        f"lpips={metrics['lpips']:.5f} clip_l2={metrics['clip_l2']:.5f}\n"
                    )
                    
                    # TensorBoard
                    if config['logging']['use_tensorboard'] and writer:
                        for k, v in metrics.items():
                            writer.add_scalar(f"val/{k}", v, global_step)
            
            # ==================== 保存 Checkpoint ====================
            if global_step % config['logging']['save_every'] == 0 and accelerator.is_main_process:
                state = {
                    "global_step": global_step,
                    "epoch": current_epoch,
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                }
                
                # 如果使用了 LoRA，单独保存 LoRA 参数
                if config['lora']['enabled']:
                    state["flux_lora"] = lora_state_dict(accelerator.unwrap_model(model).flux)
                
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{global_step:08d}.pt")
                torch.save(state, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
            
            # ==================== 检查是否达到最大步数 ====================
            if global_step >= config['training']['max_steps']:
                stop = True
                break
        
        current_epoch += 1
    
    # ==================== 10. 最终保存 ====================
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        final_ckpt_path = os.path.join(ckpt_dir, "flow_compression_final.pt")
        state = {
            "global_step": global_step,
            "epoch": current_epoch,
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        }
        
        if config['lora']['enabled']:
            state["flux_lora"] = lora_state_dict(accelerator.unwrap_model(model).flux)
        
        torch.save(state, final_ckpt_path)
        print(f"\nTraining completed. Final checkpoint saved to {final_ckpt_path}")
        
        # 关闭 TensorBoard
        if config['logging']['use_tensorboard'] and writer:
            writer.close()
            print(f"TensorBoard logs saved to: {log_dir}")
        
        # 恢复原始 stdout
        if config['logging']['save_log_file'] and log_file_handle:
            sys.stdout = original_stdout
            log_file_handle.close()
            print(f"Training log saved to: {os.path.join(log_dir, 'train.log')}")


if __name__ == "__main__":
    main()
