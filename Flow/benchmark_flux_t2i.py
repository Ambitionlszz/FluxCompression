"""
FLUX.2 纯生图推理速度测试
测试不同分辨率下的文本生成图像速度
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    get_schedule,
    scatter_ids,
)
from flux2.util import load_ae, load_flow_model, load_text_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2 text-to-image speed benchmark")

    # 模型路径
    parser.add_argument("--model_name", type=str, default="flux.2-klein-4b")
    parser.add_argument("--flux_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors")
    parser.add_argument("--ae_ckpt", type=str, default="/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors")
    parser.add_argument("--qwen_ckpt", type=str, default="/data2/luosheng/hf_models/hub/Qwen3-4B-FP8")
    
    # 测试配置
    parser.add_argument("--resolutions", type=str, default="512,1024", help="Comma-separated resolutions")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean, photorealistic, detailed")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--warmup_runs", type=int, default=5)
    parser.add_argument("--test_runs", type=int, default=10)
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="./outputs/speed_test_t2i")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_images", action="store_true", help="Save generated images")

    return parser.parse_args()


def load_models(args, device):
    """加载 FLUX.2 模型"""
    print("Loading models...")
    
    # 设置环境变量，使用本地文件（避免联网下载）
    os.environ["KLEIN_4B_MODEL_PATH"] = args.flux_ckpt
    os.environ["AE_MODEL_PATH"] = args.ae_ckpt
    
    # 加载模型
    flux = load_flow_model(args.model_name, device=device)
    ae = load_ae(args.model_name, device=device)
    
    # 手动加载 Qwen3 文本编码器（指定本地路径）
    from flux2.text_encoder import Qwen3Embedder
    text_encoder = Qwen3Embedder(model_spec=args.qwen_ckpt, device=device)
    
    flux.eval()
    ae.eval()
    text_encoder.eval()
    
    print("✓ Models loaded successfully\n")
    return flux, ae, text_encoder


@torch.no_grad()
def generate_image(flux, ae, text_encoder, prompt, height, width, num_steps, guidance, device):
    """生成单张图像"""
    # 文本编码 - text_encoder 返回 shape: [B, L, D]
    if hasattr(text_encoder, 'guidance_distilled') and text_encoder.guidance_distilled:
        # 蒸馏模型只需要一次编码
        ctx = text_encoder([prompt])  # [1, L, D]
    else:
        # 非蒸馏模型需要空文本和提示词
        ctx_empty = text_encoder([""])  # [1, L, D]
        ctx_prompt = text_encoder([prompt])  # [1, L, D]
        ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)  # [2, L, D]
    
    # 处理文本上下文
    txt, txt_ids = batched_prc_txt(ctx)
    txt = txt.to(device)
    txt_ids = txt_ids.to(device)
    
    print(f"DEBUG: txt shape = {txt.shape}, txt_ids shape = {txt_ids.shape}")
    
    # 创建噪声输入 (NCHW 格式): [B, C, H, W]
    img = torch.randn(1, 16, height // 8, width // 8, device=device, dtype=torch.bfloat16)
    print(f"DEBUG: noise img shape = {img.shape}")
    
    # 转换为 tokens
    img_tokens, img_ids = batched_prc_img(img)
    print(f"DEBUG: img_tokens shape = {img_tokens.shape}, img_ids shape = {img_ids.shape}")
    print(f"DEBUG: img_tokens dtype = {img_tokens.dtype}, device = {img_tokens.device}")
    
    # 计算 image sequence length 并获取时间步调度
    image_seq_len = img_tokens.shape[1]  # latent tokens 的数量 = H*W
    timesteps = get_schedule(num_steps, image_seq_len)
    
    # 去噪生成
    print(f"DEBUG: Calling denoise with img_tokens.shape={img_tokens.shape}, img_ids.shape={img_ids.shape}")
    img_out = denoise(
        model=flux,
        img=img_tokens,  # Tensor: [B, seq_len, C]
        img_ids=img_ids,  # Tensor: [B, seq_len, 4]
        txt=txt,
        txt_ids=txt_ids,
        timesteps=timesteps,
        guidance=guidance,
    )
    print(f"DEBUG: img_out shape = {img_out.shape}")
    
    # 将输出转回 latent
    # 需要将 batch 维度拆开成 list，再用 scatter_ids 重组
    img_out_list = [img_out[i] for i in range(img_out.shape[0])]
    img_ids_list = [img_ids[i] for i in range(img_ids.shape[0])]
    img_latent = torch.cat(scatter_ids(img_out_list, img_ids_list)).squeeze(2)
    
    # 解码到图像空间
    img = ae.decode(img_latent)
    img = (img + 1.0) * 0.5  # [-1, 1] -> [0, 1]
    img = img.clamp(0.0, 1.0)
    
    return img


def run_benchmark(flux, ae, text_encoder, args, resolution, device):
    """在指定分辨率下运行基准测试"""
    prompt = args.prompt
    num_steps = args.num_inference_steps
    guidance = args.guidance
    
    times = []
    
    # 预热
    print(f"  Warming up ({args.warmup_runs} runs)...")
    for i in range(args.warmup_runs):
        _ = generate_image(flux, ae, text_encoder, prompt, resolution, resolution, num_steps, guidance, device)
        if i % 2 == 0 or i == args.warmup_runs - 1:
            print(f"    Warmup {i+1}/{args.warmup_runs} done")
    
    torch.cuda.synchronize(device)
    
    # 正式测试
    print(f"  Running benchmark ({args.test_runs} runs)...")
    for i in range(args.test_runs):
        start_time = time.time()
        
        img = generate_image(flux, ae, text_encoder, prompt, resolution, resolution, num_steps, guidance, device)
        
        torch.cuda.synchronize(device)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        if i % 2 == 0 or i == args.test_runs - 1:
            print(f"    Run {i+1}/{args.test_runs}: {elapsed:.4f}s")
        
        # 保存图像
        if args.save_images and i == 0:
            save_path = os.path.join(args.output_dir, f"gen_{resolution}x{resolution}.png")
            save_image(img, save_path)
    
    # 统计
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = torch.std(torch.tensor(times)).item()
    
    fps = args.test_runs / sum(times)
    
    stats = {
        "resolution": resolution,
        "num_steps": num_steps,
        "guidance": guidance,
        "avg_time_sec": avg_time,
        "std_time_sec": std_time,
        "min_time_sec": min_time,
        "max_time_sec": max_time,
        "fps": fps,
    }
    
    return stats


def print_results(all_stats):
    """打印结果"""
    print("\n" + "="*80)
    print("FLUX.2 TEXT-TO-IMAGE SPEED BENCHMARK RESULTS")
    print("="*80)
    
    for stats in all_stats:
        res = stats["resolution"]
        print(f"\nResolution: {res}x{res}")
        print(f"  Inference Steps: {stats['num_steps']}")
        print(f"  Guidance: {stats['guidance']}")
        print(f"  Average Time: {stats['avg_time_sec']:.4f} ± {stats['std_time_sec']:.4f} seconds")
        print(f"  Min Time: {stats['min_time_sec']:.4f} seconds")
        print(f"  Max Time: {stats['max_time_sec']:.4f} seconds")
        print(f"  Throughput: {stats['fps']:.2f} FPS")
    
    print("\n" + "="*80)
    
    if len(all_stats) > 1:
        print("\nRESOLUTION COMPARISON")
        print("-"*80)
        print(f"{'Resolution':<15} {'Avg Time(s)':<15} {'FPS':<10}")
        print("-"*80)
        for stats in all_stats:
            print(f"{stats['resolution']:<15} {stats['avg_time_sec']:<15.4f} {stats['fps']:<10.2f}")
        print("-"*80)
    
    print("="*80 + "\n")


def main():
    args = parse_args()
    
    # 解析分辨率
    resolutions = [int(r.strip()) for r in args.resolutions.split(",")]
    resolutions.sort()
    
    print("="*80)
    print("FLUX.2 Text-to-Image Speed Benchmark")
    print("="*80)
    print(f"Prompt: {args.prompt}")
    print(f"Resolutions: {resolutions}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Test runs: {args.test_runs}")
    print("="*80 + "\n")
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 加载模型
    flux, ae, text_encoder = load_models(args, device)
    
    # 创建输出目录
    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 测试每个分辨率
    all_stats = []
    for res in resolutions:
        print(f"\nTesting {res}x{res}...")
        print("-"*80)
        
        stats = run_benchmark(flux, ae, text_encoder, args, res, device)
        all_stats.append(stats)
        
        print(f"✓ Completed {res}x{res}")
    
    # 打印结果
    print_results(all_stats)
    
    # 保存结果
    import json
    output_file = os.path.join(args.output_dir, "speed_benchmark_t2i.json")
    with open(output_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
