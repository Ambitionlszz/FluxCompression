import torch
import os
import time
from diffusers import Flux2KleinPipeline

# 锁定 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

# 设置设备（当 CUDA_VISIBLE_DEVICES="1" 时，在代码中应该用 cuda:0 访问）
device = "cuda"  # 简化为 "cuda"，让 PyTorch 自动使用 CUDA_VISIBLE_DEVICES 指定的 GPU
dtype = torch.bfloat16

# 检查 GPU 状态
print(f"\n当前设备：{device}")
print(f"GPU 数量：{torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前 GPU 索引：{torch.cuda.current_device()}")
    print(f"GPU 名称：{torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU 总显存：{total_mem:.2f} GB")
print()

def print_gpu_utilization():
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"   [GPU Memory] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# 加载模型
print("正在加载模型...")
pipe = Flux2KleinPipeline.from_pretrained(
    "/data2/luosheng/hf_models/hub/FLUX.2-klein-4B", 
    torch_dtype=dtype,
    low_cpu_mem_usage=True
)
# 将模型移动到 GPU
pipe = pipe.to(device)
print("模型加载完成")

# 测量模型加载后的基础显存
torch.cuda.synchronize()
base_allocated = torch.cuda.memory_allocated(device) / 1024**3
base_reserved = torch.cuda.memory_reserved(device) / 1024**3
print(f"\n【基础显存】Allocated: {base_allocated:.2f} GB | Reserved: {base_reserved:.2f} GB")
print("=" * 80)

# --- 配置区 ---
prompt_text = "A Chinese loong"
warmup_runs = 2  
test_runs = 5
output_dir = "output_images"  # 输出图像保存目录
os.makedirs(output_dir, exist_ok=True)
# --------------

configs = [
    {"resolution": 2048, "steps": 4},
    {"resolution": 2048, "steps": 1},
    {"resolution": 1024, "steps": 4},
    {"resolution": 1024, "steps": 1},
]

results = {}

print(f"\n开始推理测试...")
print("=" * 80)

for config in configs:
    resolution = config["resolution"]
    steps = config["steps"]
    key = f"{resolution}x{resolution}_{steps}step"
    
    print(f"\n测试配置：{resolution}x{resolution} 分辨率，{steps} step(s)")
    print("-" * 80)
    
    latencies = []
    peak_mems = []
    incremental_mems = []  # 新增：推理增量显存
    generated_images = []
    
    # 预热阶段
    if warmup_runs > 0:
        print(f"正在进行 {warmup_runs} 次预热...")
        for i in range(warmup_runs):
            torch.cuda.reset_peak_memory_stats()
            
            # 同步 CUDA，确保前面所有操作完成
            torch.cuda.synchronize()
            
            start_time = time.time()
            
            output = pipe(
                prompt=prompt_text,
                image=None,
                height=resolution,
                width=resolution,
                guidance_scale=1.0,
                num_inference_steps=steps,
                generator=torch.Generator(device=device).manual_seed(i)
            )
            
            # 等待 GPU 完成
            torch.cuda.synchronize()
            curr_time = time.time() - start_time
            
            if i == 0:
                # 保存第一张预热图像用于验证
                warmup_img_path = os.path.join(output_dir, f"flux-klein-{key}-warmup.png")
                output.images[0].save(warmup_img_path)
                print(f"  预热图像已保存到：{warmup_img_path}")
            print(f"  预热 {i+1}: {curr_time:.3f}s")
    
    # 正式测试
    print(f"正在进行 {test_runs} 次正式测试...")
    for i in range(test_runs):
        # 重置峰值统计并记录推理前的显存
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(device)
        
        # 开始计时（覆盖整个推理过程）
        start_time = time.time()
        
        output = pipe(
            prompt=prompt_text,
            image=None,
            height=resolution,
            width=resolution,
            guidance_scale=1.0,
            num_inference_steps=steps,
            generator=torch.Generator(device=device).manual_seed(i + warmup_runs)
        )
        
        # 等待 GPU 完全完成后停止计时
        torch.cuda.synchronize()
        curr_time = time.time() - start_time
        
        # 获取峰值和推理后显存
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        mem_after = torch.cuda.memory_allocated(device)
        
        # 计算增量显存（推理过程额外占用的显存）
        incremental_mem = (peak_mem - base_allocated)
        
        latencies.append(curr_time)
        peak_mems.append(peak_mem)
        incremental_mems.append(incremental_mem)
        generated_images.append(output.images[0])
        
        # 保存每张生成的图像
        img_path = os.path.join(output_dir, f"flux-klein-{key}_run{i+1}.png")
        output.images[0].save(img_path)
        print(f"  测试 {i+1}: {curr_time:.3f}s | 峰值显存：{peak_mem:.2f} GB | 增量显存：{incremental_mem:.2f} GB")
    
    print_gpu_utilization()
    
    # 统计结果
    avg_latency = sum(latencies) / len(latencies)
    std_latency = torch.std(torch.tensor(latencies)).item()
    avg_peak_mem = sum(peak_mems) / len(peak_mems)
    avg_incremental_mem = sum(incremental_mems) / len(incremental_mems)
    
    results[key] = {
        "avg_latency": avg_latency,
        "std_latency": std_latency,
        "avg_peak_mem": avg_peak_mem,
        "avg_incremental_mem": avg_incremental_mem,
        "min_latency": min(latencies),
        "max_latency": max(latencies),
    }
    
    print(f"\n【{resolution}x{resolution} {steps}step】结果汇总:")
    print(f"  平均耗时：{avg_latency:.3f}s ± {std_latency:.3f}s")
    print(f"  最小耗时：{min(latencies):.3f}s")
    print(f"  最大耗时：{max(latencies):.3f}s")
    print(f"  平均峰值显存：{avg_peak_mem:.2f} GB")
    print(f"  平均增量显存：{avg_incremental_mem:.2f} GB (相对于基础 {base_allocated:.2f} GB)")

# 打印总结果汇总
print("\n" + "=" * 80)
print("所有测试结果汇总")
print("=" * 80)
print(f"{'配置':<25} {'平均耗时 (s)':<15} {'标准差':<10} {'峰值显存 (GB)':<15} {'增量显存 (GB)':<15}")
print("-" * 80)
for key, stats in results.items():
    print(f"{key:<25} {stats['avg_latency']:<15.3f} {stats['std_latency']:<10.3f} {stats['avg_peak_mem']:<15.2f} {stats['avg_incremental_mem']:<15.2f}")
print("=" * 80)

# 额外保存一个汇总信息文件
summary_path = os.path.join(output_dir, "benchmark_summary.txt")
with open(summary_path, "w") as f:
    f.write("FLUX2-Klein 推理性能测试结果汇总\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"提示词：{prompt_text}\n")
    f.write(f"预热次数：{warmup_runs}\n")
    f.write(f"测试次数：{test_runs}\n")
    f.write(f"基础显存：{base_allocated:.2f} GB\n\n")
    
    for key, stats in results.items():
        f.write(f"[{key}]\n")
        f.write(f"  平均耗时：{stats['avg_latency']:.6f}s\n")
        f.write(f"  标准差：{stats['std_latency']:.6f}s\n")
        f.write(f"  最小耗时：{stats['min_latency']:.6f}s\n")
        f.write(f"  最大耗时：{stats['max_latency']:.6f}s\n")
        f.write(f"  平均峰值显存：{stats['avg_peak_mem']:.2f} GB\n")
        f.write(f"  平均增量显存：{stats['avg_incremental_mem']:.2f} GB\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("说明：\n")
    f.write("- 峰值显存：推理过程中的 GPU 显存总占用\n")
    f.write("- 增量显存：相对于模型加载后基础显存的额外增加量\n")
    f.write("- 禁用 CPU offload 以准确测量不同分辨率的显存差异\n")

print(f"\n汇总报告已保存到：{summary_path}")
print(f"所有输出图像已保存到：{output_dir}/ 目录下")
