import torch
import os
import time
from diffusers import Flux2KleinPipeline

# 锁定 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

device = "cuda"
dtype = torch.bfloat16

def print_gpu_utilization():
    # 获取当前已分配显存
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    # 获取自程序启动或重置以来的峰值显存
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"   [GPU Memory] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# 加载模型
print("正在加载模型...")
pipe = Flux2KleinPipeline.from_pretrained(
    "/data2/luosheng/hf_models/hub/FLUX.2-klein-4B", 
    torch_dtype=dtype,
    low_cpu_mem_usage=True
)
# 改进：直接移动到 GPU，避免 CPU offload 的性能损失
pipe.to(device)
# 如果显存不足，可以启用 xformers 或 sequential offload
# pipe.enable_xformers_memory_efficient_attention()  # 需要安装 xformers
# pipe.enable_sequential_cpu_offload()  # 更慢但更省显存
print_gpu_utilization()

# --- 配置区 ---
prompt_text = "A Chinese loong"
num_images = 3  
# --------------

print(f"\n开始推理测试...")

latencies = []
# 改进：复用 Generator，提高可复现性
generator = torch.Generator(device=device)

for i in range(num_images):
    # 重置峰值显存计数，以便测量每张图的独立峰值
    torch.cuda.reset_peak_memory_stats()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 设置种子
    generator.manual_seed(i)
    
    start_event.record()
    
    output = pipe(
        prompt=prompt_text,
        image=None,
        height=1024,
        width=1024,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=generator  # 使用复用的 generator
    )
    
    end_event.record()
    torch.cuda.synchronize()
    
    curr_time = start_event.elapsed_time(end_event) / 1000
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
    
    if i == 0:
        print(f"第 {i+1} 张 (预热): {curr_time:.3f}s | 峰值显存: {peak_mem:.2f} GB")
    else:
        print(f"第 {i+1} 张: {curr_time:.3f}s | 峰值显存: {peak_mem:.2f} GB")
        latencies.append(curr_time)
        print_gpu_utilization()
    
    output.images[0].save(f"flux-klein-{i}.png")

if latencies:
    avg_latency = sum(latencies)/len(latencies)
    print(f"\n平均耗时 (除预热外): {avg_latency:.3f}s")
    print(f"吞吐量: {1.0/avg_latency:.2f} images/sec")