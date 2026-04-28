# Flux2/Flow 推理代码使用指南

## 📋 目录
- [快速开始](#快速开始)
- [推理场景](#推理场景)
- [评估指标](#评估指标)
- [参数说明](#参数说明)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

## 🚀 快速开始

### 环境准备

```bash
# 激活虚拟环境
cd /data2/luosheng/code/flux2
source .venv/bin/activate

# 安装额外依赖（用于新指标）
pip install pytorch-msssim pyiqa
```

### 模型准备
确保以下模型已下载到本地：
- FLUX.2-klein-4B: `/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/`
- Qwen3-4B-FP8: `/data2/luosheng/hf_models/hub/Qwen3-4B-FP8/`
- CLIP: `/data2/luosheng/hf_models/hub/clip-vit-base-patch32/`

---

## 🎯 推理场景

### 场景 1: 文本生成图像 (Text-to-Image)

**适用**: 快速从文本提示生成高质量图像

**使用方法**:
```bash
cd /data2/luosheng/code/flux2

# 1. 编辑 inference.py 配置区
# 修改以下参数：
# - prompt_text: 你的文本提示
# - num_images: 生成数量
# - height/width: 图像尺寸（512, 768, 1024 等）

# 2. 运行推理
python inference.py
```

**输出**:
- `flux-klein-0.png`, `flux-klein-1.png`, ... (生成的图像)
- 终端显示每张图的推理时间和显存占用

**示例配置**:
```python
prompt_text = "A majestic Chinese dragon flying through clouds, digital art, highly detailed"
num_images = 5
height = 1024
width = 1024
guidance_scale = 1.0  # Klein 模型推荐值
num_inference_steps = 4  # 蒸馏模型只需 4 步
```

**性能参考** (RTX 4090):
- 512x512: ~0.5s/image
- 1024x1024: ~1.5s/image

---

### 场景 2: TCM 压缩重建 (核心功能) ⭐

#### ✨ 新功能亮点

1. **全面的评估指标**: PSNR、MS-SSIM、LPIPS、DISTS、BPP
2. **逐图指标输出**: 实时显示每张图的详细指标
3. **多数据集支持**: 一次推理可处理多个数据集
4. **智能输出命名**: 自动包含数据集名和平均 BPP

#### 完整流程:

**Step 1: 准备测试数据**
```bash
# 单数据集
export INPUT_DIRS=/data2/luosheng/data/Datasets/Kodak

# 多数据集（空格分隔）
export INPUT_DIRS="/data2/luosheng/data/Datasets/Kodak /data2/luosheng/data/Datasets/CLIC"
```

**Step 2: 运行推理**

**方式 A: 使用 Shell 脚本（推荐）** ⭐
```bash
cd /data2/luosheng/code/flux2/Flow

# 单数据集
export INPUT_DIRS=/data2/luosheng/data/Datasets/Kodak
export CHECKPOINT=./outputs/stage1/checkpoint_step_00100000.pt
export CUDA_DEVICES=0
bash infer_stage1.sh

# 多数据集
export INPUT_DIRS="/data2/luosheng/data/Datasets/Kodak /data2/luosheng/data/Datasets/CLIC"
bash infer_stage1.sh
```

**方式 B: 直接运行 Python**
```bash
cd /data2/luosheng/code/flux2/Flow

# 单数据集
python flux_tcm_stage1_infer.py \
  --input_dirs /data2/luosheng/data/Datasets/Kodak \
  --checkpoint ./outputs/stage1/checkpoint_step_00100000.pt \
  --infer_steps 4 \
  --do_entropy_coding

# 多数据集
python flux_tcm_stage1_infer.py \
  --input_dirs /data2/luosheng/data/Datasets/Kodak /data2/luosheng/data/Datasets/CLIC \
  --checkpoint ./outputs/stage1/checkpoint_step_00100000.pt \
  --infer_steps 4 \
  --do_entropy_coding
```

**关键参数说明**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dirs` | **必需** | 输入图像目录（支持多个，空格分隔） |
| `--output_dir` | 自动生成 | 输出目录（自动包含数据集名和 BPP） |
| `--checkpoint` | **必需** | TCM 模型 checkpoint |
| `--infer_steps` | 4 | 去噪步数（与训练一致） |
| `--do_entropy_coding` | True | 启用熵编码计算 BPP |
| `--no_entropy_coding` | False | 禁用熵编码（仅测质量） |
| `--batch_size` | 1 | 批大小（显存充足可增大） |
| `--guidance` | 1.0 | 引导系数 |
| `--mixed_precision` | bf16 | 混合精度 (no/fp16/bf16) |

---

### 场景 3: 性能基准测试

**适用**: 评估不同分辨率下的推理速度

**使用方法**:
```bash
cd /data2/luosheng/code/flux2/Flow

python benchmark_flux_t2i.py \
  --resolutions 512,768,1024 \
  --num_inference_steps 4 \
  --warmup_runs 5 \
  --test_runs 10 \
  --output_dir ./outputs/speed_test \
  --save_images \
  --prompt "A beautiful landscape with mountains and lakes, photorealistic"
```

**输出**:
- `speed_benchmark_t2i.json`: 包含每个分辨率的平均时间、标准差、FPS
- `gen_512x512.png`, `gen_1024x1024.png`: 生成的示例图像

**示例输出**:
```json
[
  {
    "resolution": 512,
    "avg_time_sec": 0.5234,
    "std_time_sec": 0.0123,
    "fps": 1.91,
    ...
  },
  {
    "resolution": 1024,
    "avg_time_sec": 1.4567,
    "std_time_sec": 0.0234,
    "fps": 0.69,
    ...
  }
]
```

---

## 📊 评估指标

### 支持的指标

| 指标 | 库/工具 | 含义 | 范围 | 好坏判断 | 典型值 |
|------|---------|------|------|----------|--------|
| **PSNR** | 内置 | 峰值信噪比 | 0-∞ dB | 越高越好 | >30dB 良好 |
| **MS-SSIM** | `pytorch_msssim` | 多尺度结构相似度 | 0-1 | 越高越好 | >0.9 优秀 |
| **LPIPS** | `lpips` | 感知图像块相似度 | 0-1 | 越低越好 | <0.1 优秀 |
| **DISTS** | `pyiqa` | 深度图像空间变换相似度 | 0-∞ | 越低越好 | <0.2 良好 |
| **BPP** | 内置 | 每像素比特数 | 0-∞ | 越低越好 | 0.1-0.5 |

### 逐图指标输出

推理时会实时打印每张图的详细指标：

```
  [1] img001.png
      PSNR:    32.4523 dB
      MS-SSIM: 0.923456
      LPIPS:   0.082345
      DISTS:   0.123456
      BPP:     0.123400

  [2] img002.png
      PSNR:    31.8765 dB
      MS-SSIM: 0.918234
      LPIPS:   0.085678
      DISTS:   0.128901
      BPP:     0.119800
```

### 输出文件结构

#### 单数据集
```
outputs/infer_Kodak_bpp0.1234/
├── recon/                      # 重建图像
│   ├── img001_recon.png
│   └── ...
├── metrics_all_datasets.csv    # 所有图像的指标
├── metrics_summary.json        # 总体汇总
└── infer_config.json           # 配置备份
```

#### 多数据集
```
outputs/infer_multi_datasets_bpp0.1234/
├── recon/                      # 所有重建图像
│   ├── img001_recon.png
│   └── ...
├── Kodak/                      # 各数据集独立结果
│   ├── metrics.csv
│   └── summary.json
├── CLIC/
│   ├── metrics.csv
│   └── summary.json
├── metrics_all_datasets.csv    # 合并所有数据
├── metrics_summary.json        # 总体汇总（含各数据集统计）
└── infer_config.json
```

### metrics_summary.json 示例

```json
{
  "total_datasets": 2,
  "total_images": 50,
  "datasets": {
    "Kodak": {
      "dataset": "Kodak",
      "num_images": 25,
      "psnr": 32.4523,
      "ms_ssim": 0.923456,
      "lpips": 0.082345,
      "dists": 0.123456,
      "bpp": 0.123400
    },
    "CLIC": {
      "dataset": "CLIC",
      "num_images": 25,
      "psnr": 30.1234,
      "ms_ssim": 0.891234,
      "lpips": 0.095678,
      "dists": 0.145678,
      "bpp": 0.112300
    }
  },
  "overall_average": {
    "psnr": 31.2879,
    "ms_ssim": 0.907345,
    "lpips": 0.089012,
    "dists": 0.134567,
    "bpp": 0.117850
  }
}
```

---

## ⚙️ 参数说明

### 环境变量

```bash
# 必需
export INPUT_DIRS="/path/to/dataset1 /path/to/dataset2"  # 支持多个
export CHECKPOINT=/path/to/model.pt

# 可选
export CUDA_DEVICES=0
export NUM_PROCESSES=1
export MIXED_PRECISION=bf16
export OUTPUT_DIR=./outputs/custom  # 自定义输出目录
```

### 命令行参数

```bash
python flux_tcm_stage1_infer.py \
  --input_dirs /dataset1 /dataset2 \
  --checkpoint /path/to/model.pt \
  --infer_steps 4 \
  --do_entropy_coding \
  --batch_size 1 \
  --guidance 1.0 \
  --mixed_precision bf16 \
  --lora_rank 32 \          # 必须与训练时一致
  --lora_alpha 32.0         # 必须与训练时一致
```

#### TCM 架构参数（通常无需修改）
```bash
--tcm_in_channels     # 输入通道数，默认: 128
--tcm_out_channels    # 输出通道数，默认: 128
--tcm_N               # 默认: 128
--tcm_M               # 默认: 320
--tcm_num_slices      # 切片数，默认: 5
--lora_rank           # LoRA rank，默认: 32（必须与训练时一致）
--lora_alpha          # LoRA alpha，默认: 32.0（必须与训练时一致）
```

**⚠️ 重要**: `--lora_rank` 和 `--lora_alpha` 必须与训练时使用的值完全一致，否则会报维度不匹配错误。

---

## 🚀 性能优化

### 1. 加速基础推理

**方法 A: 移除 CPU Offload**（已应用）
```python
# ❌ 慢：频繁在 CPU/GPU 间传输
pipe.enable_model_cpu_offload()

# ✅ 快：直接在 GPU 上运行
pipe.to(device)
```

**方法 B: 启用 XFormers**（如果可用）
```python
# 安装: pip install xformers
pipe.enable_xformers_memory_efficient_attention()
# 可提升 20-30% 速度，降低 30-40% 显存
```

**方法 C: 使用 Torch Compile**（PyTorch 2.0+）
```python
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
# 首次运行较慢（编译），后续显著提升速度
```

### 2. TCM 推理优化

**增加批大小**:
```bash
# 如果显存充足（>=24GB）
python flux_tcm_stage1_infer.py --batch_size 4
```

**禁用熵编码**（仅测试质量时）:
```bash
python flux_tcm_stage1_infer.py --no_entropy_coding
# 跳过耗时的熵编码步骤，速度提升 2-3 倍
# 注意：BPP 将为 0.0000
```

**多 GPU 并行**:
```bash
export NUM_PROCESSES=2
export CUDA_DEVICES=0,1
bash infer_stage1.sh
```

### 3. 显存优化

**如果显存不足**:
```python
# 方案 1: Sequential CPU Offload（慢但省显存）
pipe.enable_sequential_cpu_offload()

# 方案 2: 减小批大小
--batch_size 1

# 方案 3: 使用更低精度
--mixed_precision fp16  # 或 no（如果 bf16 不支持）
```

**监控显存**:
```python
# 在代码中添加
torch.cuda.memory_allocated() / 1024**3  # 已分配显存 (GB)
torch.cuda.max_memory_allocated() / 1024**3  # 峰值显存 (GB)
```

---

## ❓ 常见问题

### Q1: 缺少 pytorch_msssim 或 pyiqa

**错误信息**: `Warning: pytorch_msssim not installed`

**解决方案**:
```bash
pip install pytorch-msssim pyiqa
```

如果不想安装，对应指标会显示为 0.0，不影响其他指标。

### Q2: 如何只计算部分指标？

目前所有指标都会计算。如果不需要某个指标，可以注释掉相关代码以加快速度。

### Q3: 多数据集推理时如何查看单个数据集的结果？

每个数据集都有独立的文件夹：
```bash
cat outputs/infer_multi_datasets_bpp0.1234/Kodak/summary.json
cat outputs/infer_multi_datasets_bpp0.1234/CLIC/summary.json
```

### Q4: 如何对比不同配置的效果？

运行多次推理，然后对比 `metrics_summary.json` 中的 `overall_average` 字段。

### Q5: LoRA 维度不匹配

确保 `--lora_rank` 和 `--lora_alpha` 与训练时一致（默认都是 32）。

---

## 📊 性能参考

### FLUX.2-klein-4B 文本生成 (RTX 4090)
| 分辨率 | 步数 | 平均时间 | FPS | 显存占用 |
|--------|------|----------|-----|----------|
| 512x512 | 4 | ~0.5s | 2.0 | ~6 GB |
| 768x768 | 4 | ~1.0s | 1.0 | ~10 GB |
| 1024x1024 | 4 | ~1.5s | 0.67 | ~14 GB |

### TCM 压缩重建 (RTX 4090, batch_size=1)
| 分辨率 | 含熵编码 | 不含熵编码 | BPP (典型) |
|--------|----------|------------|------------|
| 256x256 | ~2s | ~0.5s | 0.15 |
| 512x512 | ~5s | ~1s | 0.12 |
| 1024x1024 | ~15s | ~3s | 0.10 |

*注: 实际性能取决于硬件和配置*

---

## 🔗 相关资源

- **训练文档**: [`Flow/TRAINING_README.md`](Flow/TRAINING_README.md)
- **官方文档**: [Black Forest Labs Docs](https://docs.bfl.ai)
- **模型下载**: [Hugging Face - FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

---

## 📝 更新日志

### 2026-04-10
- ✅ **新增**: MS-SSIM 和 DISTS 评估指标
- ✅ **移除**: CLIP 指标
- ✅ **新增**: 逐图打印详细指标
- ✅ **新增**: 支持多数据集同时推理
- ✅ **优化**: 更清晰的输出格式
- ✅ **修复**: Checkpoint 加载兼容两种格式
- ✅ **修复**: LoRA 参数默认值同步

---

**有问题？欢迎提 Issue 或联系维护者！**