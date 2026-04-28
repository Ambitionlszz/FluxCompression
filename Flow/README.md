# Flux2/Flow TCM 推理模块

## 🎯 功能概述

本模块实现了基于 FLUX.2-klein 和 TCM (Transform-based Compression Model) 的图像压缩与重建推理。

### 核心特性

- ✅ **高质量图像压缩**: 结合 FLUX.2 的强大生成能力和 TCM 的高效压缩
- ✅ **全面评估指标**: PSNR、MS-SSIM、LPIPS、DISTS、BPP
- ✅ **智能输出命名**: 输出目录自动包含数据集名称和平均 BPP
- ✅ **多数据集支持**: 一次推理可处理多个数据集
- ✅ **逐图指标输出**: 实时显示每张图像的详细指标
- ✅ **进度可视化**: 实时显示推理进度和当前平均指标
- ✅ **多 GPU 支持**: 使用 Accelerate 实现分布式推理
- ✅ **混合精度**: 支持 bf16/fp16 加速推理

---

## 🚀 快速开始

### 安装依赖

```bash
# 基础依赖
pip install torch torchvision accelerate lpips tqdm pillow

# MS-SSIM 支持
pip install pytorch-msssim

# DISTS 支持
pip install pyiqa
```

### 基本用法（单数据集）

```bash
cd /data2/luosheng/code/flux2/Flow

# 设置输入目录和 checkpoint
export INPUT_DIRS=/data2/luosheng/data/Datasets/Kodak
export CHECKPOINT=./outputs/stage1/checkpoint_step_00100000.pt
export CUDA_DEVICES=0

# 运行推理
bash infer_stage1.sh
```

**输出目录会自动命名为**: `outputs/infer_Kodak_bpp0.1234/`

### 多数据集推理

```bash
# 方式 1: 使用环境变量（空格分隔）
export INPUT_DIRS="/data2/luosheng/data/Datasets/Kodak /data2/luosheng/data/Datasets/CLIC /data2/luosheng/data/Datasets/Tecnick"
bash infer_stage1.sh

# 方式 2: 直接传递参数
python flux_tcm_stage1_infer.py \
  --input_dirs /data2/luosheng/data/Datasets/Kodak /data2/luosheng/data/Datasets/CLIC \
  --checkpoint ./outputs/stage1/checkpoint_step_00100000.pt
```

**输出目录**: `outputs/infer_multi_datasets_bpp0.1234/`

---

## 📁 文件说明

### 核心文件

| 文件 | 说明 |
|------|------|
| [`infer_stage1.sh`](infer_stage1.sh) | **主推理脚本**（推荐使用） |
| [`flux_tcm_stage1_infer.py`](flux_tcm_stage1_infer.py) | Python 推理实现 |
| [`modules/pipeline.py`](modules/pipeline.py) | TCM 推理管道 |

### 文档

| 文件 | 说明 |
|------|------|
| [`INFERENCE_GUIDE.md`](INFERENCE_GUIDE.md) | **详细使用指南** |
| [`TRAINING_README.md`](TRAINING_README.md) | 训练相关文档 |

---

## ✨ 最新功能

### 1. 全面的评估指标

| 指标 | 库/工具 | 含义 | 好坏判断 |
|------|---------|------|----------|
| **PSNR** | 内置 | 峰值信噪比 | 越高越好 (>30dB) |
| **MS-SSIM** | `pytorch_msssim` | 多尺度结构相似度 | 越高越好 (接近1) |
| **LPIPS** | `lpips` | 感知图像块相似度 | 越低越好 (<0.1) |
| **DISTS** | `pyiqa` | 深度图像空间变换相似度 | 越低越好 |
| **BPP** | 内置 | 每像素比特数 | 越低越好 |

### 2. 逐图指标输出

推理时会实时打印每张图的详细指标：

```
  [1] img001.png
      PSNR:    32.4523 dB
      MS-SSIM: 0.923456
      LPIPS:   0.082345
      DISTS:   0.123456
      BPP:     0.123400
```

### 3. 多数据集支持

- 一次推理可处理多个数据集
- 每个数据集有独立的统计结果
- 最终生成总体汇总报告

---

## 📊 输出指标

### 输出文件结构

```
outputs/infer_Kodak_bpp0.1234/
├── recon/                          # 重建图像
│   ├── img001_recon.png
│   ├── img002_recon.png
│   └── ...
├── Kodak/                          # 单个数据集的结果（多数据集时）
│   ├── metrics.csv                 # 每张图的详细指标
│   └── summary.json                # 该数据集的平均指标
├── metrics_all_datasets.csv        # 所有数据集的合并结果
├── metrics_summary.json            # 总体汇总（包含各数据集和总平均）
└── infer_config.json               # 推理配置
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

## ⚙️ 参数配置

### 环境变量

```bash
# 必需
export INPUT_DIRS="/path/to/dataset1 /path/to/dataset2"  # 支持多个目录
export CHECKPOINT=/path/to/model.pt                       # TCM 模型 checkpoint

# 可选
export CUDA_DEVICES=0                  # GPU 设备（默认: 0）
export NUM_PROCESSES=1                 # GPU 数量（默认: 1）
export MIXED_PRECISION=bf16            # 混合精度（默认: bf16）
export OUTPUT_DIR=./outputs/custom     # 自定义输出目录（可选）
```

### 命令行参数

```bash
python flux_tcm_stage1_infer.py \
  --input_dirs /path/to/dataset1 /path/to/dataset2 \
  --checkpoint /path/to/model.pt \
  --infer_steps 4 \              # 去噪步数（默认: 4）
  --do_entropy_coding \          # 启用熵编码（计算 BPP）
  --batch_size 1 \               # 批大小（默认: 1）
  --guidance 1.0 \               # 引导系数（默认: 1.0）
  --mixed_precision bf16         # 混合精度（默认: bf16）
```

**常用参数**:
- `--no_entropy_coding`: 禁用熵编码（仅测试质量，速度更快）
- `--infer_steps 8`: 增加去噪步数（可能提升质量）
- `--batch_size 4`: 增大批大小（需要更多显存）

---

## 🔧 常见问题

### Q1: 缺少 pytorch_msssim 或 pyiqa

**错误信息**: `Warning: pytorch_msssim not installed` 或 `Warning: pyiqa not installed`

**解决方案**:
```bash
# 安装 MS-SSIM
pip install pytorch-msssim

# 安装 DISTS
pip install pyiqa

# 如果不想安装某个库，对应指标会显示为 0.0
```

### Q2: 输出目录没有自动重命名？

**检查**:
```bash
# 1. 确认推理成功完成
ls -lh outputs/infer_*/metrics_summary.json

# 2. 查看 BPP 值
cat outputs/infer_*/metrics_summary.json | grep bpp
```

**解决**:
```bash
# 手动重命名（如果需要）
mv outputs/infer_Kodak_20260410_084530 outputs/infer_Kodak_bpp0.1234
```

### Q3: 显存不足 (OOM)

```bash
# 减小批大小
python flux_tcm_stage1_infer.py --batch_size 1

# 或使用更低精度
export MIXED_PRECISION=fp16
```

### Q4: LoRA 维度不匹配错误

**错误信息**: `RuntimeError: The size of tensor a (16) must match the size of tensor b (32)`

**解决方案**:

确保推理时使用的 LoRA 参数与训练时完全一致：

```bash
# 查看训练配置
cat /path/to/checkpoint_dir/train_config.json | grep lora

# 或者查看训练脚本的默认值
# flux_tcm_stage1_train.py 默认: --lora_rank 32 --lora_alpha 32.0

# 推理时使用相同的值（现在已自动匹配）
bash infer_stage1.sh

# 如果手动指定，确保一致
python flux_tcm_stage1_infer.py \
  --checkpoint /path/to/checkpoint.pt \
  --lora_rank 32 \      # 必须与训练时一致
  --lora_alpha 32.0     # 必须与训练时一致
```

**当前默认值**:
- 训练脚本: `--lora_rank 32 --lora_alpha 32.0`
- 推理脚本: `--lora_rank 32 --lora_alpha 32.0` ✅ 已同步

---

### Q5: Checkpoint 格式不匹配

**错误信息**: `KeyError: 'tcm'` 或 `KeyError: 'pipeline'`

**原因**: Checkpoint 有两种保存格式

**解决方案**:

推理脚本已自动兼容两种格式：

1. **定期保存格式** (`checkpoint_step_XXXXX.pt`): 包含 `"pipeline"` 键
2. **最终保存格式** (`stage1_last.pt`): 包含 `"tcm"` 和 `"flux_lora"` 键

无需手动处理，脚本会自动识别。

---

### Q6: 图像尺寸错误 (EinopsError)

**错误信息**: 
```
einops.EinopsError: Shape mismatch, can't divide axis of length 169 in chunks of 2
Input tensor shape: torch.Size([1, 32, 169, 255])
```

**原因**: 
AutoEncoder 要求输入图像的尺寸必须能被 patch size（通常是 2）整除。DIV2K 等数据集中有些图像尺寸是奇数（如 169x255），导致编码失败。

**解决方案**: 

✅ **已自动修复** - 推理脚本现在会自动处理任意尺寸的图像：

1. **编码前自动 padding**: 使用 reflect padding 将图像尺寸补齐到 2 的倍数
2. **解码后自动裁剪**: 去除 padding，恢复原始尺寸
3. **完全透明**: 无需任何额外配置，所有尺寸的图像都能正常处理

**技术细节**:
```python
# 自动计算需要的 padding
pad_h = (2 - height % 2) % 2
pad_w = (2 - width % 2) % 2

# 使用 reflect padding（比 zero padding 效果更好）
x_padded = F.pad(x01, (0, pad_w, 0, pad_h), mode='reflect')

# 编码、处理、解码...

# 最后裁剪回原始尺寸
x = x[:, :, :original_height, :original_width]
```

**支持的图像尺寸**: 
- ✅ 任意尺寸（包括奇数尺寸）
- ✅ 不同长宽比
- ✅ 多数据集混合（每个图像独立处理）

**示例**:
```bash
# 现在可以安全地处理包含各种尺寸图像的数据集
export INPUT_DIRS="/data2/luosheng/data/Datasets/Kodak /data2/luosheng/data/Datasets/DIV2K_valid_HR"
bash infer_stage1.sh
```

---

### Q7: 缺少 pytorch_msssim 或 pyiqa

**错误信息**: `Warning: pytorch_msssim not installed` 或 `Warning: pyiqa not installed`

**解决方案**:
```
# 安装 MS-SSIM
pip install pytorch-msssim

# 安装 DISTS
pip install pyiqa

# 如果不想安装某个库，对应指标会显示为 0.0
```

---

## 📈 性能参考

### TCM 压缩重建 (RTX 4090, batch_size=1)

| 分辨率 | 含熵编码 | 不含熵编码 | BPP (典型) |
|--------|----------|------------|------------|
| 256x256 | ~2s | ~0.5s | 0.15 |
| 512x512 | ~5s | ~1s | 0.12 |
| 1024x1024 | ~15s | ~3s | 0.10 |

*注: 实际性能取决于硬件和配置*

---

## 🔗 相关链接

- **详细使用指南**: [`INFERENCE_GUIDE.md`](INFERENCE_GUIDE.md)
- **训练文档**: [`TRAINING_README.md`](TRAINING_README.md)
- **官方文档**: [Black Forest Labs Docs](https://docs.bfl.ai)
- **模型下载**: [Hugging Face - FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

---

## 📝 更新日志

### 2026-04-10
- ✅ **新增**: 输出目录自动命名功能（包含数据集名和平均 BPP）
- ✅ **新增**: 支持多数据集同时推理
- ✅ **新增**: 添加 MS-SSIM 和 DISTS 评估指标
- ✅ **移除**: CLIP 指标
- ✅ **优化**: 逐图打印详细指标
- ✅ **优化**: 添加 tqdm 进度条和实时指标显示
- ✅ **优化**: 添加图像加载错误处理
- ✅ **优化**: 改进输出格式和提示信息
- ✅ **修复**: Checkpoint 加载兼容两种格式
- ✅ **修复**: LoRA 参数默认值与训练脚本同步

---

**有问题？欢迎提 Issue 或联系维护者！**
