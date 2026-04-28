# 推理性能优化指南

## 问题诊断

原始推理时间：**~10秒/张**  
目标推理时间：**1-2秒/张**

## 发现的性能瓶颈

### 🔴 严重问题（已修复）

#### 1. Padding对齐过于严格（影响最大）
**位置**: `modules/pipeline.py` - `encode_images()` 方法

**问题**: 
- 原代码要求图像尺寸能被 **256** 整除
- 对于常见的 512x512 或 1024x1024 图像，这会导致大量不必要的padding
- 例如：513x513 的图像会被padding到 768x768，增加了 **2.25倍** 的计算量！

**修复**:
```python
# 修改前
required_alignment = 256  # ❌ 太严格

# 修改后  
required_alignment = 32   # ✅ 正确对齐要求
```

**理论依据**:
- FLUX.2 AE: 下采样16倍 → z = x/16
- TCM g_a: conv3x3(stride=2) → y = z*16/2 = x/2 (相对于输入)
- TCM window_size=8: y需要能被8整除
- h_a atten_window_size=4: z需要能被4整除 → y需要能被16整除
- 综合：y需要能被lcm(8,16)=16整除
- 因为 y = x/2，所以 **x需要能被32整除**（不是256！）

**预期提升**: 减少padding带来的计算量，对于非标准尺寸图像可提升 **30-50%**

---

#### 2. TCM概率模型重复更新
**位置**: `modules/pipeline.py` - `forward_stage1_infer_with_timing()` 方法

**问题**:
- `self.tcm.update(force=True)` 每次推理都重新计算概率模型
- 这是一个耗时的CPU操作，可能占用 **1-2秒**

**修复**:
```python
# 添加标记，只在首次调用时更新
if not hasattr(self, '_tcm_updated') or self._tcm_updated:
    self.tcm.update(force=True)
    self._tcm_updated = False
```

**预期提升**: 从第二张图开始节省 **1-2秒**

---

#### 3. 频繁的显存清理阻塞GPU
**位置**: `flux_tcm_stage1_infer.py` - 主循环

**问题**:
- 每处理5张图就执行 `torch.cuda.empty_cache()` 和 `gc.collect()`
- 这些操作会阻塞GPU流水线，造成 **200-500ms** 的停顿

**修复**:
```python
# 修改前
if count % 5 == 0:  # ❌ 太频繁
    torch.cuda.empty_cache()
    gc.collect()

# 修改后
if count % 20 == 0:  # ✅ 降低频率
    torch.cuda.empty_cache()
    gc.collect()
```

**预期提升**: 平均每张图节省 **50-100ms**

---

### 🟡 中等问题（可选优化）

#### 4. 熵编码本身就很慢
**位置**: `LIC_TCM/models/tcm_latent.py` - `compress()` / `decompress()`

**问题**:
- RANS熵编码/解码是CPU密集操作，无法通过GPU加速
- 对于高分辨率图像，可能需要 **2-4秒**

**解决方案**:
- 如果不需要精确的BPP计算，使用 `--no_entropy_coding` 跳过熵编码
- 或者考虑使用更快的熵编码算法（如ANS而非RANS）

**测试命令**:
```bash
python flux_tcm_stage1_infer.py \
    --input_dirs /path/to/images \
    --checkpoint /path/to/checkpoint.pth \
    --no_entropy_coding  # 跳过熵编码，速度提升2-4倍
```

---

#### 5. Flux Denoise步骤数
**位置**: 命令行参数 `--infer_steps`

**问题**:
- 默认4步denoise，每步都要运行完整的Flux模型
- Flux模型很大（4B参数），单步可能需要 **0.5-1秒**

**解决方案**:
- 尝试减少到2-3步（可能略微降低质量）
- 或使用蒸馏模型（如果可用）

**测试命令**:
```bash
python flux_tcm_stage1_infer.py \
    --input_dirs /path/to/images \
    --checkpoint /path/to/checkpoint.pth \
    --infer_steps 2  # 从4步减到2步，速度提升约50%
```

---

## 性能分析工具

运行推理时会自动输出详细的性能分解：

```
================================================================================
Performance Breakdown (avg per image, after 10 images):
================================================================================
  Pipeline Total:    2.3456s
    ├─ AE Encode:    0.4523s (19.3%)
    ├─ TCM Compress: 1.2345s (52.6%)     ← 主要瓶颈
    ├─ Flux Denoise: 0.5234s (22.3%)
    └─ AE Decode:    0.1354s (5.8%)
  Metrics:           0.0234s
  I/O Overhead:      0.0123s
  TOTAL:             2.3813s
================================================================================
```

关注 **TCM Compress** 的时间，如果超过1秒，说明熵编码是主要瓶颈。

---

## 推荐配置

### 快速模式（~1-2秒/张）
```bash
python flux_tcm_stage1_infer.py \
    --input_dirs /path/to/images \
    --output_dir ./outputs/fast \
    --checkpoint /path/to/checkpoint.pth \
    --batch_size 1 \
    --infer_steps 2 \
    --no_entropy_coding \
    --skip_metrics \
    --num_workers 4
```

### 平衡模式（~2-3秒/张）
```bash
python flux_tcm_stage1_infer.py \
    --input_dirs /path/to/images \
    --output_dir ./outputs/balanced \
    --checkpoint /path/to/checkpoint.pth \
    --batch_size 1 \
    --infer_steps 4 \
    --do_entropy_coding \
    --num_workers 4
```

### 高质量模式（~4-6秒/张）
```bash
python flux_tcm_stage1_infer.py \
    --input_dirs /path/to/images \
    --output_dir ./outputs/hq \
    --checkpoint /path/to/checkpoint.pth \
    --batch_size 1 \
    --infer_steps 8 \
    --do_entropy_coding \
    --verbose \
    --num_workers 4
```

---

## 进一步优化建议

### 1. 批量推理（如果显存允许）
```bash
--batch_size 4  # 一次处理4张图，吞吐量提升3-4倍
```

### 2. 使用混合精度
```bash
--mixed_precision fp16  # 比bf16更快，但可能略微降低精度
```

### 3. 启用Gradient Checkpointing（仅训练时需要）
推理时不要使用 `--use_gradient_checkpointing`，这会降低20-30%的速度。

### 4. 预热GPU
第一张图的推理通常较慢（CUDA kernel编译、内存分配等），建议先运行一张图预热：
```bash
# 先运行一张图预热
python flux_tcm_stage1_infer.py --input_dirs /path/to/single_image ...

# 再运行完整数据集
python flux_tcm_stage1_infer.py --input_dirs /path/to/dataset ...
```

---

## 故障排查

### 如果速度仍然很慢

1. **检查GPU利用率**:
   ```bash
   nvidia-smi dmon -s u
   ```
   如果GPU利用率低于50%，说明瓶颈在CPU或I/O。

2. **检查是否在使用正确的设备**:
   确保模型和数据都在GPU上：
   ```python
   print(f"Model device: {next(model.parameters()).device}")
   print(f"Data device: {data.device}")
   ```

3. **检查是否有其他进程占用GPU**:
   ```bash
   nvidia-smi
   ```

4. **验证padding是否正确**:
   查看日志中的 `[TCM]` 输出，确认压缩/解压时间是否合理。

---

## 总结

通过以上优化，预期可以将推理时间从 **10秒** 降低到 **1-3秒**：

| 优化项 | 预期提升 | 状态 |
|--------|---------|------|
| Padding对齐修正 | 30-50% | ✅ 已修复 |
| TCM概率模型缓存 | 1-2秒 | ✅ 已修复 |
| 减少显存清理频率 | 50-100ms/张 | ✅ 已修复 |
| 跳过熵编码（可选） | 2-4秒 | ⚠️ 需手动启用 |
| 减少infer_steps（可选） | 0.5-1秒 | ⚠️ 需手动设置 |

**最终目标**: 在保持质量的前提下，达到 **1-2秒/张** 的推理速度。
