# newFlow/modules

本目录包含 newFlow (DiT-IC 架构的 Flux2 压缩模型) 训练和评估所需的核心模块。

## 📁 模块结构

```
modules/
├── __init__.py          # 包初始化,导出所有公共接口
├── data.py              # 数据加载和预处理
├── losses.py            # 损失函数
├── evaluators.py        # 评估器
├── lora.py              # LoRA 微调支持
└── utils.py             # 工具函数
```

## 🔧 各模块说明

### 1. data.py - 数据加载

提供图像数据集的递归扫描、预处理和数据加载功能。

**核心类:**
- `RecursiveImageDataset`: 自动扫描目录及其子目录下的所有图像
- `ResizeIfSmall`: 如果图像小于指定尺寸则放大

**核心函数:**
- `build_train_transform(image_size)`: 构建训练变换 (RandomCrop + Flip)
- `build_val_transform(image_size)`: 构建验证变换 (CenterCrop)
- `build_dataloader(...)`: 构建 DataLoader

**使用示例:**
```python
from newFlow.modules import build_train_transform, RecursiveImageDataset, build_dataloader

transform = build_train_transform(image_size=256)
dataset = RecursiveImageDataset(root="/path/to/images", transform=transform)
loader = build_dataloader(dataset, batch_size=2, num_workers=4, shuffle=True, drop_last=True)
```

---

### 2. losses.py - 损失函数

提供综合的损失函数,包含码率、重建质量、感知质量和语义一致性。

**核心类:**
- `Stage1Loss`: Stage1 训练的综合损失函数
  - BPP (码率损失)
  - MSE (像素级重建)
  - LPIPS (感知质量)
  - CLIP L2 (语义一致性)

**使用示例:**
```python
from newFlow.modules import Stage1Loss

loss_fn = Stage1Loss(
    clip_path="/path/to/clip-vit-base-patch32",
    lambda_rate=0.5,
    d1_mse=2.0,
    d2_lpips=1.0,
    d3_clip=0.1,
)

losses = loss_fn(x01, x_hat, likelihoods)
total_loss = losses["loss"]
```

---

### 3. evaluators.py - 评估器

在训练过程中定期评估模型性能,并保存可视化对比图。

**核心类:**
- `Stage1Evaluator`: 评估 FlowCompression 模型的重建质量

**使用示例:**
```python
from newFlow.modules import Stage1Evaluator

evaluator = Stage1Evaluator(
    output_dir="./outputs/eval",
    eval_batches=10,
    accelerator=accelerator,  # 可选
)

metrics = evaluator.evaluate(
    model=model,
    elic_model=elic_encoder,
    val_loader=val_loader,
    global_step=step,
    clip_ckpt="/path/to/clip",
    infer_steps=4,
)

print(f"BPP: {metrics['bpp']:.4f}, MSE: {metrics['mse']:.6f}")
```

---

### 4. lora.py - LoRA 微调

为 Flux Transformer 提供高效的参数微调方法。

**核心类:**
- `LoRALinear`: 带 LoRA 的线性层
- `LoRAStats`: LoRA 注入统计信息

**核心函数:**
- `inject_lora(...)`: 为模型注入 LoRA
- `lora_state_dict(...)`: 提取 LoRA 参数
- `load_lora_state_dict(...)`: 加载 LoRA 参数

**使用示例:**
```python
from newFlow.modules import inject_lora, lora_state_dict

# 注入 LoRA
stats = inject_lora(
    model=flux_transformer,
    rank=32,
    alpha=32.0,
    dropout=0.0,
    target_regex=r".*\.attn\..*",  # 只注入注意力层
)
print(f"Injected {stats.injected_layers} layers, {stats.trainable_params} params")

# 保存 LoRA 参数
torch.save(lora_state_dict(model), "lora_weights.pth")
```

---

### 5. utils.py - 工具函数

提供常用的辅助函数。

**核心类:**
- `AverageMeter`: 平均值统计器

**核心函数:**
- `set_global_seed(seed)`: 设置全局随机种子
- `ensure_dir(path)`: 确保目录存在
- `save_json(path, obj)`: 保存 JSON 文件
- `save_checkpoint(path, state)`: 保存模型检查点
- `write_csv(path, rows)`: 写入 CSV 文件

**使用示例:**
```python
from newFlow.modules import AverageMeter, set_global_seed, save_checkpoint

# 设置随机种子
set_global_seed(42)

# 统计平均值
meter = AverageMeter()
for loss in losses:
    meter.update(loss.item(), batch_size)
print(f"Average loss: {meter.avg:.4f}")

# 保存检查点
save_checkpoint("checkpoint.pth", {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": current_step,
})
```

---

## 🎯 与 Flow/modules 的区别

| 特性 | Flow/modules | newFlow/modules |
|------|-------------|----------------|
| **架构** | TCMLatent (原始 Flow) | FlowCompression (DiT-IC) |
| **辅助编码** | 无 | ELIC Aux Encoder |
| **评估器输入** | pipeline | model + elic_model |
| **文档** | 基础注释 | 详细中文文档 |

---

## 📝 注意事项

1. **图像尺寸**: 建议使用 256x256 (与 Flow 保持一致),通过 `build_train_transform(256)` 配置
2. **ELIC 集成**: 评估时需要提供 ELIC 辅助编码器以生成 `z_aux`
3. **LoRA 目标**: 根据任务选择合适的 `target_regex`,避免注入过多层导致显存不足
4. **多 GPU**: 如果使用 Accelerate,评估器会自动处理多 GPU 同步

---

## 🚀 快速开始

完整的训练流程请参考 [`newFlow/train.py`](../train.py)。

基本步骤:
```python
from newFlow.modules import *

# 1. 准备数据
transform = build_train_transform(256)
dataset = RecursiveImageDataset("/path/to/train", transform)
loader = build_dataloader(dataset, batch_size=2, num_workers=4, shuffle=True)

# 2. 定义损失
loss_fn = Stage1Loss(clip_path="/path/to/clip")

# 3. 注入 LoRA (可选)
stats = inject_lora(model.flux, rank=32, alpha=32.0, dropout=0.0, target_regex=r".*")

# 4. 训练循环
for batch in loader:
    x01 = batch.to(device)
    z_aux = elic_model(x01)
    output = model(x01, z_aux, train_schedule_steps=50)
    losses = loss_fn(x01, output["x_hat"], output["likelihoods"])
    losses["loss"].backward()
    optimizer.step()
```

---

**最后更新**: 2026-04-20  
**作者**: Lingma Assistant
