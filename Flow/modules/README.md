# Flow Modules 模块说明

本目录包含 FLUX.2 + TCM 项目的核心功能模块，采用高内聚低耦合的设计原则。

## 模块概览

### 📊 `evaluators.py` - 评估器模块

**核心类**: `Stage1Evaluator`

**功能**:
- 实现混合评估策略（单步 + 多步）
- 自动管理评估图像保存
- 随机采样 batch 增加多样性
- 多 GPU 指标同步

**使用示例**:
```python
from Flow.modules.evaluators import Stage1Evaluator

# 初始化评估器
evaluator = Stage1Evaluator(
    accelerator=accelerator,
    output_dir="./outputs/stage1",
    eval_batches=50,
)

# 执行评估
metrics = evaluator.evaluate(
    pipeline=pipeline,
    criterion=criterion,
    val_loader=val_loader,
    global_step=1000,
    use_multistep=True,      # True: 4 步去噪，False: 单步
    clip_ckpt="/path/to/clip",
)
```

**关键方法**:
- `evaluate()`: 主评估接口
- `_compute_multistep_metrics()`: 多步模式指标计算
- `_save_comparison_images()`: 保存对比图
- `_sample_batch_indices()`: 随机采样 batch

---

### 🔄 `pipeline.py` - 流程管道模块

**核心类**: `FlowTCMStage1Pipeline`

**功能**:
- 封装训练和推理的完整流程
- 管理模型组件（Flux, AE, TCM, Text Encoder）
- 提供统一的前向传播接口

**主要方法**:
- `forward_stage1_train()`: 训练模式前向传播
- `forward_stage1_infer()`: 推理模式前向传播
- `encode_images()`: 图像编码
- `decode_latents()`: 潜变量解码

---

### 🎯 `losses.py` - 损失函数模块

**核心类**:
- `Stage1Loss`: Stage1 训练的复合损失
- `CLIPL2Loss`: CLIP 特征空间 L2 损失

**损失组成**:
```python
total_loss = (
    lambda_rate * bpp +           # 率损失
    d1_mse * mse +                # 重建质量
    d2_lpips * lpips_loss +       # 感知质量
    d3_clip * clip_l2             # 语义相似性
)
```

**使用示例**:
```python
from Flow.modules.losses import Stage1Loss

criterion = Stage1Loss(
    clip_path="/path/to/clip",
    lambda_rate=0.5,
    d1_mse=2.0,
    d2_lpips=1.0,
    d3_clip=0.1,
)

loss_dict = criterion(x01, x_hat01, likelihoods)
```

---

### 🔧 `lora.py` - LoRA 适配模块

**核心类**: `LoRALinear`

**功能**:
- LoRA 权重注入到 Linear 层
- 支持正则表达式匹配目标层
- 独立的 LoRA 参数保存和加载

**使用示例**:
```python
from Flow.modules.lora import inject_lora, lora_state_dict, load_lora_state_dict

# 注入 LoRA
stats = inject_lora(
    model=flux_model,
    rank=16,
    alpha=16.0,
    dropout=0.0,
    target_regex=r".*",  # 匹配所有 Linear 层
)

# 保存 LoRA 权重
lora_weights = lora_state_dict(flux_model)
torch.save(lora_weights, "lora.pt")

# 加载 LoRA 权重
missing = load_lora_state_dict(flux_model, torch.load("lora.pt"))
```

---

### 📁 `data.py` - 数据模块

**核心类**:
- `RecursiveImageDataset`: 递归图像数据集
- `ResizeIfSmall`: 智能尺寸调整

**数据变换**:
```python
from Flow.modules.data import build_train_transform, build_val_transform

train_transform = build_train_transform(image_size=256)
val_transform = build_val_transform(image_size=256)
```

**支持的图像格式**:
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`

---

### 🛠️ `utils.py` - 工具函数模块

**核心工具**:
- `AverageMeter`: 滑动平均统计
- `ensure_dir()`: 目录创建
- `save_checkpoint()`: checkpoint 保存
- `save_json()`: JSON 序列化
- `write_csv()`: CSV 写入

**使用示例**:
```python
from Flow.modules.utils import AverageMeter, ensure_dir

# 统计指标
meter = AverageMeter()
meter.update(0.5, n=32)  # 值=0.5, batch_size=32
print(f"平均值：{meter.avg}")

# 确保目录存在
ensure_dir("./outputs/stage1")
```

---

### 📦 `model.py` - 统一模型架构 ⭐ NEW

**核心类**: `FluxTCMModel`

**功能**:
- 整合 FLUX.2 + TCM 的完整流程
- 提供训练和推理两种模式的清晰接口
- 参考业界领先的代码组织结构（如 DiT-IC）

**主要接口**:
```python
from Flow.model import FluxTCMModel

# 初始化模型
model = FluxTCMModel(
    model_name="flux.2-klein-4b",
    flux_ckpt="/path/to/flux.safetensors",
    ae_ckpt="/path/to/ae.safetensors",
    qwen_ckpt="/path/to/qwen",
    tcm_config=tcm_args,
    device=torch.device("cuda"),
)

# 训练模式 - forward
output = model(x01, train_schedule_steps=50)

# 推理模式 - compress
z_clean = model.encode_images(x01)
compressed = model.compress(z_clean, do_entropy_coding=True)

# 推理模式 - decompress
recon = model.decompress(compressed["strings"], compressed["shapes"], infer_steps=4)

# 或者使用一体化接口
output = model.forward_infer(x01, infer_steps=4, do_entropy_coding=True)
```

**设计优势**:
- ✅ **职责分离**: 训练 (`forward`) 和推理 (`compress`/`decompress`) 接口清晰
- ✅ **易于部署**: 压缩和解压可作为独立操作调用
- ✅ **配置集中**: TCM 参数通过配置字典统一管理
- ✅ **兼容标准**: 参考主流压缩框架的 API 设计，便于扩展

---

## 模块依赖关系

```
flux_tcm_stage1_train.py
├── modules/evaluators.py
│   ├── modules/utils.py
│   └── modules/losses.py (CLIPL2Loss)
├── modules/pipeline.py
│   ├── flux2.* (外部依赖)
│   └── LIC_TCM.models (外部依赖)
├── modules/lora.py
├── modules/data.py
└── modules/utils.py
```

---

## 扩展指南

### 添加新的评估指标

1. 在 `evaluators.py` 的 `meters` 字典中添加新键
2. 在 `_compute_multistep_metrics()` 中计算该指标
3. 更新返回的 `loss_dict`

### 添加新的损失函数

1. 在 `losses.py` 中创建新的 `nn.Module` 类
2. 实现 `forward()` 方法
3. 在 `Stage1Loss` 中集成使用

### 自定义数据增强

1. 在 `data.py` 中创建新的 transform 类
2. 修改 `build_train_transform()` 或 `build_val_transform()`

---

## 最佳实践

✅ **单一职责**: 每个模块只做一件事  
✅ **接口清晰**: 通过 docstring 明确输入输出  
✅ **延迟初始化**: 如 metrics 在需要时才创建  
✅ **配置分离**: 超参数通过构造函数传入  
✅ **错误处理**: 关键位置添加异常检查和日志  

---

## 常见问题

**Q: 为什么评估器要延迟初始化 metrics？**  
A: 避免在不需要多步评估时浪费显存，同时简化训练脚本的初始化逻辑。

**Q: 如何调整评估时保存的图片数量？**  
A: 修改 `_sample_batch_indices()` 中的 `n_samples` 参数（默认 5）。

**Q: 多步评估的 bpp 为什么是 0？**  
A: 多步推理模式不使用熵编码，无法获得真实的概率估计，因此 bpp 设为占位符 0。
