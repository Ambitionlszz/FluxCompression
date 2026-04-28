# FlowCompression 训练架构与参数说明

## 1. 配置来源 (`config/train_config.yaml`)

| 模块 | 关键参数 | 说明 |
| :--- | :--- | :--- |
| **Paths** | `flux_ckpt`, `ae_ckpt`, `qwen_ckpt` | Flux 核心组件及文本编码器的权重路径 |
| **GPU** | `cuda_visible_devices`, `mixed_precision` | 指定 GPU 设备及混合精度类型 (bf16) |
| **Data** | `image_size`, `batch_size` | 输入图像尺寸 (256) 与单卡批次大小 |
| **Training** | `max_steps`, `lr`, `grad_clip` | 总步数 (200k), 学习率 (1e-5), 梯度裁剪 |
| **Codec** | `ch_emd`, `channel`, `num_slices` | Codec 内部通道维度及切片/掩码数量 |
| **LoRA** | `rank`, `alpha` | 针对 Flux Transformer 的微调配置 (Rank 32) |
| **Loss** | `lambda_rate`, `d1_mse`, `d2_lpips` | 码率、重建质量及语义一致性的权重分配 |

## 2. 模型组件构成 (`FlowCompression`)

### 核心处理流
1.  **AutoEncoder (AE)**: Flux 原生 VAE，负责像素 $\leftrightarrow$ Latent 转换 (**冻结**)。
2.  **Text Encoder (Qwen3)**: 将文本转为 Embedding，提供语义条件 (**冻结**)。
3.  **ELIC Aux Encoder**: 提取辅助特征，增强上下文信息 (**冻结**)。
4.  **DiT-IC Codec (可训练)**:
    *   **g_a (Analysis)**: 融合 AE Latent (16ch) 与 ELIC 特征 (320ch)，下采样至 320ch。
    *   **Entropy Model**: 包含 `GaussianConditional` 和 `SpatialContext`，执行量化与熵估计。
    *   **g_s (Synthesis)**: 将量化后的 Latent 上采样回原始维度。
    *   **Aux Decoder**: 辅助分支，输出残差以优化重建细节。
5.  **Flux Transformer (DiT)**: 核心生成模块，通过 **LoRA** 注入可训练参数。

## 3. 训练机制

*   **梯度累积**: 通过 `accelerator.accumulate` 实现。等效 Batch Size = `batch_size` × `accumulation_steps` × `num_gpus`。
*   **梯度检查点**: 可选开启，针对 Flux 和 Codec 节省显存（约 40%），但会增加计算耗时。
*   **损失函数**: 联合优化码率 (BPP)、像素级误差 (MSE)、感知质量 (LPIPS) 及语义相似度 (CLIP)。
