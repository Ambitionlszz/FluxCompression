# FluxCodec Stage 2 (GAN 对抗微调) 移植与设计文档

本项目第二阶段训练（Stage 2）的目的是在第一阶段重构质量（R-D Loss）的基础上，引入生成对抗网络（GAN）以进一步改善生成图像的纹理细节和真实感。

本设计完全参考并对齐了 **StableCodec** (`StableCodec/src/finetune.py`) 的训练机制，仅在 Flow Matching 骨干网络（FLUX.2）的特定前向传播中结合了 StableCodec 的损失与判别器计算。

---

## 1. 损失函数设计 (Loss Formulations)

Stage 2 的生成器损失遵循 StableCodec 的 R-D + GAN 联合损失设计：

$$\mathcal{L}_{\text{total}} = \lambda_{\text{rate}} \cdot \mathcal{L}_{\text{rate}} + \mathcal{L}_{D}$$

其中，失真与对抗损失项 $\mathcal{L}_{D}$ 定义为：

$$\mathcal{L}_{D} = \lambda_{l2} \cdot \mathcal{L}_{\text{MSE}} + \lambda_{\text{lpips}} \cdot \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{clip}} \cdot \mathcal{L}_{\text{CLIP}} + \lambda_{\text{gan}} \cdot \mathcal{L}_{\text{adv}}$$

各损失项的具体计算和对齐如下：

*   **Rate Loss ($\mathcal{L}_{\text{rate}}$)**：利用 LatentCodec 输出的 likelihoods 计算的平均 bpp（Bits Per Pixel）。
*   **MSE Loss ($\mathcal{L}_{\text{MSE}}$)**：重建图像 $x_{\text{hat}}$ 与原图 $x$ 的均方误差。
*   **LPIPS Loss ($\mathcal{L}_{\text{LPIPS}}$)**：利用 VGG 提取的感知损失（输入调整至 $[-1, 1]$ 范围内）。
*   **CLIP Loss ($\mathcal{L}_{\text{CLIP}}$)**：利用 CLIP Vision Model 提取特征，并计算其余弦相似度距离（归一化后的 $L_2$ 范数），确保高级语义的一致性。
*   **GAN Adversarial Loss ($\mathcal{L}_{\text{adv}}$)**：生成器对抗损失。由判别器计算 $x_{\text{hat}}$ 的预测得分，公式为：  
    $$\mathcal{L}_{\text{adv}} = \text{mean}(\text{Discriminator}(x_{\text{hat}}, \text{for\_G=True}))$$

---

## 2. 判别器设计 (Discriminator)

直接使用 StableCodec 的 `vision_aided_loss.Discriminator`。该判别器基于预训练的 **DINOv2** 特征骨干：

*   **骨干网络**：`dinov2_reg`，提取多尺度特征表示。在训练中，**DINOv2 骨干网络保持冻结**（`requires_grad_(False)`）。
*   **可学习头部**：判别器分类头部使用多尺度卷积线性投影层（`conv_multi_level`），用于对特征图的真实性进行分类。
*   **损失类型**：默认使用多尺度带平滑的 Sigmoid 损失（`multilevel_sigmoid_s`），对生成样本和真实样本进行二分类判别。

---

## 3. 优化与训练流程 (Training & Optimization Loop)

训练采用交替优化的三步模式，完全对齐 StableCodec 的更新机制：

1.  **生成器前向与更新 (Generator Step)**：
    *   通过 FLUX 管道生成重建图像 $x_{\text{hat}}$。
    *   通过 `net_disc(x_hat, for_G=True)` 评估对抗损失 `loss_adv`。
    *   计算联合 R-D 损失 `gen_loss`。
    *   反向传播并优化 **Generator 参数**（包括 `LatentCodec` 以及 FLUX 的 `LoRA` 权重）。
    *   *注：若有辅助参数（如熵模型的分位数），则在该步骤后单独更新 `aux_optimizer`。*
2.  **判别器 Real 更新 (Discriminator Real Step)**：
    *   计算真实图像的判别器损失：`loss_real = net_disc(x.detach(), for_real=True) * lambda_gan`。
    *   反向传播并仅优化 **Discriminator 参数**。
3.  **判别器 Fake 更新 (Discriminator Fake Step)**：
    *   计算重建图像（需 `detach()`）的判别器损失：`loss_fake = net_disc(x_hat.detach(), for_real=False) * lambda_gan`。
    *   反向传播并仅优化 **Discriminator 参数**。

---

## 4. 关键文件与结构映射

以下是为 FluxCodec 移植的 Stage 2 代码文件：

*   **损失层设计**：`FluxCodec/modules/losses_stage2.py`
    *   实现 `Stage2Loss` 类。
    *   将对抗损失项解耦为外部传入的参数 `loss_adv`，从而保持纯粹的 $R + \lambda D$ 的逻辑。
*   **训练脚本**：`FluxCodec/train_stage2.py`
    *   集成了 `vision_aided_loss` 判别器。
    *   实现交替三步训练循环，并包括自动在指定 milestone 手动调整 Generator 学习率（LR decay）的功能。
*   **启动 Shell 脚本**：`FluxCodec/train_stage2.sh`
    *   封装训练参数。支持混合精度（`bf16`），单卡或多卡 `accelerate launch`。

---

## 5. 快速启动与参数配置

### 配置文件调整

在启动前，请确保在 `train_stage2.sh` 中修改 `--stage1_ckpt` 为你实际的第一阶段训练权重（例如：`stage1_last.pt`），否则初始化时会报错。

### 运行指令

```bash
cd d:/Projects/FluxCompression/FluxCodec
bash train_stage2.sh
```

### 推荐超参数设置

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--lr` | `5e-5` | Generator (LoRA + Codec) 起始学习率 |
| `--lr_disc` | `2e-5` | Discriminator 起始学习率 |
| `--lambda_rate` | `0.3` | 速率控制权重（$\lambda_{\text{rate}}$） |
| `--lambda_l2` | `2.0` | 像素误差控制权重（$\lambda_{l2}$） |
| `--lambda_lpips` | `1.0` | LPIPS 感知误差权重（$\lambda_{\text{lpips}}$） |
| `--lambda_clip` | `0.1` | CLIP 高级语义相似度权重（$\lambda_{\text{clip}}$） |
| `--lambda_gan` | `0.1` | 判别器对抗损失权重（$\lambda_{\text{gan}}$） |
| `--lr_decay_steps` | `"10000,20000,35000"` | 学习率衰减触发步数 |
| `--lr_decay_values`| `"2e-5,1e-5,1e-6"` | 对应步数下的学习率数值 |
| `--gan_loss_type` | `"multilevel_sigmoid_s"` | 判别器二分类损失类型 |
| `--disc_cv_type` | `"dinov2_reg"` | 判别器使用的骨干网络特征源 |

---

## 6. 与 Stage 1 的主要差异

1.  **从已训权重恢复**：Stage 2 明确需要加载 Stage 1 checkpoint 进行微调（LoRA + Codec 均为预训练状态）。
2.  **网络结构扩展**：在内存中引入了 DINOv2 判别器网络（`net_disc`）。
3.  **学习率调度变化**：由第一阶段的恒定/余弦调度（或微小调整），变为在 10k, 20k, 35k 步数时，手动进行大幅降阶的阶梯式衰减（对齐 StableCodec 策略以保证 GAN 微调阶段的稳定性）。
1