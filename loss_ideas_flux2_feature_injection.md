# FLUX.2[Klein] 压缩重建质量提升：Feature Injection Loss Ideas

## 目标
在现有 `TCM -> FLUX.2[Klein] -> VAE` 链路上，设计更有新意、且可能有效提升重建质量的 loss，避免只做常规 pixel MSE 堆叠。

---

## 1. Flow-Path Invariant Loss（最推荐）
**灵感**：RDEIC 的 fixed-step train/infer 对齐。  
**核心思想**：同一图像在不同时间点预测到的 clean latent 应该一致。

设两个时刻 `t_a, t_b`：

```math
\hat z_0^{(a)} = x_{t_a} - t_a v_\theta(x_{t_a}, t_a),
\quad
\hat z_0^{(b)} = x_{t_b} - t_b v_\theta(x_{t_b}, t_b)
```

```math
L_{path} = \|\hat z_0^{(a)} - \hat z_0^{(b)}\|_2^2
```

**为什么可能有效**：
- 直接约束 rectified-flow 的时间一致性。
- 对少步（single-step/4-step）解码特别关键，可减轻结构漂移。

---

## 2. Relay Self-Distill Loss（单步对齐多步）
**灵感**：RDEIC 的 relay residual。  
**核心思想**：用 4-step ODE 的结果作为 teacher，蒸馏给单步重建。

```math
L_{relay} = \|\hat z_{0,single} - \hat z_{0,ode4}^{sg}\|_2^2
```

其中 `sg` 表示 stop-gradient。

**为什么可能有效**：
- 让单步路径学习多步路径的稳定目标。
- 低码率下可减少纹理抖动与几何错位。

---

## 3. Uncertainty-Weighted Latent MSE（内容自适应加权）
**灵感**：CADC 的 uncertainty-guided quantization。  
**核心思想**：不同区域应使用不同强度监督，不做“全局均匀惩罚”。

利用 hyperprior `scales`（或单独 uncertainty head）得到权重图 `w`：

```math
L_{uw} = \frac{1}{N} \sum_i w_i (\hat z_{0,i} - z_{0,i})^2
```

**为什么可能有效**：
- 高频复杂区域更受约束，平坦区域少干预。
- 通常比统一 MSE 更符合 RD 权衡。

---

## 4. Hyperprior-to-FLUX Bridge Loss（新意较强）
**灵感**：DiffEIC 的空间对齐 + CADC 信息聚焦。  
**核心思想**：把 TCM 可熵编码先验（`means/scales`）对齐到 FLUX 中间层语义特征。

对若干中层特征 `h_k`：

```math
L_{bridge} = \sum_k \|P_k([\mu, \sigma]) - Pool(h_k)^{sg}\|_2^2
```

- `P_k`: 小投影器（MLP/1x1 Conv）
- `Pool`: token 到空间或全局聚合

**为什么可能有效**：
- 让“可编码统计信息”与“生成语义特征”建立稳定映射。
- 尤其利于低码率下大结构与细节共存。

---

## 5. Dual-Branch Agreement Loss（主辅分支一致性）
**灵感**：StableCodec 的 dual-branch。  
**核心思想**：主分支（FLUX）与辅分支（轻量重建头）互相约束，降低 hallucination。

```math
L_{dual} = \|\hat z^{main} - \hat z^{aux}\|_2^2
```

可选也在 pixel 空间加一致性：

```math
L_{dual,pix} = \|\hat x^{main} - \hat x^{aux}\|_2^2
```

**为什么可能有效**：
- 辅分支提供保真锚点，主分支负责细节生成。
- 在 GAN 或强生成先验阶段更稳。

---

## 6. Bitrate-Free Semantic Context Distill Loss（可选激进）
**灵感**：CADC 的 bitrate-free contextual conditioning。  
**核心思想**：把 hyperprior 变成 pseudo-context token（不进比特流），并对齐冻结语义编码器特征。

```math
L_{sem} = \|T(hyper) - E_{img}(x)\|_2^2
```

- `T`: hyper -> context token 投影
- `E_img`: 冻结图像语义编码器（如 DINO/CLIP image branch）

**为什么可能有效**：
- 你的 FLUX.2 本身是 context 驱动结构，充分利用其优势。
- 不增加传输码率，但可提升语义保真与结构稳定。

---

## 建议优先级（从快到激进）
1. `L_path`
2. `L_relay`
3. `L_bridge`
4. `L_uw`
5. `L_dual`
6. `L_sem`

如果先做一个“小而新”的组合，建议：
- `L_path + L_relay + L_bridge`

---

## 参考论文/项目
- DiffEIC / Toward Extreme Image Compression with Latent Feature Guidance and Diffusion Prior  
  https://arxiv.org/abs/2404.18820
- RDEIC  
  https://arxiv.org/abs/2410.02640
- StableCodec  
  https://arxiv.org/abs/2506.21977
- CADC  
  https://arxiv.org/abs/2602.21591
- DiffEIC GitHub  
  https://github.com/huai-chang/DiffEIC
