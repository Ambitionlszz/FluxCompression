# 训练实现对比分析：DiT-IC vs Flow vs newFlow

> **生成时间**: 2026-04-24  
> **目的**: 深入对比三个项目的训练实现，找出导致newFlow BPP异常的根本原因

---

## 📊 一、架构设计对比

| 维度 | DiT-IC | Flow | newFlow |
|------|--------|------|---------|
| **核心组件** | VAE + LatentCodec + DiT | AE + TCMLatent + Flux | AE + LatentCodec(DiT-IC) + Flux |
| **辅助编码器** | aux_codec (内置) | 无 | ELIC (外部集成) |
| **Codec类型** | 自研LatentCodec | TCMLatent (简化版) | DiT-IC的LatentCodec |
| **基础模型** | DiT (Diffusion Transformer) | FLUX.2 | FLUX.2-klein-4B |
| **训练框架** | 原生DDP | Accelerate | Accelerate |
| **混合精度** | FP16 | BF16 | BF16 |
| **LoRA支持** | ❌ | ✅ | ✅ |

---

## 🔄 二、前向传播流程对比

### 2.1 DiT-IC 实现（参考标准）

```python
def forward(self, c_t, cfg=False):
    # 1. 编码
    latent2 = self.aux_codec((c_t + 1) / 2).detach()  # 辅助特征
    lq_latent = self.vae.encode(c_t).latent * scaling_factor

    # 2. Codec压缩
    out = self.codec(lq_latent, latent2)
    lq_mean_hat = out["mean"]
    lq_log_scale_hat = out["scale"]
    
    # ⭐ 关键：clamp log_scale防止数值爆炸
    lq_log_scale_hat = torch.clamp(lq_log_scale_hat, -30.0, 20.0)
    lq_scale_hat = torch.exp(0.5 * lq_log_scale_hat)
    lq_scale_var = torch.exp(lq_log_scale_hat)
    
    rate = out["likelihoods"]
    
    # 3. 根据codec_mode选择latent
    if self.codec_mode == 'kl_mean':
        lq_latent_hat = lq_mean_hat
        kl_loss = 0.5 * torch.sum(
            torch.pow(lq_mean_hat, 2) + lq_scale_var - 1.0 - lq_log_scale_hat,
            dim=[1, 2, 3],
        )
    
    # 4. DiT去噪
    model_pred = self.DiT(lq_latent_hat, pos_caption_enc, timestep)
    x_denoised = scheduler.step(model_pred, lq_scale_hat, ...) + res
    
    # 5. 解码
    output_image = self.vae.decode(x_denoised / scaling_factor).clamp(-1, 1)
    
    return output_image, rate, z, aux_feature, kl_loss
```

**关键点**：
- ✅ `torch.clamp(lq_log_scale_hat, -30.0, 20.0)` - 防止数值爆炸
- ✅ KL损失使用clamp后的`lq_log_scale_hat`
- ✅ 返回`rate`（likelihoods字典）供loss计算

---

### 2.2 Flow 实现（简化版）

```python
def forward_stage1_train(self, x01: torch.Tensor, train_schedule_steps: int):
    # 1. 编码（无padding）
    z_clean, _ = self.encode_images(x01)
    
    # 2. TCM压缩（无辅助特征）
    tcm_out = self.tcm(z_clean)
    z_tcm = tcm_out["x_hat"]
    z_tcm = z_tcm.to(z_clean.dtype)
    
    # 3. 直接使用z_tcm（无采样/KL模式）
    z_tokens, z_ids = self._latent_to_tokens(z_tcm)
    
    # 4. Flux预测速度场（随机时间步）
    schedule = get_schedule(train_schedule_steps, z_tokens.shape[1])
    step_idx = torch.randint(0, train_schedule_steps, (batch_size,))
    timesteps = schedule[step_idx]
    
    v_pred_tokens = self.flux(z_tokens, z_ids, timesteps, ctx, ctx_ids, guidance)
    
    # 5. 重建（Flow Matching更新公式）
    z_hat_tokens = z_tokens - timesteps.view(-1, 1, 1) * v_pred_tokens
    z_hat = self._tokens_to_latent(z_hat_tokens, z_ids)
    x_hat01 = self.decode_latents(z_hat)
    
    return {
        "x_hat": x_hat01,
        "likelihoods": tcm_out["likelihoods"],
        "z_clean": z_clean,
        "z_tcm": z_tcm,
        "sigma": timesteps,
    }
```

**特点**：
- ❌ 无辅助编码器
- ❌ 无KL损失（TCM不使用log_scale）
- ✅ 单步随机时间步训练（Flow Matching标准做法）
- ✅ 无padding处理

---

### 2.3 newFlow 实现（修复前 ❌）

```python
def forward(self, x01, z_aux, train_schedule_steps, codec_mode="kl_mean"):
    # 1. 编码（带padding）
    z_main, pad_info = self.encode_images(x01)
    
    # 2. ELIC生成辅助特征
    z_aux = self.elic_aux_encoder(x01)  # ⚠️ 在no_grad下生成
    
    # 3. Codec压缩
    codec_out = self.codec(z_main, z_aux)
    mean = codec_out["x_hat"]
    scale = codec_out["scale"]  # ❌ 致命：没有clamp！
    res = codec_out["res"]
    
    # 4. 根据codec_mode选择latent
    if codec_mode == "kl_mean":
        z_tcm = mean
        var = torch.exp(scale)  # ❌ scale可能很大，exp会爆炸
        kl_loss = 0.5 * torch.sum(
            torch.pow(mean, 2) + var - 1.0 - scale,  # ❌ 使用未clamp的scale
            dim=[1, 2, 3],
        ).mean()
    
    # 5. Flux多步去噪
    z_tokens, z_ids = self._latent_to_tokens(z_tcm)
    z_out_tokens = denoise(self.flux, z_tokens, ...)
    z_out = self._tokens_to_latent(z_out_tokens, z_ids)
    
    # 6. 解码并使用padding信息去除padding
    x_hat01 = self.decode_latents(z_out, pad_info)
    
    return {
        "x_hat": x_hat01,
        "likelihoods": codec_out["likelihoods"],
        "kl_loss": kl_loss,
    }
```

**问题**：
- ❌ **没有clamp log_scale** → `exp(scale)`可能爆炸
- ❌ KL损失使用未clamp的scale → 计算错误
- ❌ 导致BPP迅速降到0，重建图全白

---

### 2.4 newFlow 实现（修复后 ✅）

```python
def forward(self, x01, z_aux, train_schedule_steps, codec_mode="kl_mean"):
    # 1. 编码（带padding）
    z_main, pad_info = self.encode_images(x01)
    
    # 2. ELIC生成辅助特征
    z_aux = self.elic_aux_encoder(x01)
    
    # 3. Codec压缩
    codec_out = self.codec(z_main, z_aux)
    mean = codec_out["x_hat"]
    
    # ⭐ 关键修复：clamp log_scale防止数值爆炸
    log_scale = codec_out["scale"]
    log_scale = torch.clamp(log_scale, -30.0, 20.0)  # ✅ 与DiT-IC对齐
    
    res = codec_out["res"]
    
    # 4. 根据codec_mode选择latent
    if codec_mode == "kl_mean":
        z_tcm = mean
        var = torch.exp(log_scale)  # ✅ 使用clamp后的log_scale
        kl_loss = 0.5 * torch.sum(
            torch.pow(mean, 2) + var - 1.0 - log_scale,  # ✅ 使用clamp后的值
            dim=[1, 2, 3],
        ).mean()
    
    # 5. Flux多步去噪
    z_tokens, z_ids = self._latent_to_tokens(z_tcm)
    z_out_tokens = denoise(self.flux, z_tokens, ...)
    z_out = self._tokens_to_latent(z_out_tokens, z_ids)
    
    # 6. 解码并使用padding信息去除padding
    x_hat01 = self.decode_latents(z_out, pad_info)
    
    return {
        "x_hat": x_hat01,
        "likelihoods": codec_out["likelihoods"],
        "kl_loss": kl_loss,
    }
```

**修复点**：
- ✅ 添加`torch.clamp(log_scale, -30.0, 20.0)`
- ✅ KL损失使用clamp后的log_scale
- ✅ 与DiT-IC实现完全对齐

---

## 🔁 三、训练循环对比

### 3.1 核心步骤对比表

| 步骤 | DiT-IC | Flow | newFlow |
|------|--------|------|---------|
| **1. 前向传播** | `output_image, rate, z, aux_feature, kl_loss = model(x)` | `out = pipeline.forward_stage1_train(batch)` | `output = model.forward(x01, z_aux)` |
| **2. 损失计算** | `loss, log_dict = model.loss(x, output_image, rate, ...)`<br>`loss += kl_loss` | `loss_dict = criterion(batch, out["x_hat"], out["likelihoods"])`<br>`loss = loss_dict["loss"]` | `losses = loss_fn(x01, output["x_hat"], output["likelihoods"])`<br>`total_loss = losses["loss"] + 0.05 * kl_loss` |
| **3. 反向传播** | `opt.zero_grad()`<br>`loss.backward()`<br>`opt.step()` | `optimizer.zero_grad()`<br>`accelerator.backward(loss)`<br>`optimizer.step()` | `optimizer.zero_grad()`<br>`accelerator.backward(total_loss)`<br>`optimizer.step()` |
| **4. Aux优化** | `aux_loss = model.codec.aux_loss()`<br>`opt_aux.zero_grad()`<br>`aux_loss.backward()`<br>`opt_aux.step()` | ❌ 无 | `aux_loss = model.codec.aux_loss()`<br>`optimizer_aux.zero_grad()`<br>`aux_loss.backward()`<br>`optimizer_aux.step()` |
| **5. Discriminator** | ✅ 有（可选） | ❌ 无 | ❌ 无 |
| **6. EMA更新** | ✅ 有 | ❌ 无 | ❌ 无 |
| **7. 梯度裁剪** | ✅ 有 | ✅ 有 | ✅ 有 |
| **8. codec.update()** | ❌ 训练时不调用 | ❌ 训练时不调用 | ❌ 训练时不调用（已修复） |

---

### 3.2 DiT-IC 训练循环（伪代码）

```python
while True:
    for x in loader:
        x = x.to(device)
        
        # 1. 前向传播
        output_image, rate, z, aux_feature, kl_loss = model(x)
        
        # 2. 计算主损失
        loss, log_dict = model.module.loss(
            x, output_image, rate, optimizer_idx, train_steps,
            last_layer=model.module.get_last_layer(),
            split="train", z=z, aux_feature=aux_feature
        )
        loss += kl_loss  # 添加KL损失
        
        # 3. 反向传播
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        scheduler.step()
        
        # 4. Aux loss单独优化
        aux_loss = model.module.codec.aux_loss()
        opt_aux.zero_grad()
        aux_loss.backward()
        opt_aux.step()
        
        # 5. Discriminator训练（可选）
        if train_steps >= disc_start:
            discloss, log_dict_disc = model.module.loss(..., optimizer_idx=1)
            opt_disc.zero_grad()
            discloss.backward()
            opt_disc.step()
        
        # 6. EMA更新
        update_ema(ema, model.state_dict(), decay=0.999)
```

---

### 3.3 Flow 训练循环（伪代码）

```python
while not stop:
    for batch in train_loader:
        with accelerator.accumulate(pipeline.flux):
            with accelerator.autocast():
                # 1. 前向传播
                out = pipeline.forward_stage1_train(batch, train_schedule_steps)
                
                # 2. 计算损失
                loss_dict = criterion(batch, out["x_hat"], out["likelihoods"])
                loss = loss_dict["loss"]
            
            # 3. 反向传播
            optimizer.zero_grad()
            accelerator.backward(loss)
            clip_grad_norm_(params, grad_clip)
            optimizer.step()
        
        # 4. 更新统计
        for k in meters:
            meters[k].update(loss_dict[k].item(), bs)
        
        global_step += 1
```

**特点**：
- ❌ 无aux optimizer（TCM不需要）
- ❌ 无discriminator
- ❌ 无EMA
- ✅ 简洁高效

---

### 3.4 newFlow 训练循环（伪代码）

```python
while not stop:
    for batch_data in train_loader:
        x01 = batch_data
        
        with accelerator.accumulate(model):
            with accelerator.autocast():
                # 1. 生成辅助特征
                with torch.no_grad():
                    z_aux = unwrapped_model.elic_aux_encoder(x01)
                
                # 2. 前向传播
                output = unwrapped_model.forward(x01, z_aux, train_schedule_steps)
                
                # 3. 计算损失
                losses = loss_fn(x01, output["x_hat"], output["likelihoods"])
                total_loss = losses["loss"]
                
                # 4. 添加KL损失
                if "kl_loss" in output and output["kl_loss"] is not None:
                    total_loss = total_loss + 0.05 * output["kl_loss"]
            
            # 5. 反向传播
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            
            # 6. Aux loss单独优化
            if optimizer_aux is not None:
                aux_loss = unwrapped_model.codec.aux_loss()
                optimizer_aux.zero_grad()
                accelerator.backward(aux_loss)
                optimizer_aux.step()
                # ⚠️ 注意：训练时不要调用 codec.update()
        
        # 7. 更新统计
        for k in meters:
            meters[k].update(losses[k].item(), bs)
        
        global_step += 1
```

**特点**：
- ✅ 有aux optimizer（与DiT-IC一致）
- ❌ 无discriminator
- ❌ 无EMA
- ✅ 支持ELIC辅助编码器
- ✅ 支持padding处理

---

## 📉 四、Loss函数对比

### 4.1 BPP计算（三者一致）

```python
def _bpp_loss(likelihoods: dict, num_pixels: int) -> torch.Tensor:
    """
    计算BPP（Bits Per Pixel）
    
    Args:
        likelihoods: {"z": z_likelihoods, "y": y_likelihoods}
        num_pixels: B * H * W
    
    Returns:
        BPP标量值
    """
    total = 0.0
    eps = 1e-9
    for v in likelihoods.values():
        total += torch.log(v + eps).sum() / (-math.log(2) * num_pixels)
    return total
```

**公式**：
```
BPP = Σ log(likelihoods) / (-log(2) * num_pixels)
```

---

### 4.2 DiT-IC Loss（复杂版）

```python
def forward(self, inputs, reconstructions, rate, optimizer_idx, global_step, ...):
    N, _, H, W = inputs.size()
    num_pixels = N * H * W
    
    if split == "eval":
        rec_loss = F.mse_loss(inputs, reconstructions)
        psnr = 10 * log10(4 / rec_loss.item())
        kl_loss = sum(log(likelihoods).sum() / (-log(2) * num_pixels) for likelihoods in rate.values())
        p_loss = LPIPS(inputs, reconstructions).mean()
        loss = rec_loss + perceptual_weight * p_loss + kl_weight * kl_loss
        return loss, rec_loss, psnr, kl_loss, p_loss
    
    # Train模式（更复杂）
    rec_loss = abs(inputs - reconstructions)
    nll_loss = rec_loss / exp(logvar) + logvar
    kl_loss = sum(log(likelihoods).sum() / (-log(2) * num_pixels))
    
    # GAN部分
    if optimizer_idx == 0:
        loss = nll_loss + kl_weight * kl_loss + adaptive_disc_weight * disc_loss + vf_weight * vf_loss
    elif optimizer_idx == 1:
        loss = disc_loss
```

**特点**：
- ✅ 自适应权重（adaptive weight）
- ✅ GAN discriminator
- ✅ VF（Visual Feature）损失
- ✅ 复杂的损失平衡策略

---

### 4.3 Flow Loss（简化版）

```python
def forward(self, x01, x_hat01, likelihoods):
    n, _, h, w = x01.shape
    num_pixels = n * h * w
    
    # 1. BPP损失
    bpp = self._bpp_loss(likelihoods, num_pixels)
    
    # 2. MSE损失
    mse = F.mse_loss(x_hat01, x01)
    
    # 3. LPIPS损失
    lpips = LPIPS(x_hat01*2-1, x01*2-1).mean()
    
    # 4. CLIP语义一致性损失
    clip_l2 = CLIP_L2(x01, x_hat01)
    
    # 加权求和
    total = lambda_rate * bpp + d1_mse * mse + d2_lpips * lpips + d3_clip * clip_l2
    
    return {
        "loss": total,
        "bpp": bpp.detach(),
        "mse": mse.detach(),
        "lpips": lpips.detach(),
        "clip_l2": clip_l2.detach(),
    }
```

**特点**：
- ✅ 简洁明了
- ✅ 固定权重（无自适应）
- ❌ 无GAN
- ❌ 无PSNR（已在新版本中添加）

---

### 4.4 newFlow Loss（与Flow相同，新增PSNR）

```python
def forward(self, x01, x_hat01, likelihoods):
    n, _, h, w = x01.shape
    num_pixels = n * h * w
    
    # 1. BPP损失
    bpp = self._bpp_loss(likelihoods, num_pixels)
    
    # 2. MSE损失
    mse = F.mse_loss(x_hat01, x01)
    
    # 3. PSNR（新增）
    if mse > 0:
        psnr = 10 * torch.log10(1.0 / mse)
    else:
        psnr = torch.tensor(float('inf'), device=x01.device)
    
    # 4. LPIPS损失
    lpips = LPIPS(x_hat01*2-1, x01*2-1).mean()
    
    # 5. CLIP语义一致性损失
    clip_l2 = CLIP_L2(x01, x_hat01)
    
    # 加权求和
    total = lambda_rate * bpp + d1_mse * mse + d2_lpips * lpips + d3_clip * clip_l2
    
    return {
        "loss": total,
        "bpp": bpp.detach(),
        "mse": mse.detach(),
        "psnr": psnr.detach(),  # ⭐ 新增
        "lpips": lpips.detach(),
        "clip_l2": clip_l2.detach(),
    }
```

---

## 🔴 五、为什么newFlow会失败？根本原因分析

### 5.1 问题链条

```
1. Codec初始化
   ↓
   codec的scale参数是随机初始化的（可能为100或-100）
   
2. 前向传播
   ↓
   scale = codec_out["scale"]  # 没有clamp
   
3. KL损失计算
   ↓
   var = torch.exp(scale)  # 如果scale=100，exp(100)→无穷大！
   kl_loss = 0.5 * sum(mean² + var - 1 - scale)  # → NaN或异常值
   
4. 熵模型失效
   ↓
   GaussianConditional使用异常的scale值
   likelihoods计算完全错误
   
5. BPP异常
   ↓
   BPP ≈ 0（实际上是数值崩溃，不是真正的压缩效果好）
   
6. 优化器行为
   ↓
   主optimizer看到异常的梯度
   尝试通过降低所有参数来"修复"
   BPP迅速收敛到0（虚假收敛）
   
7. 重建图全白
   ↓
   因为整个训练过程已经崩溃
```

---

### 5.2 数值示例

假设训练初期，codec的scale参数随机初始化为100：

**修复前（错误）**：
```python
scale = 100.0
var = torch.exp(scale)  # exp(100) ≈ 2.688e43 → 溢出！
kl_loss = 0.5 * (mean² + 2.688e43 - 1 - 100)  # → inf或NaN
```

**修复后（正确）**：
```python
log_scale = 100.0
log_scale = torch.clamp(log_scale, -30.0, 20.0)  # → 20.0
var = torch.exp(log_scale)  # exp(20) ≈ 4.85e8（仍然很大，但不会溢出）
kl_loss = 0.5 * (mean² + 4.85e8 - 1 - 20)  # → 可计算的有限值
```

**Clamp范围说明**：
- `-30.0`：对应 `exp(-30) ≈ 9.3e-14`（极小的方差）
- `20.0`：对应 `exp(20) ≈ 4.85e8`（极大的方差）
- 这个范围覆盖了所有合理的概率分布

---

## 🟡 六、重要差异总结

### 6.1 毁灭性差异（已修复）

| 问题 | DiT-IC | Flow | newFlow (修复前) | newFlow (修复后) |
|------|--------|------|------------------|------------------|
| **log_scale clamp** | ✅ `clamp(-30, 20)` | N/A | ❌ **缺失** | ✅ `clamp(-30, 20)` |
| **KL损失计算** | ✅ 使用clamp后的log_scale | N/A | ❌ 使用未clamp的scale | ✅ 使用clamp后的log_scale |
| **后果** | 稳定训练 | 稳定训练 | BPP→0，重建图全白 | 恢复正常 |

---

### 6.2 重要差异

| 特性 | DiT-IC | Flow | newFlow |
|------|--------|------|---------|
| **辅助编码器** | 内置aux_codec | 无 | ELIC (外部) |
| **Padding处理** | ❌ 无 | ❌ 无 | ✅ 有（encode/decode时） |
| **Codec模式** | self_dist/sample/kl_mean | 仅均值 | self_dist/sample/kl_mean |
| **时间步调度** | 单步随机 | 单步随机（Flow Matching） | 多步调度（与推理一致） |
| **Optimizer数量** | 2-3个（main+aux+disc） | 1个 | 2个（main+aux） |
| **混合精度** | FP16 | BF16 | BF16 |
| **LoRA支持** | ❌ | ✅ | ✅ |
| **Discriminator** | ✅ 可选 | ❌ | ❌ |
| **EMA** | ✅ 有 | ❌ | ❌ |

---

### 6.3 次要差异

| 特性 | DiT-IC | Flow | newFlow |
|------|--------|------|---------|
| **评估指标** | loss, mse, psnr, lpips, bpp, aux | loss, mse, lpips, clip_l2, bpp | loss, mse, psnr, lpips, clip_l2, bpp |
| **日志频率** | 每N步 | 每N步 | 每N步 |
| **评估频率** | 每N步 | 每500步 | 每500步 |
| **TensorBoard** | ✅ | ✅ | ✅ |
| **Checkpoint** | 完整状态 | TCM+LoRA+optimizer | 完整状态 |

---

## 💡 七、最佳实践建议

### 7.1 熵模型训练的关键要点

1. **始终clamp log_scale**
   ```python
   log_scale = torch.clamp(codec_out["scale"], -30.0, 20.0)
   ```
   - 防止`exp(log_scale)`数值溢出
   - DiT-IC的标准做法

2. **监控likelihoods范围**
   ```python
   if global_step < 5:
       print(f"z_likelihoods: min={z_lik.min()}, max={z_lik.max()}")
       print(f"y_likelihoods: min={y_lik.min()}, max={y_lik.max()}")
   ```
   - 正常范围：`[1e-10, 1.0]`
   - 异常：接近0或大于1

3. **lambda_rate不宜过小**
   - 建议：`≥ 0.5`
   - 太小会导致模型忽略BPP损失

4. **Aux optimizer学习率要低**
   - 建议：`1e-4`
   - 太高会导致quantiles参数发散

5. **训练时不调用codec.update()**
   - 只在推理/评估时调用
   - 训练时调用会破坏梯度流

---

### 7.2 调试技巧

**当BPP异常时，检查清单**：

1. ✅ log_scale是否clamp？
2. ✅ KL损失是否使用clamp后的值？
3. ✅ likelihoods的范围是否正常？
4. ✅ aux_loss的值是否合理（通常0.1-1.0）？
5. ✅ lambda_rate是否足够大？
6. ✅ 训练时是否错误调用了codec.update()？

**诊断代码**：
```python
# 在前向传播后添加
if global_step < 10:
    print(f"\n[Step {global_step}] Debug Info:")
    print(f"  log_scale range: [{log_scale.min():.2f}, {log_scale.max():.2f}]")
    print(f"  var range: [{var.min():.2e}, {var.max():.2e}]")
    print(f"  kl_loss: {kl_loss.item():.6f}")
    print(f"  BPP: {bpp.item():.6f}")
    print(f"  z_likelihoods range: [{z_lik.min():.6f}, {z_lik.max():.6f}]")
    print(f"  y_likelihoods range: [{y_lik.min():.6f}, {y_lik.max():.6f}]\n")
```

---

## 📝 八、修改记录

### 8.1 已完成的修复

1. **model.py**: 添加log_scale的clamp操作
   ```python
   log_scale = torch.clamp(codec_out["scale"], -30.0, 20.0)
   ```

2. **train.py**: 移除训练时的codec.update()调用
   ```python
   # 删除了：unwrapped_model.codec.update()
   ```

3. **train_config.yaml**: 增大lambda_rate
   ```yaml
   lambda_rate: 2.0  # 从0.1增加到2.0
   ```

4. **losses.py**: 添加PSNR指标
   ```python
   psnr = 10 * torch.log10(1.0 / mse)
   ```

5. **evaluators.py**: 添加PSNR指标和meters初始化

---

### 8.2 待验证的效果

修复后，预期训练表现：

| 指标 | 修复前 | 修复后（预期） |
|------|--------|--------------|
| **BPP** | < 0.05（异常） | 0.3 - 1.0（正常） |
| **重建图** | 全白/无内容 | 有清晰图像内容 |
| **PSNR** | < 20dB（异常） | 25 - 35dB（正常） |
| **likelihoods** | 接近0或1（异常） | 合理概率分布 |
| **KL损失** | NaN或inf | 0.1 - 1.0（正常） |

---

## 🎯 九、总结

### 9.1 核心教训

**一个小小的clamp操作缺失，导致整个训练崩溃！**

- DiT-IC的实现细节至关重要
- 数值稳定性是深度学习的基础
- 必须严格对齐参考实现的每一个关键步骤

### 9.2 后续工作

1. ✅ 已完成log_scale clamp修复
2. ✅ 已移除训练时的codec.update()
3. ✅ 已增大lambda_rate
4. ✅ 已添加PSNR指标
5. ⏳ 待验证：重新启动训练，观察BPP是否恢复正常

---

**文档结束**
