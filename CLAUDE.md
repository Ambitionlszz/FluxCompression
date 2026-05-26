# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FluxCompression is a research codebase for learned image compression built around FLUX.2 Klein. The core idea: use a neural codec (TCM or LatentCodec) to compress FLUX.2 AE latents, then use FLUX.2's flow-matching transformer to denoise/refine the compressed latents before decoding back to pixels.

## Key Architecture

The pipeline has three main stages inside every training/inference forward pass:

1. **AE Encode** — Image → FLUX.2 autoencoder → 128-channel latent at 16× downsample
2. **Codec Compress** — Latent → neural codec (TCM or LatentCodec) → compressed latent. LatentCodec optionally takes ELIC auxiliary features as a second input for better compression.
3. **FLUX Denoise** — Compressed latent → FLUX.2 transformer (guided by a fixed text prompt) predicts velocity field → single-step or multi-step flow-matching reconstruction → AE decode back to pixels

Training is done in two stages:
- **Stage 1** (`Flow/train_stage1.sh`, `FluxCodec/train.sh`): Rate-distortion training with MSE + LPIPS + DISTS + BPP loss. FLUX is frozen, only LoRA adapters on FLUX and the codec are trained.
- **Stage 2** (`FluxCodec/train_stage2.sh`): GAN fine-tuning from a Stage 1 checkpoint, adding a DINOv2-based discriminator and perceptual+adversarial losses.
- **AE Decoder LoRA** (`--use_ae_lora 1 --ae_lora_rank 32`): Injects LoRA into the FLUX AE decoder's Conv2d layers. On by default. Inference auto-detects from checkpoint.
- **EMA** (`--use_ema 1 --ema_decay 0.9999`): Exponential moving average of all trainable parameters. Used during eval; checkpoints save EMA shadows separately.

## Module Map

| Path | Role |
|---|---|
| `src/flux2/` | FLUX.2 model, autoencoder, sampling/denoising, text encoder, utilities. Read-only upstream code — do not modify. |
| `Flow/modules/pipeline.py` | `FlowTCMStage1Pipeline` — wraps FLUX.2 + TCM for Stage 1. Older pipeline used by `Flow/train_stage1.sh` and `Flow/infer_stage1.sh`. |
| `FluxCodec/modules/pipeline.py` | `FluxCodecPipeline` — newer pipeline replacing TCM with LatentCodec + ELIC aux encoder/decoder. Used by `FluxCodec/train.py`, `FluxCodec/inference.py`, `FluxCodec/train_stage2.py`. |
| `FluxCodec/modules/latent_codec.py` | `LatentCodec` — ELIC-style codec with analysis/synthesis transforms, hyperprior, spatial context model with checkerboard autoregressive decoding (4-stage), and LRP refinement. Uses `compressai` for entropy coding. |
| `FluxCodec/modules/lora.py` | LoRA injection for FLUX.2 linear layers (`LoRALinear`, `inject_lora`) and AE decoder conv layers (`LoRAConv2d`, `inject_lora_conv`). State dict helpers handle both types transparently. |
| `FluxCodec/modules/losses.py` | `Stage1Loss` — BPP + MSE + LPIPS + DISTS. |
| `FluxCodec/modules/losses_stage2.py` | `Stage2Loss` — same as Stage 1 plus GAN adversarial loss. |
| `FluxCodec/modules/utils.py` | `AverageMeter`, `EMA` (exponential moving average), `save_checkpoint`, `save_json`, `write_csv`. |
| `FluxCodec/modules/discriminator.py` | Re-exports `Discriminator` from `vision_aided_loss/`. |
| `FluxCodec/vision_aided_loss/` | Discriminator and CV loss modules migrated from StableCodec. |
| `FluxCodec/elic_aux_encoder.py` | Loads ELIC's `g_a` from a DiT-IC checkpoint as a frozen auxiliary encoder. |
| `Flow/model.py` | `FluxTCMModel` — older unified TCM+FLUX model class. |
| `LIC_TCM/` | TCM (Transformer-based Compression Model) implementation and training scripts. |
| `DiT-IC/` | Not in this repo — referenced for ELIC checkpoint loading. |

## Training

All training uses HuggingFace `accelerate` with bf16 mixed precision.

**FluxCodec Stage 1:**
```bash
cd FluxCodec
# Edit train.sh to set checkpoint paths, then:
bash train.sh
```

**FluxCodec Stage 2 (GAN):**
```bash
cd FluxCodec
# Edit train_stage2.sh to set --stage1_ckpt and other paths, then:
bash train_stage2.sh
```

**Resume:** Pass `--resume` to `train.py` or set `RESUME=1` for the Flow pipeline.

## Inference

```bash
cd FluxCodec
# Edit inference.sh to set INPUT_DIRS and CHECKPOINT, then:
bash inference.sh
```

Output includes reconstructions, per-image metrics (PSNR, MS-SSIM, LPIPS, DISTS, BPP), and summary JSON/CSV. The script auto-renames the output directory with the average BPP.

## Key Conventions

- Images: always `[0, 1]` range in pipeline inputs/outputs. The AE encode step converts to `[-1, 1]`.
- Latent padding: FLUX AE requires dimensions divisible by 16. LatentCodec additionally pads latents for hyperprior shape closure. All pipelines handle this via `encode_images`/`decode_latents` with `pad_info` dicts.
- The `LatentCodec`'s aux decoder final conv is zero-initialized so the residual path is identity at the start of training.
- Shell scripts contain machine-specific paths — treat as examples, override with environment variables.
- **FLUX LoRA**: `LoRALinear`, injected via `inject_lora()`. Rank typically 32-64, applied to all FLUX linear layers via regex `.*`. Injected after model creation; moved to model device explicitly.
- **AE Decoder LoRA**: `LoRAConv2d`, injected via `inject_lora_conv()`. Uses `nn.Conv2d` (not raw `nn.Parameter`) for `lora_A`/`lora_B` so autocast handles dtype. Must call `.to(device=...)` after creation. Rank typically 32.
- **EMA**: Created AFTER `accelerator.prepare()` using `accelerator.unwrap_model()` to get the actual model. This ensures EMA shadows are on the correct device. Saved/loaded via `ema.state_dict()` / `ema.load_state_dict()`.
- **Checkpoint keys**: `codec`, `flux_lora`, `ae_decoder_lora`, `ema_codec`, `ema_flux`, `ema_ae_decoder`, `optimizer`, `aux_optimizer`, `discriminator` (Stage 2), `optimizer_disc` (Stage 2).
- **Output directory naming**: `run_{timestamp}_lr{lr}_ld{lambda_rate}_ael/noael_ema/noema`. Stage 2 adds `_gan{lambda_gan}_lrd{lr_disc}`. `noael`/`noema` only appear when the feature is off.
- Inference auto-detects `use_aux_encoder`/`use_aux_decoder` from codec state dict shape, and `use_ae_lora` from checkpoint key presence. No manual flags needed.
- Stage 2 uses `gan_loss_type=multilevel_sigmoid_s` with DINOv2-registered discriminator (`disc_cv_type=dinov2_reg`).

## Environment

Python >=3.10,<3.13. Core deps: torch 2.8.0, torchvision 0.23.0, transformers, safetensors, accelerate, compressai, einops. Install with `pip install -e .` plus `pip install pytorch-msssim pyiqa lpips tqdm pillow tensorboard`.

Lint with ruff (configured in pyproject.toml, line-length 110, ignore E501). No test suite exists — this is research code verified by running training/inference and checking metrics.
