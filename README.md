# FluxCompression

FluxCompression is a research codebase for image compression experiments built around
FLUX.2 Klein, learned latent compression, LoRA adaptation, and GAN/perceptual
fine-tuning. The repository keeps the original FLUX.2 inference code under `src/`,
then adds compression-oriented training and inference pipelines in `Flow/`,
`FluxCodec/`, `LIC_TCM/`, `DiT-IC/`, and `StableCodec/`.

The current main workflow is:

1. Train a Stage 1 rate-distortion model with FLUX.2 Klein + TCM/latent codec.
2. Run compression inference and export reconstructions plus metrics.
3. Optionally fine-tune Stage 2 with perceptual and adversarial losses.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/flux2/` | FLUX.2 model, autoencoder, sampling, and text encoder code. |
| `Flow/` | Stage 1 FLUX + TCM training/inference pipeline. |
| `FluxCodec/` | Newer FluxCodec training, inference, Stage 2 GAN fine-tuning, and tuning scripts. |
| `LIC_TCM/` | TCM and learned image compression modules. |
| `DiT-IC/` | DiT-IC reference code and ELIC components/checkpoint utilities. |
| `StableCodec/` | StableCodec reference implementation used by the GAN fine-tuning design. |
| `docs/` | Project notes, including Stage 2 GAN training design. |
| `scripts/` | Upstream FLUX.2 interactive CLI. |

## Environment

The project is Python-based and expects CUDA-capable PyTorch for practical use.
The checked-in `pyproject.toml` targets Python `>=3.10,<3.13`.

Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Extra packages used by some metrics or submodules may be needed depending on the
script you run:

```bash
pip install pytorch-msssim pyiqa lpips tqdm pillow accelerate tensorboard
```

For CUDA builds, install PyTorch from the wheel index that matches your machine.
The project lock/config currently uses PyTorch `2.8.0` and torchvision `0.23.0`.

## Required Checkpoints

Most workflows assume local model weights. Update the paths in the shell scripts
or pass the corresponding command-line arguments.

Common paths used by the scripts:

| Argument / variable | Meaning |
| --- | --- |
| `FLUX_CKPT`, `--flux_ckpt` | FLUX.2 Klein transformer weights, for example `flux-2-klein-4b.safetensors`. |
| `AE_CKPT`, `--ae_ckpt` | FLUX.2 autoencoder weights, for example `ae.safetensors`. |
| `QWEN_CKPT`, `--qwen_ckpt` | Local Qwen text model used by the pipeline. |
| `CLIP_CKPT`, `--clip_ckpt` | CLIP checkpoint used by Stage 1 training losses. |
| `--elic_ckpt` | ELIC checkpoint used by FluxCodec/Stage 2 components. |
| `--stage1_ckpt` | A trained Stage 1 checkpoint used to initialize Stage 2. |
| `--dinov2_weights` | DINOv2 weights for the Stage 2 discriminator. |

## Stage 1 Training

Stage 1 trains the compression pipeline with rate-distortion and perceptual
objectives. The main launcher is `Flow/train_stage1.sh`.

```bash
cd Flow

export CUDA_DEVICES=0
export NUM_PROCESSES=1
export TRAIN_ROOT=/path/to/train/images
export VAL_ROOT=/path/to/val/images
export OUTPUT_DIR=./outputs/stage1

export FLUX_CKPT=/path/to/FLUX.2-klein-4B/flux-2-klein-4b.safetensors
export AE_CKPT=/path/to/FLUX.2-klein-4B/ae.safetensors
export QWEN_CKPT=/path/to/Qwen3-4B-FP8
export CLIP_CKPT=/path/to/clip-vit-base-patch32

bash train_stage1.sh
```

To resume from the latest run under the configured output directory:

```bash
RESUME=1 bash train_stage1.sh
```

The launcher writes run folders under `OUTPUT_DIR`, including checkpoints,
logs, and the saved training configuration.

## Stage 1 Inference

Use `Flow/infer_stage1.sh` for dataset inference. It supports one or more input
directories and writes reconstructions, per-image metrics, and summary JSON/CSV
files.

```bash
cd Flow

export CUDA_DEVICES=0
export NUM_PROCESSES=1
export INPUT_DIRS="/path/to/Kodak /path/to/CLIC"
export CHECKPOINT=/path/to/stage1/checkpoint.pt

export FLUX_CKPT=/path/to/FLUX.2-klein-4B/flux-2-klein-4b.safetensors
export AE_CKPT=/path/to/FLUX.2-klein-4B/ae.safetensors
export QWEN_CKPT=/path/to/Qwen3-4B-FP8

bash infer_stage1.sh
```

Useful inference flags:

```bash
bash infer_stage1.sh --batch_size 1 --infer_steps 4 --skip_metrics
bash infer_stage1.sh --no_entropy_coding
```

Typical output structure:

```text
outputs/infer_Kodak_bpp0.1234/
  recon/
  metrics_all_datasets.csv
  metrics_summary.json
  infer_config.json
```

Metrics include PSNR, MS-SSIM, LPIPS, DISTS, and BPP when the required metric
packages and entropy coding path are enabled.

## FluxCodec Workflow

`FluxCodec/` contains the newer training and inference scripts. These scripts
follow the same checkpoint/path assumptions, but keep the code and outputs under
the `FluxCodec` module.

Training:

```bash
cd FluxCodec
bash train.sh
```

Inference:

```bash
cd FluxCodec
bash inference.sh
```

Before launching, edit the dataset, checkpoint, and model paths in the shell
scripts or pass equivalent CLI arguments directly to `train.py` / `inference.py`.

## Stage 2 GAN Fine-Tuning

Stage 2 starts from a Stage 1 checkpoint and adds GAN/perceptual fine-tuning,
borrowing the discriminator pattern from StableCodec. The main launcher is
`FluxCodec/train_stage2.sh`.

```bash
cd FluxCodec

# Edit --stage1_ckpt, dataset paths, FLUX/AE/Qwen paths, ELIC path, and DINOv2 paths first.
bash train_stage2.sh
```

For implementation notes and loss design, see:

- `docs/stage2_gan_training.md`
- `FluxCodec/train_stage2.py`
- `FluxCodec/modules/losses_stage2.py`

There is also a short-run tuning helper:

```bash
cd FluxCodec
bash tune_stage2.sh
```

It launches multiple shorter trials and writes a ranked summary plus a suggested
full-training script.

## Upstream FLUX.2 CLI

The original FLUX.2 CLI is still available for generation/editing experiments:

```bash
PYTHONPATH=src python scripts/cli.py
```

If the environment variables for FLUX.2 weights are not set, the upstream code
may attempt to download weights automatically. For offline or cluster runs, set
local model paths explicitly.

## Notes

- Many shell scripts currently contain machine-specific default paths. Treat them
  as examples and override with environment variables or edit them for your
  machine.
- Large model weights and trained checkpoints are not expected to be committed.
- Some older markdown files and comments may show mojibake/encoding artifacts;
  the root README is the clean entry point for the current workflow.
- This repository combines upstream and experimental research code. Check
  individual model licenses before using weights or outputs commercially.

## License

See `LICENSE.md` and the license files of the upstream components and model
weights you use. FLUX.2, StableCodec, DiT-IC, DINOv2, and any downloaded
checkpoints may have separate terms.
