#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES=${NUM_PROCESSES:-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
MAIN_PORT=${MAIN_PORT:-29500}
CUDA_DEVICES=${CUDA_DEVICES:-6}

TRAIN_ROOT=${TRAIN_ROOT:-/data2/luosheng/code/flux2/datasets/train}
VAL_ROOT=${VAL_ROOT:-/data2/luosheng/data/Datasets/clic2024_test_image}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/stage1}
LOG_DIR=${LOG_DIR:-./outputs/logs}

FLUX_CKPT=${FLUX_CKPT:-/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors}
AE_CKPT=${AE_CKPT:-/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors}
QWEN_CKPT=${QWEN_CKPT:-/data2/luosheng/hf_models/hub/Qwen3-4B-FP8}
CLIP_CKPT=${CLIP_CKPT:-/data2/luosheng/hf_models/hub/clip-vit-base-patch32}

if [ -n "${CUDA_DEVICES}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
  echo "[train_stage1] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

if [ "${RESUME:-0}" = "1" ]; then
  RESUME_FLAG="--resume"
else
  RESUME_FLAG=""
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  flux_tcm_stage1_train.py \
  --train_root "${TRAIN_ROOT}" \
  --val_root "${VAL_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --flux_ckpt "${FLUX_CKPT}" \
  --ae_ckpt "${AE_CKPT}" \
  --qwen_ckpt "${QWEN_CKPT}" \
  --clip_ckpt "${CLIP_CKPT}" \
  --lr 1e-5 \
  ${RESUME_FLAG} \
  "$@"
