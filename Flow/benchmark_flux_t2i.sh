#!/usr/bin/env bash
set -euo pipefail

# FLUX.2 文生图速度测试脚本

CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/speed_test_t2i}

# 模型路径
FLUX_CKPT=${FLUX_CKPT:-/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors}
AE_CKPT=${AE_CKPT:-/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors}
QWEN_CKPT=${QWEN_CKPT:-/data2/luosheng/hf_models/hub/Qwen3-4B-FP8}

# 测试配置
RESOLUTIONS=${RESOLUTIONS:-2048}
PROMPT="${PROMPT:-A beautiful sunset over the ocean, photorealistic, detailed}"
NUM_STEPS=${NUM_STEPS:-4}
WARMUP_RUNS=${WARMUP_RUNS:-5}
TEST_RUNS=${TEST_RUNS:-10}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

echo "=========================================="
echo "FLUX.2 Text-to-Image Speed Test"
echo "=========================================="
echo "Resolutions: ${RESOLUTIONS}"
echo "Prompt: ${PROMPT}"
echo "Inference steps: ${NUM_STEPS}"
echo "Warmup runs: ${WARMUP_RUNS}"
echo "Test runs: ${TEST_RUNS}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=========================================="

cd "$(dirname "$0")"

python benchmark_flux_t2i.py \
  --flux_ckpt "${FLUX_CKPT}" \
  --ae_ckpt "${AE_CKPT}" \
  --qwen_ckpt "${QWEN_CKPT}" \
  --resolutions "${RESOLUTIONS}" \
  --prompt "${PROMPT}" \
  --num_inference_steps "${NUM_STEPS}" \
  --warmup_runs "${WARMUP_RUNS}" \
  --test_runs "${TEST_RUNS}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_images \
  "$@"
