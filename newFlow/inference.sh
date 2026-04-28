#!/usr/bin/env bash
set -euo pipefail

# 获取脚本所在目录的父目录（即项目根目录 /data2/luosheng/code/flux2）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# 将项目根目录添加到 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "[newflow_infer] PYTHONPATH set to: ${PYTHONPATH}"

NUM_PROCESSES=${NUM_PROCESSES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
MAIN_PORT=${MAIN_PORT:-29503}
CUDA_DEVICES=${CUDA_DEVICES:-4}

INPUT_DIRS=${INPUT_DIRS:-/data2/luosheng/data/Datasets/DIV2K_valid_HR}
CHECKPOINT=${CHECKPOINT:-./outputs/newflow_stage1/checkpoints/latest.pt}

FIRST_INPUT_DIR=$(echo ${INPUT_DIRS} | awk '{print $1}')
DATASET_NAME=$(basename "${FIRST_INPUT_DIR}")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR=${OUTPUT_DIR:-outputs/infer_newflow_${DATASET_NAME}_${TIMESTAMP}}

FLUX_CKPT=${FLUX_CKPT:-/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors}
AE_CKPT=${AE_CKPT:-/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors}
QWEN_CKPT=${QWEN_CKPT:-/data2/luosheng/hf_models/hub/Qwen3-4B-FP8}

if [ -n "${CUDA_DEVICES}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
  echo "[newflow_infer] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "[newflow_infer] Dataset: ${INPUT_DIRS}"
echo "[newflow_infer] Output directory: ${OUTPUT_DIR}"

INPUT_DIR_ARGS=""
for dir in ${INPUT_DIRS}; do
  INPUT_DIR_ARGS="${INPUT_DIR_ARGS} --input_dirs ${dir}"
done

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  inference.py \
  ${INPUT_DIR_ARGS} \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --flux_ckpt "${FLUX_CKPT}" \
  --ae_ckpt "${AE_CKPT}" \
  --qwen_ckpt "${QWEN_CKPT}" \
  "$@"

echo ""
echo "=========================================="
echo "✓ Inference completed!"
echo "Output dir:  ${OUTPUT_DIR}"
echo "=========================================="
