#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES=${NUM_PROCESSES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
MAIN_PORT=${MAIN_PORT:-29501}
CUDA_DEVICES=${CUDA_DEVICES:-4}

# 支持多个输入目录（用空格分隔）
# 例如: INPUT_DIRS="/data2/luosheng/data/Datasets/Kodak /data2/luosheng/data/Datasets/CLIC"
INPUT_DIRS=${INPUT_DIRS:-/data2/luosheng/data/Datasets/DIV2K_valid_HR}
CHECKPOINT=${CHECKPOINT:-/data2/luosheng/code/flux2/Flow/outputs/stage1/run_20260408_033759/checkpoints/checkpoint_step_00100000.pt}

# Gradient Checkpointing 开关（节省 4-6 GB 显存，但速度慢 20-30%）
# 设置为 "true" 启用，"false" 或不设置则禁用
# 注意：推理时应该禁用 gradient checkpointing 以获得最佳性能
USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-false}

# 从第一个输入目录提取主要数据集名称用于输出目录命名
FIRST_INPUT_DIR=$(echo ${INPUT_DIRS} | awk '{print $1}')
DATASET_NAME=$(basename "${FIRST_INPUT_DIR}")
NUM_DATASETS=$(echo ${INPUT_DIRS} | wc -w)

# 生成带时间戳的临时输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ "${NUM_DATASETS}" -eq 1 ]; then
  TEMP_OUTPUT_DIR="outputs/infer_${DATASET_NAME}_${TIMESTAMP}"
else
  TEMP_OUTPUT_DIR="outputs/infer_multi_datasets_${TIMESTAMP}"
fi
OUTPUT_DIR=${OUTPUT_DIR:-${TEMP_OUTPUT_DIR}}

FLUX_CKPT=${FLUX_CKPT:-/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors}
AE_CKPT=${AE_CKPT:-/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors}
QWEN_CKPT=${QWEN_CKPT:-/data2/luosheng/hf_models/hub/Qwen3-4B-FP8}

if [ -n "${CUDA_DEVICES}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
  echo "[infer_stage1] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "[infer_stage1] Number of datasets: ${NUM_DATASETS}"
echo "[infer_stage1] Datasets: ${INPUT_DIRS}"
echo "[infer_stage1] Output directory: ${OUTPUT_DIR}"
echo "[infer_stage1] Qwen model: ${QWEN_CKPT}"
echo "[infer_stage1] Gradient Checkpointing: ${USE_GRADIENT_CHECKPOINTING}"

# 将输入目录转换为参数列表
INPUT_DIR_ARGS=""
for dir in ${INPUT_DIRS}; do
  INPUT_DIR_ARGS="${INPUT_DIR_ARGS} --input_dirs ${dir}"
done

# 构建 Gradient Checkpointing 参数
GC_ARG=""
if [ "${USE_GRADIENT_CHECKPOINTING}" = "true" ]; then
  GC_ARG="--use_gradient_checkpointing"
  echo "[infer_stage1] ✓ Gradient checkpointing enabled (saves 4-6 GB VRAM)"
else
  echo "[infer_stage1] ✗ Gradient checkpointing disabled (faster inference)"
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  flux_tcm_stage1_infer.py \
  ${INPUT_DIR_ARGS} \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --flux_ckpt "${FLUX_CKPT}" \
  --ae_ckpt "${AE_CKPT}" \
  --qwen_ckpt "${QWEN_CKPT}" \
  --infer_steps 4 \
  --do_entropy_coding \
  ${GC_ARG} \
  "$@"

# 推理完成后，读取平均 BPP 并重命名目录
if [ -f "${OUTPUT_DIR}/metrics_summary.json" ]; then
  # 使用 Python 提取总体平均 BPP 值
  AVG_BPP=$(python3 -c "
import json
with open('${OUTPUT_DIR}/metrics_summary.json', 'r') as f:
    data = json.load(f)
    bpp = data.get('overall_average', {}).get('bpp', 0.0)
    print(f'{bpp:.4f}')
")
  
  # 构建新的目录名称
  if [ "${NUM_DATASETS}" -eq 1 ]; then
    NEW_DIR_NAME="infer_${DATASET_NAME}_bpp${AVG_BPP}"
  else
    NEW_DIR_NAME="infer_multi_datasets_bpp${AVG_BPP}"
  fi
  
  NEW_OUTPUT_DIR="$(dirname "${OUTPUT_DIR}")/${NEW_DIR_NAME}"
  
  # 重命名目录
  if [ "${OUTPUT_DIR}" != "${NEW_OUTPUT_DIR}" ]; then
    mv "${OUTPUT_DIR}" "${NEW_OUTPUT_DIR}"
    echo ""
    echo "=========================================="
    echo "✓ Inference completed successfully!"
    echo "=========================================="
    echo "Datasets:    ${NUM_DATASETS}"
    echo "Average BPP: ${AVG_BPP}"
    echo "Output dir:  ${NEW_OUTPUT_DIR}"
    echo "=========================================="
  else
    echo ""
    echo "=========================================="
    echo "✓ Inference completed!"
    echo "Output dir:  ${OUTPUT_DIR}"
    echo "Average BPP: ${AVG_BPP}"
    echo "=========================================="
  fi
else
  echo ""
  echo "⚠ Warning: metrics_summary.json not found."
  echo "Output dir:  ${OUTPUT_DIR}"
  echo "Skipping directory rename."
fi
