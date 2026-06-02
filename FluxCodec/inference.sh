#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES=${NUM_PROCESSES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
MAIN_PORT=${MAIN_PORT:-29501}
CUDA_DEVICES=${CUDA_DEVICES:-1}

INPUT_DIRS=${INPUT_DIRS:-/data2/luosheng/data/Datasets/Kodak}
INPUT_DIRS=${INPUT_DIRS//,/ }
CHECKPOINT=${CHECKPOINT:-/data2/luosheng/code/flux2/FluxCodec/outputs/fluxcodec_stage1/run_20260601_081705_lr5e-5_ld0.5_noael_ema_trandom_noauxe/checkpoints/stage1_best.pt}

USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-false}
COLOR_FIX=${COLOR_FIX:-false}
USE_EMA_WEIGHTS=${USE_EMA_WEIGHTS:-1}
USE_TILE=${USE_TILE:-false}
TILE_LATENT_SIZE=${TILE_LATENT_SIZE:-64}
TILE_LATENT_OVERLAP=${TILE_LATENT_OVERLAP:-16}

FIRST_INPUT_DIR=$(echo ${INPUT_DIRS} | awk '{print $1}')
DATASET_NAME=$(basename "${FIRST_INPUT_DIR}")
NUM_DATASETS=$(echo ${INPUT_DIRS} | wc -w)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ "${NUM_DATASETS}" -eq 1 ]; then
  TEMP_OUTPUT_DIR="outputs/inference_stage1/infer_${DATASET_NAME}_${TIMESTAMP}"
else
  TEMP_OUTPUT_DIR="outputs/inference_stage1/infer_multi_datasets_${TIMESTAMP}"
fi
OUTPUT_DIR=${OUTPUT_DIR:-${TEMP_OUTPUT_DIR}}

FLUX_CKPT=${FLUX_CKPT:-/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors}
AE_CKPT=${AE_CKPT:-/data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors}
QWEN_CKPT=${QWEN_CKPT:-/data2/luosheng/hf_models/hub/Qwen3-4B-FP8}
ELIC_CKPT=${ELIC_CKPT:-/data2/luosheng/code/DiT-IC/checkpoints/elic_official.pth}

if [ -n "${CUDA_DEVICES}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
  echo "[inference] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

echo "[inference] Number of datasets: ${NUM_DATASETS}"
echo "[inference] Datasets: ${INPUT_DIRS}"
echo "[inference] Output directory: ${OUTPUT_DIR}"
echo "[inference] Checkpoint: ${CHECKPOINT}"

INPUT_DIR_ARGS=()
for dir in ${INPUT_DIRS}; do
  INPUT_DIR_ARGS+=("${dir}")
done

GC_ARG=""
if [ "${USE_GRADIENT_CHECKPOINTING}" = "true" ]; then
  GC_ARG="--use_gradient_checkpointing"
  echo "[inference] ✓ Gradient checkpointing enabled"
else
  echo "[inference] ✗ Gradient checkpointing disabled"
fi

COLOR_FIX_ARG=""
if [ "${COLOR_FIX}" = "true" ]; then
  COLOR_FIX_ARG="--color_fix"
  echo "[inference] ✓ Color fix enabled"
fi

TILE_ARGS=()
if [ "${USE_TILE}" = "true" ]; then
  TILE_ARGS=(
    "--use_tile"
    "--tile_latent_size" "${TILE_LATENT_SIZE}"
    "--tile_latent_overlap" "${TILE_LATENT_OVERLAP}"
  )
  echo "[inference] Tile inference enabled: latent_size=${TILE_LATENT_SIZE}, overlap=${TILE_LATENT_OVERLAP}"
else
  echo "[inference] Tile inference disabled"
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  inference.py \
  --input_dirs "${INPUT_DIR_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --flux_ckpt "${FLUX_CKPT}" \
  --ae_ckpt "${AE_CKPT}" \
  --qwen_ckpt "${QWEN_CKPT}" \
  --elic_ckpt "${ELIC_CKPT}" \
  --infer_steps 4 \
  --do_entropy_coding \
  --use_ema_weights "${USE_EMA_WEIGHTS}" \
  ${GC_ARG} \
  ${COLOR_FIX_ARG} \
  "${TILE_ARGS[@]}" \
  "$@"

if [ -f "${OUTPUT_DIR}/metrics_summary.json" ]; then
  AVG_BPP=$(python3 -c "
import json
with open('${OUTPUT_DIR}/metrics_summary.json', 'r') as f:
    data = json.load(f)
    bpp = data.get('overall_average', {}).get('bpp', 0.0)
    print(f'{bpp:.4f}')
")

  AUX_SUFFIX=$(python3 -c "
import json, os
suffix = ''
if os.path.exists('${OUTPUT_DIR}/infer_config.json'):
    with open('${OUTPUT_DIR}/infer_config.json', 'r') as f:
        data = json.load(f)
        enc = data.get('use_aux_encoder', 0)
        dec = data.get('use_aux_decoder', 0)
        if enc and dec:
            suffix = '_auxED'
        elif enc:
            suffix = '_auxE'
        elif dec:
            suffix = '_auxD'
        else:
            suffix = '_no_aux'
        if data.get('use_ema_weights_applied', 0):
            suffix += '_ema'
print(suffix)
")
  
  if [ "${NUM_DATASETS}" -eq 1 ]; then
    NEW_DIR_NAME="infer_${DATASET_NAME}_${TIMESTAMP}_bpp${AVG_BPP}${AUX_SUFFIX}"
  else
    NEW_DIR_NAME="infer_multi_datasets_${TIMESTAMP}_bpp${AVG_BPP}${AUX_SUFFIX}"
  fi
  
  NEW_OUTPUT_DIR="$(dirname "${OUTPUT_DIR}")/${NEW_DIR_NAME}"
  
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
