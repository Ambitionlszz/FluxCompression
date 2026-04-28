#!/usr/bin/env bash
set -eo pipefail

# FlowCompression/NewFlow 训练启动脚本
# 支持单卡和多卡分布式训练 (基于 Accelerate)
# 使用 YAML 配置文件管理所有参数

# ==================== 环境设置 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "[newflow_train] Working directory: ${SCRIPT_DIR}"

# ==================== 加载配置文件 ====================
CONFIG_FILE="${CONFIG_FILE:-config/train_config.yaml}"

# 检查配置文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "[newflow_train] ERROR: Config file not found: ${CONFIG_FILE}"
  exit 1
fi

echo "[newflow_train] Loading config from: ${CONFIG_FILE}"

# 从 YAML 配置中读取 GPU 设置（优先使用环境变量覆盖）
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  # 尝试从配置文件读取
  CONFIG_GPUS=$(python3 -c "
import yaml
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('gpu', {}).get('cuda_visible_devices', ''))
" 2>/dev/null || echo "")
  
  if [ -n "${CONFIG_GPUS}" ]; then
    export CUDA_VISIBLE_DEVICES="${CONFIG_GPUS}"
    echo "[newflow_train] CUDA_VISIBLE_DEVICES loaded from config: ${CUDA_VISIBLE_DEVICES}"
  else
    echo "[newflow_train] ERROR: CUDA_VISIBLE_DEVICES is not set!"
    echo "[newflow_train] Please set it in config file or environment variable."
    echo "[newflow_train] Example: CUDA_VISIBLE_DEVICES=0,1,2,3 ./train.sh"
    exit 1
  fi
else
  echo "[newflow_train] CUDA_VISIBLE_DEVICES set from environment: ${CUDA_VISIBLE_DEVICES}"
fi

# 自动计算 GPU 数量
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES=${#GPUS[@]}
echo "[newflow_train] Number of GPUs: ${NUM_PROCESSES}"

# 从配置文件读取其他 GPU 相关设置
MIXED_PRECISION=${MIXED_PRECISION:-$(python3 -c "
import yaml
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('gpu', {}).get('mixed_precision', 'bf16'))
" 2>/dev/null || echo "bf16")}

MAIN_PORT=${MAIN_PORT:-$(python3 -c "
import yaml
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('gpu', {}).get('main_port', 29500))
" 2>/dev/null || echo "29500")}

# ==================== 恢复训练逻辑 ====================
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
  RESUME_FLAG="--resume"
  echo "[newflow_train] Resume mode enabled"
fi

# 指定 checkpoint 路径 (可选)
RESUME_PATH_FLAG=""
if [ -n "${RESUME_PATH:-}" ]; then
  RESUME_PATH_FLAG="--resume_path ${RESUME_PATH}"
  echo "[newflow_train] Resume from: ${RESUME_PATH}"
fi

# ==================== 打印配置 ====================
echo "[newflow_train] ========================================"
echo "[newflow_train] Configuration:"
echo "[newflow_train]   Config file:     ${CONFIG_FILE}"
echo "[newflow_train]   GPU devices:     ${CUDA_VISIBLE_DEVICES}"
echo "[newflow_train]   Num processes:   ${NUM_PROCESSES} (auto-detected)"
echo "[newflow_train]   Mixed precision: ${MIXED_PRECISION}"
echo "[newflow_train]   Main port:       ${MAIN_PORT}"
echo "[newflow_train]   Resume:          ${RESUME:-0}"
echo "[newflow_train] ========================================"

# ==================== 启动训练 ====================
echo "[newflow_train] Starting training..."
echo ""

if [ "${NUM_PROCESSES}" -eq 1 ]; then
  # 单卡训练
  python3 train.py \
    --config "${CONFIG_FILE}" \
    ${RESUME_FLAG} \
    ${RESUME_PATH_FLAG} \
    "$@"
else
  # 多卡分布式训练
  accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PORT}" \
accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  train.py \
  --config "${CONFIG_FILE}" \
  ${RESUME_FLAG} \
  ${RESUME_PATH_FLAG} \
  "$@"

echo ""
echo "[newflow_train] Training finished."
