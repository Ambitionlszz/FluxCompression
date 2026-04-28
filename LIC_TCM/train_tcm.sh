#!/bin/bash
# ============================================================
# TCMLatent Training Script
# ============================================================
# Usage:
#   bash train_tcm.sh          # 使用默认配置
#   bash train_tcm.sh 4        # 指定 GPU 数量
# ============================================================

# ---------- 可配置参数 ----------
NUM_GPUS=${1:-1}                                    # GPU 数量 (默认 4, 可通过命令行参数覆盖)
CUDA_DEVICES="1"                              # 使用的 GPU 编号
MASTER_PORT=29500                                    # 分布式通信端口

# ---------- 运行 ----------
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

echo "============================================================"
echo "  TCMLatent Training"
echo "  GPUs: ${NUM_GPUS} (${CUDA_DEVICES})"
echo "  Master Port: ${MASTER_PORT}"
echo "============================================================"

# 进入脚本所在目录
cd "$(dirname "$0")"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --main_process_port=${MASTER_PORT} \
    --mixed_precision=no \
    train_tcm.py

echo ""
echo "Training finished!"
echo "Checkpoints: ./checkpoints/tcm_latent/"
echo "TensorBoard: tensorboard --logdir ./tensorboard_logs/tcm/"
