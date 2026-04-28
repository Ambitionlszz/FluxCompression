#!/bin/bash
# ============================================================
# TCMLatent Inference Script
# ============================================================
# Usage:
#   bash infer_tcm.sh          # 单 GPU 推理
#   bash infer_tcm.sh 2        # 多 GPU 并行推理
# ============================================================

# ---------- 可配置参数 ----------
NUM_GPUS=${1:-1}                                    # GPU 数量 (默认 1)
CUDA_DEVICES="0"                                    # 使用的 GPU 编号
MASTER_PORT=29501                                    # 分布式通信端口 (与训练不同)

# ---------- 运行 ----------
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

echo "============================================================"
echo "  TCMLatent Inference"
echo "  GPUs: ${NUM_GPUS} (${CUDA_DEVICES})"
echo "============================================================"

# 进入脚本所在目录
cd "$(dirname "$0")"

if [ "${NUM_GPUS}" -eq 1 ]; then
    # 单 GPU 直接运行
    python infer_tcm.py
else
    # 多 GPU
    accelerate launch \
        --num_processes=${NUM_GPUS} \
        --main_process_port=${MASTER_PORT} \
        --mixed_precision=no \
        infer_tcm.py
fi

echo ""
echo "Inference finished!"
echo "Results: ./results/"
