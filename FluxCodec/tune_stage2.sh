#!/bin/bash
set -e

# Short-run Stage2 hyperparameter tuning.
# Edit STAGE1_CKPT first, then run:
#   bash tune_stage2.sh

GPU_ID=0
STAGE1_CKPT=./outputs/fluxcodec_stage1/run_20260519_024026/checkpoints/stage1_step_00260000.pt

export CUDA_VISIBLE_DEVICES=$GPU_ID

python tune_stage2.py \
    --gpu "$GPU_ID" \
    --trials 20 \
    --steps_per_trial 2000 \
    --eval_every 1000 \
    --eval_batches 20 \
    --refine_top_k 3 \
    --refine_steps 10000 \
    --refine_eval_every 2000 \
    --refine_eval_batches 50 \
    --stage1_ckpt "$STAGE1_CKPT" \
    --train_root /data2/luosheng/code/flux2/datasets/train \
    --val_root /data2/luosheng/data/Datasets/Kodak \
    --output_dir ./outputs/fluxcodec_stage2_tuning \
    --flux_ckpt /data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors \
    --ae_ckpt /data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors \
    --qwen_ckpt /data2/luosheng/hf_models/hub/Qwen3-4B-FP8 \
    --elic_ckpt /data2/luosheng/code/DiT-IC/checkpoints/elic_official.pth \
    --dinov2_repo /data2/luosheng/hf_models/hub/dinov2 \
    --dinov2_weights /data2/luosheng/hf_models/hub/dinov2/dinov2_vitb14_reg4_pretrain.pth
