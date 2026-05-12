#!/bin/bash
# FluxCodec Stage1 训练启动脚本

# ===== 设置使用的 GPU =====
GPU_ID=1        # 单卡: 0, 1, 2, ... 多卡: 0,1

export CUDA_VISIBLE_DEVICES=$GPU_ID

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    train.py \
    --train_root /data2/luosheng/code/flux2/datasets/train \
    --val_root /data2/luosheng/data/Datasets/Kodak \
    --output_dir ./outputs/fluxcodec_stage1 \
    --model_name flux.2-klein-4b \
    --flux_ckpt /data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors \
    --ae_ckpt /data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors \
    --qwen_ckpt /data2/luosheng/hf_models/hub/Qwen3-4B-FP8 \
    --clip_ckpt /data2/luosheng/hf_models/hub/clip-vit-base-patch32 \
    --elic_ckpt /data2/luosheng/code/DiT-IC/checkpoints/elic_official.pth \
    --image_size 256 \
    --batch_size 2 \
    --num_workers 4 \
    --max_steps 400000 \
    --lr 1e-5 \
    --grad_clip 1.0 \
    --train_schedule_steps 1000 \
    --guidance 1.0 \
    --lambda_rate 0.3 \
    --d1_mse 2.0 \
    --d2_lpips 1.0 \
    --d3_clip 0.1 \
    --lora_rank 32 \
    --lora_alpha 32.0 \
    --codec_ch_emd 128 \
    --codec_channel 320 \
    --codec_channel_out 128 \
    --codec_num_slices 5 \
    --use_aux_encoder 0 \
    --use_aux_decoder 0 \
    --log_every 20 \
    --eval_every 2000 \
    --save_every 20000 \
    --eval_batches 50 \
    --use_tensorboard \
    --save_log_file
