#!/bin/bash
# FluxCodec Stage1 Training Script

# ===== GPU Configuration =====
GPU_ID=0        # Single GPU: 0, 1, 2... Multi-GPU: 0,1

export CUDA_VISIBLE_DEVICES=$GPU_ID
CLIP_CKPT=${CLIP_CKPT:-/data2/luosheng/hf_models/hub/clip-vit-base-patch32}

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
    --elic_ckpt /data2/luosheng/hf_models/hub/elic_official.pth \
    --image_size 256 \
    --batch_size 2 \
    --num_workers 4 \
    --max_steps 400000 \
    --lr 5e-5 \
    --grad_clip 1.0 \
    --train_schedule_steps 100 \
    --train_timestep_mode infer_schedule \
    --train_infer_steps 4 \
    --fixed_timestep_index 0 \
    --guidance 1.0 \
    --lambda_rate 0.5 \
    --d1_mse 2.0 \
    --d2_lpips 1.0 \
    --d3_metric dists \
    --d3_dists 0.1 \
    --d3_clip 0.1 \
    --clip_ckpt "$CLIP_CKPT" \
    --lora_rank 64 \
    --lora_alpha 64.0 \
    --use_ae_lora 1 \
    --ae_lora_rank 32 \
    --ae_lora_alpha 32.0 \
    --use_ae_encoder_lora 0 \
    --ae_encoder_lora_rank 32 \
    --ae_encoder_lora_alpha 32.0 \
    --use_ema 1 \
    --ema_decay 0.9999 \
    --codec_ch_emd 128 \
    --codec_channel 320 \
    --codec_channel_out 128 \
    --codec_num_slices 5 \
    --use_aux_encoder 1 \
    --use_aux_decoder 1 \
    --aux_decoder_zero_init 0 \
    --elic_proj_channels 64 \
    --log_every 50 \
    --eval_every 5000 \
    --save_every 20000 \
    --eval_batches 24 \
    --use_tensorboard \
    --save_log_file \
