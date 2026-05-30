#!/bin/bash
# FluxCodec Stage2 Training (GAN fine-tuning, StableCodec pattern)
# Requires a Stage1 checkpoint as starting point.

# ===== GPU Configuration =====
GPU_ID=0

export CUDA_VISIBLE_DEVICES=$GPU_ID

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    train_stage2.py \
    --train_root /data2/luosheng/code/flux2/datasets/train \
    --val_root /data2/luosheng/data/Datasets/Kodak \
    --output_dir ./outputs/fluxcodec_stage2 \
    --stage1_ckpt ./outputs/fluxcodec_stage1/run_20260519_024026/checkpoints/stage1_step_00260000.pt \
    --model_name flux.2-klein-4b \
    --flux_ckpt /data2/luosheng/hf_models/hub/FLUX.2-klein-4B/flux-2-klein-4b.safetensors \
    --ae_ckpt /data2/luosheng/hf_models/hub/FLUX.2-klein-4B/ae.safetensors \
    --qwen_ckpt /data2/luosheng/hf_models/hub/Qwen3-4B-FP8 \
    --elic_ckpt /data2/luosheng/code/DiT-IC/checkpoints/elic_official.pth \
    --image_size 256 \
    --batch_size 2 \
    --num_workers 4 \
    --max_steps 100000 \
    --lr 5e-5 \
    --lr_disc 2e-5 \
    --grad_clip 1.0 \
    --train_schedule_steps 100 \
    --guidance 1.0 \
    --lambda_rate 0.5 \
    --lambda_l2 1.0 \
    --lambda_lpips 1.0 \
    --lambda_dists 0.2 \
    --lambda_gan 0.1 \
    --gan_loss_type multilevel_sigmoid_s \
    --disc_cv_type dinov2_reg \
    --dinov2_repo /data2/luosheng/hf_models/hub/dinov2 \
    --dinov2_weights /data2/luosheng/hf_models/hub/dinov2/dinov2_vitb14_reg4_pretrain.pth \
    --lr_decay_steps "10000,20000,35000" \
    --lr_decay_values "2e-5,1e-5,1e-6" \
    --lora_rank 32 \
    --lora_alpha 32.0 \
    --use_ae_lora 1 \
    --ae_lora_rank 32 \
    --ae_lora_alpha 32.0 \
    --use_ae_encoder_lora 1 \
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
    --log_every 20 \
    --eval_every 5000 \
    --save_every 10000 \
    --eval_batches 24 \
    --use_tensorboard \
    --save_log_file
