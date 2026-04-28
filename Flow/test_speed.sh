#!/bin/bash
# 快速性能测试脚本

echo "======================================================================"
echo "FLUX.2 TCM 推理速度快速测试"
echo "======================================================================"

# 配置
IMAGE="/path/to/test/image.png"  # 修改为你的测试图片路径
CHECKPOINT="/path/to/checkpoint.pth"  # 修改为你的 checkpoint 路径
INPUT_DIR="/path/to/DIV2K_valid_HR"  # 修改为你的测试集路径

echo ""
echo "请先修改脚本中的 IMAGE, CHECKPOINT, INPUT_DIR 路径"
echo ""

# 测试 1: 单图性能分析
echo "【测试 1】单图详细性能分析"
echo "----------------------------------------------------------------------"
python /data2/luosheng/code/flux2/Flow/profile_inference.py \
    --image "$IMAGE" \
    --checkpoint "$CHECKPOINT" \
    --infer_steps 4 \
    --do_entropy_coding \
    --warmup 1 \
    --num_runs 3

echo ""
echo "【测试 2】不使用熵编码（最快模式）"
echo "----------------------------------------------------------------------"
python /data2/luosheng/code/flux2/Flow/profile_inference.py \
    --image "$IMAGE" \
    --checkpoint "$CHECKPOINT" \
    --infer_steps 4 \
    --no_entropy_coding \
    --warmup 1 \
    --num_runs 3

echo ""
echo "【测试 3】减少 infer_steps 到 2"
echo "----------------------------------------------------------------------"
python /data2/luosheng/code/flux2/Flow/profile_inference.py \
    --image "$IMAGE" \
    --checkpoint "$CHECKPOINT" \
    --infer_steps 2 \
    --do_entropy_coding \
    --warmup 1 \
    --num_runs 3

echo ""
echo "======================================================================"
echo "完整数据集测试（使用前 10 张图）"
echo "======================================================================"
echo ""
echo "【测试 4】标准推理（带优化）"
echo "----------------------------------------------------------------------"
python /data2/luosheng/code/flux2/Flow/flux_tcm_stage1_infer.py \
    --input_dirs "$INPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --batch_size 1 \
    --infer_steps 4 \
    --metrics_batch_size 8 \
    --verbose

echo ""
echo "【测试 5】跳过慢速指标（LPIPS/DISTS）"
echo "----------------------------------------------------------------------"
python /data2/luosheng/code/flux2/Flow/flux_tcm_stage1_infer.py \
    --input_dirs "$INPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --batch_size 2 \
    --infer_steps 4 \
    --skip_metrics \
    --metrics_batch_size 8

echo ""
echo "【测试 6】最快模式（无熵编码 + 跳过指标）"
echo "----------------------------------------------------------------------"
python /data2/luosheng/code/flux2/Flow/flux_tcm_stage1_infer.py \
    --input_dirs "$INPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --batch_size 4 \
    --infer_steps 2 \
    --no_entropy_coding \
    --skip_metrics

echo ""
echo "======================================================================"
echo "测试完成！请查看上面的性能数据"
echo "======================================================================"
