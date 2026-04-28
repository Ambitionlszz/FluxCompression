#!/usr/bin/env bash
# 示例脚本：演示多数据集推理和新指标功能

set -euo pipefail

echo "=========================================="
echo "Flux2/Flow 推理示例 - 多数据集 + 新指标"
echo "=========================================="
echo ""

# 配置 checkpoint 路径
export CHECKPOINT=/data2/luosheng/code/flux2/Flow/outputs/stage1/run_20260408_033759/checkpoints/checkpoint_step_00100000.pt
export CUDA_DEVICES=0

# 示例 1: 单数据集推理（Kodak）
echo "【示例 1】单数据集推理 - Kodak"
echo "----------------------------------------"
export INPUT_DIRS=/data2/luosheng/data/Datasets/Kodak
bash infer_stage1.sh --infer_steps 4

echo ""
echo "预期输出目录: outputs/infer_Kodak_bppXXXXX/"
echo "包含指标: PSNR, MS-SSIM, LPIPS, DISTS, BPP"
echo ""

# 示例 2: 多数据集推理
if [ -d "/data2/luosheng/data/Datasets/CLIC" ] && [ -d "/data2/luosheng/data/Datasets/Tecnick" ]; then
  echo "【示例 2】多数据集推理 - Kodak + CLIC + Tecnick"
  echo "----------------------------------------"
  export INPUT_DIRS="/data2/luosheng/data/Datasets/Kodak /data2/luosheng/data/Datasets/CLIC /data2/luosheng/data/Datasets/Tecnick"
  bash infer_stage1.sh --infer_steps 4
  
  echo ""
  echo "预期输出目录: outputs/infer_multi_datasets_bppXXXXX/"
  echo "包含每个数据集的独立统计和总体平均"
  echo ""
fi

# 示例 3: 禁用熵编码（仅测试质量，速度更快）
echo "【示例 3】禁用熵编码 - 快速质量测试"
echo "----------------------------------------"
export INPUT_DIRS=/data2/luosheng/data/Datasets/Kodak
bash infer_stage1.sh --infer_steps 4 --no_entropy_coding

echo ""
echo "预期输出目录: outputs/infer_Kodak_bpp0.0000/"
echo "BPP 将为 0，但其他指标正常计算"
echo ""

echo "=========================================="
echo "所有示例完成！"
echo "=========================================="
echo ""
echo "查看输出目录:"
ls -lh outputs/ | grep "infer_"
echo ""
echo "提示: XXXXX 代表实际的 BPP 值（例如 0.1234）"
echo ""
echo "查看指标文件:"
echo "  - metrics_summary.json (总体汇总)"
echo "  - metrics_all_datasets.csv (所有数据)"
echo "  - {dataset_name}/summary.json (单个数据集)"
