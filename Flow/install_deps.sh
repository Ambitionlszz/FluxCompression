#!/usr/bin/env bash
# 安装推理所需的额外依赖库

set -euo pipefail

echo "=========================================="
echo "Installing additional dependencies for inference"
echo "=========================================="
echo ""

# 检查是否在虚拟环境中
if [ -z "${VIRTUAL_ENV}" ]; then
    echo "⚠ Warning: Not in a virtual environment."
    echo "   It's recommended to use a virtual environment."
    echo ""
fi

echo "Installing pytorch-msssim (for MS-SSIM metric)..."
pip install pytorch-msssim
echo "✓ pytorch-msssim installed"
echo ""

echo "Installing pyiqa (for DISTS metric)..."
pip install pyiqa
echo "✓ pyiqa installed"
echo ""

echo "=========================================="
echo "All dependencies installed successfully!"
echo "=========================================="
echo ""
echo "You can now run inference with full metrics:"
echo "  bash infer_stage1.sh"
echo ""
