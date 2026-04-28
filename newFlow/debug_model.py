"""
快速诊断脚本：测试newFlow模型的前向传播是否正常

用法：
    python debug_model.py --config config/train_config.yaml
"""
import torch
import yaml
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train_config.yaml")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("newFlow Model Debug Test")
    print("=" * 80)
    
    # 创建dummy输入
    batch_size = 1
    height = 256
    width = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # 创建dummy图像 [0, 1]范围
    x01 = torch.rand(batch_size, 3, height, width).to(device)
    print(f"\nInput image shape: {x01.shape}")
    print(f"Input range: [{x01.min():.4f}, {x01.max():.4f}], mean={x01.mean():.4f}")
    
    # TODO: 初始化模型（需要根据实际配置调整）
    # from model import FlowCompression
    # model = FlowCompression(...)
    
    # TODO: 测试forward
    # output = model.forward(x01, z_aux, global_step=0)
    
    # TODO: 检查输出
    # print(f"Output range: [{output['x_hat'].min():.4f}, {output['x_hat'].max():.4f}]")
    
    print("\n请根据实际的模型初始化代码补充测试逻辑")
    print("关键检查点：")
    print("  1. Codec输出是否有NaN或异常值")
    print("  2. Flux输出是否正常")
    print("  3. 最终重建图像是否在[0,1]范围内")

if __name__ == "__main__":
    main()
