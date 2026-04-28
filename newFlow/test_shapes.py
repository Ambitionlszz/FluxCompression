import torch
import sys
import os

# 确保可以导入 Block 和 newFlow
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Block.latent_codec import LatentCodec

def test_block_dimensions():
    print("=== Testing Block/latent_codec Dimensions ===")
    
    # 模拟 Flux AE Main Enc 输出 (B, 16, 32, 32) for 512x512 input
    z_main = torch.randn(1, 16, 32, 32).cuda()
    # 模拟 ELIC Aux Enc 输出 (B, 320, 32, 32)
    z_aux = torch.randn(1, 320, 32, 32).cuda()
    
    # 初始化 Codec (ch_emd=16 to match Flux AE)
    codec = LatentCodec(ch_emd=16, channel=320, channel_out=16).cuda()
    codec.eval()
    
    print(f"Input z_main shape: {z_main.shape}")
    print(f"Input z_aux shape: {z_aux.shape}")
    
    with torch.no_grad():
        # 1. 测试 g_a (Analysis Transform)
        y = codec.g_a(z_main, z_aux)
        print(f"g_a output (y) shape: {y.shape}") 
        # Expected: (1, 320, 8, 8) -> 4x downsampling from 32x32
        
        # 2. 测试 h_a (Hyper Analysis)
        z = codec.h_a(y)
        print(f"h_a output (z) shape: {z.shape}")
        # Expected: (1, 160, 2, 2) -> 4x downsampling from 8x8
        
        # 3. 测试 h_s (Hyper Synthesis)
        z_hat = torch.round(z) # 模拟量化
        base = codec.h_s(z_hat)
        print(f"h_s output (base) shape: {base.shape}")
        # Expected: (1, 320, 8, 8) -> 4x upsampling from 2x2
        
        # 4. 测试完整前向传播
        out = codec(z_main, z_aux)
        
        # 检查返回值类型（现在返回的是字典）
        if isinstance(out, dict):
            print(f"Codec mean output shape: {out['mean'].shape}")
            print(f"Codec scale output shape: {out['scale'].shape}")
            print(f"Codec res (AuxDecoder) output shape: {out['res'].shape}")
            
            # 关键检查：g_s 和 AuxDecoder 的输出尺寸必须一致，以便后续残差相加
            assert out['mean'].shape == out['res'].shape, "Mismatch between g_s and AuxDecoder output!"
        else:
            # 兼容旧版本返回列表/元组的情况
            mean, scale, res = out
            print(f"Codec mean output shape: {mean.shape}")
            print(f"Codec scale output shape: {scale.shape}")
            print(f"Codec res (AuxDecoder) output shape: {res.shape}")
            assert mean.shape == res.shape, "Mismatch between g_s and AuxDecoder output!"
            
        print("✅ Dimension Alignment Check Passed!")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU.")
    else:
        test_block_dimensions()
