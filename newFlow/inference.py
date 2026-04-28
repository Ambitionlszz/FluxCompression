import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import argparse

from newFlow.model import FlowCompression

def load_image(image_path, device):
    """加载并预处理图像"""
    img = Image.open(image_path).convert("RGB")
    # 转换为 Tensor 并归一化到 [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

def save_image(tensor, path):
    """保存重建后的图像"""
    tensor = tensor.squeeze(0).clamp(0, 1).cpu()
    transforms.ToPILImage()(tensor).save(path)

def main():
    parser = argparse.ArgumentParser(description="Flux2 Flow Compression Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="reconstructed.png", help="Path to output image")
    parser.add_argument("--flux_ckpt", type=str, default="/path/to/flux.safetensors", help="Path to Flux checkpoint")
    parser.add_argument("--ae_ckpt", type=str, default="/path/to/ae.safetensors", help="Path to AE checkpoint")
    parser.add_argument("--qwen_ckpt", type=str, default="/path/to/qwen", help="Path to Qwen checkpoint")
    parser.add_argument("--elic_ckpt", type=str, default="/data2/luosheng/code/DiT-IC/checkpoints/elic_official.pth", help="Path to ELIC checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    print("Initializing FlowCompression model...")
    model = FlowCompression(
        model_name="flux.2-klein-4b",
        flux_ckpt=args.flux_ckpt,
        ae_ckpt=args.ae_ckpt,
        qwen_ckpt=args.qwen_ckpt,
        codec_config={"ch_emd": 128, "channel": 320, "channel_out": 128},
        device=device,
        elic_ckpt=args.elic_ckpt,  # ELIC辅助编码器权重路径
    )
    model.eval()

    # 加载图像
    print(f"Loading image from {args.input}...")
    x01 = load_image(args.input, device)

    print("Running compression and reconstruction...")
    with torch.no_grad():
        # 执行推理（ELIC编码器已集成到模型内部）
        output = model.forward_stage1_infer(
            x01, 
            infer_steps=4, 
            do_entropy_coding=True
        )
        
        # 获取重建图像
        x_hat = output["x_hat"]
        
    save_image(x_hat, args.output)
    print(f"Reconstruction saved to {args.output}")

if __name__ == "__main__":
    main()
