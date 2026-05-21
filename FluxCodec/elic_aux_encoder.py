"""
ELIC Auxiliary Encoder Loader.
"""
import sys
import os
import torch
import torch.nn as nn


def load_elic_encoder(elic_ckpt: str, device: torch.device = torch.device("cpu")) -> nn.Module:
    """
    Load ELIC model and extract its g_a as the auxiliary encoder.
    """
    # Infer DiT-IC directory from checkpoint path
    ckpt_abs = os.path.abspath(elic_ckpt)
    dit_ic_dir = os.path.dirname(os.path.dirname(ckpt_abs))  # checkpoints → DiT-IC
    if dit_ic_dir not in sys.path:
        sys.path.insert(0, dit_ic_dir)

    from ELIC.elic_official import ELIC

    model = ELIC()
    if os.path.exists(elic_ckpt):
        checkpoint = torch.load(elic_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint)
        print(f"[ELIC] Loaded weights from: {elic_ckpt}")
    else:
        print(f"[ELIC] WARNING: Checkpoint not found at {elic_ckpt}, using random init")

    aux_encoder = model.g_a
    aux_encoder.eval()
    aux_encoder.requires_grad_(False)
    aux_encoder.to(device)
    return aux_encoder

