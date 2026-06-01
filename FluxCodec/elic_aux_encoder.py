"""
ELIC Auxiliary Encoder Loader.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _normalize_checkpoint(checkpoint):
    """Return a plain state dict and log common checkpoint wrappers."""
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                print(f"[ELIC] Found checkpoint wrapper key '{key}', using nested state dict")
                checkpoint = value
                break

    if isinstance(checkpoint, dict) and any(str(k).startswith("module.") for k in checkpoint.keys()):
        print("[ELIC] Detected 'module.' prefix in checkpoint keys, stripping it")
        checkpoint = {
            str(k)[7:] if str(k).startswith("module.") else k: v
            for k, v in checkpoint.items()
        }
    return checkpoint


def _log_g_a_load_check(model: nn.Module, state_dict: dict, max_examples: int = 5) -> None:
    expected = {f"g_a.{k}": v for k, v in model.g_a.state_dict().items()}
    found = {
        k: v for k, v in state_dict.items()
        if isinstance(k, str) and k.startswith("g_a.")
    }

    if not found:
        print("[ELIC] WARNING: No 'g_a.' keys found in checkpoint; aux encoder is likely random init")
        return

    missing = [k for k in expected.keys() if k not in state_dict]
    unexpected = [k for k in found.keys() if k not in expected]
    shape_mismatch = []
    matched = 0
    for key, expected_tensor in expected.items():
        loaded_tensor = state_dict.get(key)
        if loaded_tensor is None or not torch.is_tensor(loaded_tensor):
            continue
        if tuple(loaded_tensor.shape) != tuple(expected_tensor.shape):
            shape_mismatch.append((key, tuple(loaded_tensor.shape), tuple(expected_tensor.shape)))
        else:
            matched += 1

    total = len(expected)
    if missing or unexpected or shape_mismatch:
        print(
            f"[ELIC] WARNING: g_a checkpoint check found issues "
            f"(matched={matched}/{total}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}, shape_mismatch={len(shape_mismatch)})"
        )
        for key in missing[:max_examples]:
            print(f"[ELIC]   missing g_a key: {key}")
        for key in unexpected[:max_examples]:
            print(f"[ELIC]   unexpected g_a key: {key}")
        for key, loaded_shape, expected_shape in shape_mismatch[:max_examples]:
            print(
                f"[ELIC]   shape mismatch: {key} "
                f"checkpoint={loaded_shape} expected={expected_shape}"
            )
    else:
        print(f"[ELIC] Verified g_a weights: {matched}/{total} tensors matched checkpoint shapes")


def _load_elic_class():
    repo_root = Path(__file__).resolve().parent.parent
    candidates = (
        repo_root / "DiT-IC",
        repo_root / "StableCodec",
        Path("/data2/luosheng/code/DiT-IC"),
        Path("/data2/luosheng/code/flux2/DiT-IC"),
    )

    tried = []
    for root in candidates:
        tried.append(str(root))
        if not root.exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        for name in list(sys.modules):
            if name == "ELIC" or name.startswith("ELIC."):
                del sys.modules[name]
        try:
            from ELIC.elic_official import ELIC  # type: ignore
            print(f"[ELIC] Imported ELIC from {root}")
            return ELIC
        except ModuleNotFoundError:
            try:
                from ELIC.model.elic_official import ELIC  # type: ignore
                print(f"[ELIC] Imported ELIC from {root}")
                return ELIC
            except ModuleNotFoundError:
                continue

    raise ModuleNotFoundError(
        "Could not import ELIC model definition from local repo. "
        "The checkpoint file only contains weights, so the ELIC Python source is still required. "
        f"Tried: {', '.join(tried)}"
    )


def load_elic_encoder(elic_ckpt: str, device: torch.device = torch.device("cpu")) -> nn.Module:
    """
    Load ELIC model and extract its g_a as the auxiliary encoder.
    """
    ELIC = _load_elic_class()

    model = ELIC()
    if os.path.exists(elic_ckpt):
        checkpoint = torch.load(elic_ckpt, map_location="cpu")
        checkpoint = _normalize_checkpoint(checkpoint)
        if isinstance(checkpoint, dict):
            _log_g_a_load_check(model, checkpoint)
        else:
            print(f"[ELIC] WARNING: Checkpoint is not a state dict: {type(checkpoint).__name__}")
        model.load_state_dict(checkpoint)
        print(f"[ELIC] Loaded weights from: {elic_ckpt}")
    else:
        print(f"[ELIC] WARNING: Checkpoint not found at {elic_ckpt}, using random init")

    aux_encoder = model.g_a
    aux_encoder.eval()
    aux_encoder.requires_grad_(False)
    aux_encoder.to(device)
    return aux_encoder
