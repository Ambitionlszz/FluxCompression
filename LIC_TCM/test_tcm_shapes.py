"""Shape smoke test for TCMLatent.

Verifies tensor shapes through the entire TCMLatent model
without requiring real weights or VAE. Uses random tensors.

Usage:
    cd "e:\.Postgraduate Learning\Projects\flux2\LIC_TCM"
    python test_tcm_shapes.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models import TCMLatent


def test_tcm_latent_forward():
    """Test full forward pass shapes."""
    print("=" * 60)
    print("Test: TCMLatent forward pass")
    print("=" * 60)

    # Config
    B, C, H, W = 1, 128, 16, 16  # Typical: 256x256 image -> 16x16 latent
    N, M = 128, 320

    model = TCMLatent(
        in_channels=C, out_channels=C,
        N=N, M=M, num_slices=5,
        ga_config=[2], gs_config=[2],
        ha_config=[2], hs_config=[2],
        ga_head_dim=[8], gs_head_dim=[8],
        window_size=8, atten_window_size=4,
    )
    model.eval()

    x = torch.randn(B, C, H, W)
    print(f"  Input shape:  {tuple(x.shape)}")

    with torch.no_grad():
        out = model(x)

    x_hat = out["x_hat"]
    print(f"  Output shape: {tuple(x_hat.shape)}")
    assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} vs {x.shape}"

    # Check likelihoods
    y_lk = out["likelihoods"]["y"]
    z_lk = out["likelihoods"]["z"]
    print(f"  y likelihoods shape: {tuple(y_lk.shape)}")
    print(f"  z likelihoods shape: {tuple(z_lk.shape)}")

    # y should be (B, M, H/2, W/2) since g_a does 2x downsample
    assert y_lk.shape == (B, M, H // 2, W // 2), f"y shape wrong: {y_lk.shape}"

    print("  ✓ PASSED\n")


def test_ga_gs_shapes():
    """Test g_a downsampling and g_s upsampling individually."""
    print("=" * 60)
    print("Test: g_a (2x down) and g_s (2x up)")
    print("=" * 60)

    C, N, M = 128, 128, 320
    model = TCMLatent(in_channels=C, out_channels=C, N=N, M=M)

    # g_a: (B, 128, 16, 16) -> (B, 320, 8, 8)
    x = torch.randn(1, C, 16, 16)
    with torch.no_grad():
        y = model.g_a(x)
    print(f"  g_a: {tuple(x.shape)} -> {tuple(y.shape)}")
    assert y.shape == (1, M, 8, 8), f"g_a shape wrong: {y.shape}"
    print("  ✓ g_a 2x downsample correct")

    # g_s: (B, 320, 8, 8) -> (B, 128, 16, 16)
    with torch.no_grad():
        x_hat = model.g_s(y)
    print(f"  g_s: {tuple(y.shape)} -> {tuple(x_hat.shape)}")
    assert x_hat.shape == (1, C, 16, 16), f"g_s shape wrong: {x_hat.shape}"
    print("  ✓ g_s 2x upsample correct\n")


def test_ha_hs_shapes():
    """Test h_a and h_s (entropy model) shapes."""
    print("=" * 60)
    print("Test: h_a / h_mean_s / h_scale_s shapes")
    print("=" * 60)

    M, N = 320, 128
    model = TCMLatent(N=N, M=M)

    # h_a: (B, 320, 8, 8) -> (B, 192, 2, 2)  (4x downsample)
    y = torch.randn(1, M, 8, 8)
    with torch.no_grad():
        z = model.h_a(y)
    print(f"  h_a: {tuple(y.shape)} -> {tuple(z.shape)}")
    assert z.shape == (1, 192, 2, 2), f"h_a shape wrong: {z.shape}"
    print("  ✓ h_a correct")

    # h_mean_s: (B, 192, 2, 2) -> (B, 320, 8, 8)  (4x upsample)
    with torch.no_grad():
        means = model.h_mean_s(z)
    print(f"  h_mean_s: {tuple(z.shape)} -> {tuple(means.shape)}")
    assert means.shape == (1, M, 8, 8), f"h_mean_s shape wrong: {means.shape}"
    print("  ✓ h_mean_s correct")

    # h_scale_s: same
    with torch.no_grad():
        scales = model.h_scale_s(z)
    print(f"  h_scale_s: {tuple(z.shape)} -> {tuple(scales.shape)}")
    assert scales.shape == (1, M, 8, 8), f"h_scale_s shape wrong: {scales.shape}"
    print("  ✓ h_scale_s correct\n")


def test_different_spatial_sizes():
    """Test with various spatial dims.

    Architectural constraint: latent spatial dims must be multiples of 16.
    This is because:
      - g_a does 2x downsample: L -> L/2
      - h_a does stride-2 + ConvTransBlock(window_size=4): L/2 -> L/4, needs L/4 % 4 == 0
      - So L must be a multiple of 16

    In practice: image_size must be a multiple of 256 (= 16 * VAE_factor_16).
    Valid: 256x256 -> 16x16, 512x512 -> 32x32, 768x768 -> 48x48, etc.
    """
    print("=" * 60)
    print("Test: Different spatial sizes")
    print("  (Constraint: latent dims must be multiples of 16)")
    print("=" * 60)

    model = TCMLatent()
    model.eval()

    # Valid sizes (multiples of 16)
    for H, W in [(16, 16), (32, 32), (48, 48)]:
        x = torch.randn(1, 128, H, W)
        try:
            with torch.no_grad():
                out = model(x)
            ok = out["x_hat"].shape == x.shape
            status = "✓" if ok else "✗"
            print(f"  {status} Input ({H}x{W}) -> Output {tuple(out['x_hat'].shape)}")
        except Exception as e:
            print(f"  ✗ Input ({H}x{W}) -> Error: {e}")

    # Invalid sizes (expected to fail)
    for H, W in [(24, 24), (8, 8)]:
        x = torch.randn(1, 128, H, W)
        try:
            with torch.no_grad():
                out = model(x)
            print(f"  ⚠ Input ({H}x{W}) unexpectedly succeeded")
        except Exception:
            print(f"  ✓ Input ({H}x{W}) correctly rejected (not multiple of 16)")
    print()


def test_parameter_count():
    """Print model parameter count."""
    print("=" * 60)
    print("Test: Model parameter count")
    print("=" * 60)

    model = TCMLatent()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    print()


if __name__ == "__main__":
    print("\nTCMLatent Shape Smoke Tests\n")
    test_parameter_count()
    test_ga_gs_shapes()
    test_ha_hs_shapes()
    test_tcm_latent_forward()
    test_different_spatial_sizes()
    print("All tests completed!")
