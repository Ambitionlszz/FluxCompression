"""
FluxCodec Stage2 discriminator.

The implementation lives in FluxCodec/vision_aided_loss, migrated from the
StableCodec training dependency so FluxCodec no longer imports code through
StableCodec/src at runtime.
"""

if __package__.startswith("FluxCodec."):
    from ..vision_aided_loss import Discriminator
else:
    from FluxCodec.vision_aided_loss import Discriminator

__all__ = ["Discriminator"]
