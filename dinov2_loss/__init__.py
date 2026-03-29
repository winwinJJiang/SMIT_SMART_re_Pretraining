"""DINOv3-3D training losses."""

from dinov3_3d.loss.dino_clstoken_loss import DINOLoss
from dinov3_3d.loss.ibot_patch_loss import iBOTPatchLoss
from dinov3_3d.loss.gram_loss import GramLoss
from dinov3_3d.loss.koleo_loss import KoLeoLoss

__all__ = [
    "DINOLoss",
    "iBOTPatchLoss",
    "GramLoss",
    "KoLeoLoss",
]
