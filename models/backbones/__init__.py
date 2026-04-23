from .backbone_convnextv2 import build_backbone_convnextv2
from .backbone_vgg import build_backbone_vgg

build_backbone = build_backbone_convnextv2

__all__ = [
    'build_backbone',
    'build_backbone_convnextv2',
    'build_backbone_vgg',
]
