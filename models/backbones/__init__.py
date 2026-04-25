from .backbone_convnextv2 import (
    build_backbone_convnextv2,
    get_convnextv2_training_defaults,
    resolve_convnextv2_backbone_name,
)
from .backbone_vgg import build_backbone_vgg

build_backbone = build_backbone_convnextv2

__all__ = [
    'build_backbone',
    'build_backbone_convnextv2',
    'build_backbone_vgg',
    'get_convnextv2_training_defaults',
    'resolve_convnextv2_backbone_name',
]
