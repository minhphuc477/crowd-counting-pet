from .backbone_convnextv2 import (
    build_backbone_convnextv2,
    build_backbone_timm,
    get_convnextv2_training_defaults,
    get_timm_training_defaults,
    is_modern_timm_backbone,
    resolve_convnextv2_backbone_name,
    resolve_timm_backbone_name,
)
from .backbone_vgg import build_backbone_vgg


def build_backbone(args):
    backbone_name = getattr(args, 'backbone', 'vgg16_bn')
    if is_modern_timm_backbone(backbone_name):
        return build_backbone_timm(args)
    if backbone_name.startswith('vgg'):
        return build_backbone_vgg(args)
    raise ValueError(f'Unsupported backbone: {backbone_name}')

__all__ = [
    'build_backbone',
    'build_backbone_convnextv2',
    'build_backbone_timm',
    'build_backbone_vgg',
    'get_convnextv2_training_defaults',
    'get_timm_training_defaults',
    'is_modern_timm_backbone',
    'resolve_convnextv2_backbone_name',
    'resolve_timm_backbone_name',
]
