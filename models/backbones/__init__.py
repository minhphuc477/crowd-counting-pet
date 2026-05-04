from .backbone_vgg import build_backbone_vgg
from .backbone_timm import (
    build_backbone_convnextv2,
    build_backbone_maxvit,
    build_backbone_swinv2,
    build_backbone_timm,
    get_supported_timm_backbones,
    is_timm_backbone,
    resolve_timm_backbone_name,
)

def build_backbone(args):
    backbone_name = getattr(args, 'backbone', 'vgg16_bn')
    if is_timm_backbone(backbone_name):
        return build_backbone_timm(args)
    if backbone_name.startswith('vgg'):
        return build_backbone_vgg(args)
    raise ValueError(f'Unsupported backbone: {backbone_name}')

__all__ = [
    'build_backbone',
    'build_backbone_convnextv2',
    'build_backbone_maxvit',
    'build_backbone_swinv2',
    'build_backbone_timm',
    'build_backbone_vgg',
    'get_supported_timm_backbones',
    'resolve_timm_backbone_name',
]
