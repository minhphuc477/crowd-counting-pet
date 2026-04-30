from .backbone_convnextv2 import build_backbone_convnextv2
from .backbone_vgg import build_backbone_vgg

def build_backbone(args):
    backbone_name = getattr(args, 'backbone', 'vgg16_bn')
    if backbone_name == 'convnextv2_base':
        return build_backbone_convnextv2(args)
    if backbone_name.startswith('vgg'):
        return build_backbone_vgg(args)
    raise ValueError(f'Unsupported backbone: {backbone_name}')

__all__ = [
    'build_backbone',
    'build_backbone_convnextv2',
    'build_backbone_vgg',
]