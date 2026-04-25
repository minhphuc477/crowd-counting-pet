"""
Backbone modules for ConvNeXt V2.
"""
from typing import Dict
import importlib
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import NestedTensor
from ..position_encoding import build_position_encoding


CONVNEXTV2_VARIANTS = (
    'convnextv2_nano',
    'convnextv2_small',
    'convnextv2_base',
    'convnextv2_large',
)


def resolve_convnextv2_backbone_name(name: str) -> str:
    """
    Resolve the ConvNeXt V2 backbone name.
    """
    if name != 'auto':
        if name not in CONVNEXTV2_VARIANTS:
            raise ValueError(f'Unsupported ConvNeXt V2 backbone: {name}')
        return name

    if not torch.cuda.is_available():
        return 'convnextv2_nano'

    total_memory_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
    if total_memory_gb >= 24:
        return 'convnextv2_large'
    if total_memory_gb >= 14:
        return 'convnextv2_base'
    if total_memory_gb >= 8:
        return 'convnextv2_small'
    return 'convnextv2_nano'


def get_convnextv2_training_defaults(backbone_name: str):
    """
    Conservative batch-size and learning-rate defaults for a resolved ConvNeXt V2 backbone.
    """
    if backbone_name == 'convnextv2_large':
        return 1, 1.25e-5, 1.25e-6
    if backbone_name == 'convnextv2_base':
        return 2, 2.5e-5, 2.5e-6
    if backbone_name == 'convnextv2_small':
        return 4, 5e-5, 5e-6
    return 8, 1e-4, 1e-5


class ConvNeXtV2FPN(nn.Module):
    def __init__(self, stage_channels, hidden_size=256, out_size=256, out_kernel=3):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(ch, hidden_size, kernel_size=1, stride=1, padding=0)
            for ch in stage_channels
        )
        self.output_convs = nn.ModuleList(
            nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel // 2)
            for _ in stage_channels
        )

    def forward(self, inputs):
        c2, c3, c4, c5 = inputs

        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')

        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        p5 = self.output_convs[3](p5)
        return [p2, p3, p4, p5]


class BackboneBase_ConvNeXtV2(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        self.backbone = backbone
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

        stage_channels = [info['num_chs'] for info in backbone.feature_info.get_dicts()]
        if len(stage_channels) != 4:
            raise ValueError(
                f'ConvNeXt V2 backbone must expose 4 stages, but got {len(stage_channels)} stages.'
            )
        self.fpn = ConvNeXtV2FPN(stage_channels, hidden_size=num_channels, out_size=num_channels, out_kernel=3)

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        feats = self.backbone(xs)

        # FPN-like fusion keeps the 4x/8x outputs compatible with the existing PET pipeline.
        features_fpn = self.fpn(feats)
        features_fpn_4x = features_fpn[0]
        features_fpn_8x = features_fpn[1]

        m = tensor_list.mask
        assert m is not None
        mask_4x = F.interpolate(m[None].float(), size=features_fpn_4x.shape[-2:]).to(torch.bool)[0]
        mask_8x = F.interpolate(m[None].float(), size=features_fpn_8x.shape[-2:]).to(torch.bool)[0]

        out: Dict[str, NestedTensor] = {}
        out['4x'] = NestedTensor(features_fpn_4x, mask_4x)
        out['8x'] = NestedTensor(features_fpn_8x, mask_8x)
        return out


class Backbone_ConvNeXtV2(BackboneBase_ConvNeXtV2):
    """
    ConvNeXt V2 backbone
    """

    def __init__(self, name: str, return_interm_layers: bool, pretrained: bool = True):
        if not name.startswith('convnextv2_'):
            raise ValueError(f'Unsupported ConvNeXt V2 backbone: {name}')

        timm = importlib.import_module('timm')

        try:
            backbone = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        except (RuntimeError, OSError) as exc:
            if not pretrained:
                raise
            warnings.warn(
                f'Could not load pretrained weights for {name}: {exc}. Falling back to random init.',
                RuntimeWarning,
            )
            backbone = timm.create_model(
                name,
                pretrained=False,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )

        num_channels = 256
        super().__init__(backbone, num_channels, return_interm_layers)


class Joiner(nn.Module):
    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list)
        out: Dict[str, NestedTensor] = {}
        pos = {}
        for name, x in xs.items():
            out[name] = x
            # Mã hóa vị trí không gian nhằm giữ lại cấu trúc hình học khi đưa đặc trưng vào transformer.
            pos[name] = self.position_embedding(x).to(x.tensors.dtype)
        return out, pos


def build_backbone_convnextv2(args):
    position_embedding = build_position_encoding(args)
    backbone_name = resolve_convnextv2_backbone_name(args.backbone)
    args.backbone = backbone_name
    backbone = Backbone_ConvNeXtV2(backbone_name, True)
    model = Joiner(backbone, position_embedding)
    return model


if __name__ == '__main__':
    Backbone_ConvNeXtV2('convnextv2_nano', True)