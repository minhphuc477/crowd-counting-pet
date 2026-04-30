"""
ConvNeXtV2 backbone modules for PET.
"""
from typing import Dict
import importlib
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import NestedTensor
from ..position_encoding import build_position_encoding


class BackboneFPN(nn.Module):
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
    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        self.backbone = backbone
        self.num_channels = num_channels

        self.feature_info = backbone.feature_info.get_dicts()
        self.stage_channels = [info['num_chs'] for info in self.feature_info]
        self.stage_reductions = [info['reduction'] for info in self.feature_info]
        if len(self.stage_channels) != 4:
            raise ValueError(
                f'ConvNeXtV2 base must expose 4 stages, but got {len(self.stage_channels)} stages.'
            )
        self.output_reduction_to_index = {reduction: idx for idx, reduction in enumerate(self.stage_reductions)}
        missing_reductions = [reduction for reduction in (4, 8) if reduction not in self.output_reduction_to_index]
        if missing_reductions:
            raise ValueError(
                f'ConvNeXtV2 base must expose 4x and 8x features for PET. '
                f'Got reductions {self.stage_reductions}; missing {missing_reductions}.'
            )
        self.fpn = BackboneFPN(self.stage_channels, hidden_size=num_channels, out_size=num_channels, out_kernel=3)

    @staticmethod
    def _to_nchw(feature: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feature.dim() != 4:
            raise ValueError(f'Expected a 4D feature tensor, but got shape {tuple(feature.shape)}')
        if feature.shape[1] == expected_channels:
            return feature
        if feature.shape[-1] == expected_channels:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f'Could not infer channel dimension for feature shape {tuple(feature.shape)} '
            f'with expected channels {expected_channels}.'
        )

    def forward(self, tensor_list: NestedTensor):
        feats = self.backbone(tensor_list.tensors)
        feats = [self._to_nchw(feat, ch) for feat, ch in zip(feats, self.stage_channels)]

        features_fpn = self.fpn(feats)
        features_fpn_4x = features_fpn[self.output_reduction_to_index[4]]
        features_fpn_8x = features_fpn[self.output_reduction_to_index[8]]

        m = tensor_list.mask
        assert m is not None
        mask_4x = F.interpolate(m[None].float(), size=features_fpn_4x.shape[-2:]).to(torch.bool)[0]
        mask_8x = F.interpolate(m[None].float(), size=features_fpn_8x.shape[-2:]).to(torch.bool)[0]

        out: Dict[str, NestedTensor] = {}
        out['4x'] = NestedTensor(features_fpn_4x, mask_4x)
        out['8x'] = NestedTensor(features_fpn_8x, mask_8x)
        return out


class Backbone_ConvNeXtV2(BackboneBase_ConvNeXtV2):
    def __init__(self, pretrained: bool = True):
        timm = importlib.import_module('timm')
        try:
            backbone = timm.create_model(
                'convnextv2_base',
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        except (RuntimeError, OSError) as exc:
            if not pretrained:
                raise
            warnings.warn(
                f'Could not load pretrained weights for convnextv2_base: {exc}. Falling back to random init.',
                RuntimeWarning,
            )
            backbone = timm.create_model(
                'convnextv2_base',
                pretrained=False,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        super().__init__(backbone, num_channels=256)


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
            pos[name] = self.position_embedding(x).to(x.tensors.dtype)
        return out, pos


def build_backbone_convnextv2(args):
    if getattr(args, 'backbone', 'convnextv2_base') != 'convnextv2_base':
        raise ValueError(f'Unsupported ConvNeXtV2 backbone: {args.backbone}')
    position_embedding = build_position_encoding(args)
    backbone = Backbone_ConvNeXtV2(pretrained=True)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model