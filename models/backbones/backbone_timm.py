"""
timm backbone adapters for PET.
"""
from typing import Dict
import importlib
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import NestedTensor
from ..position_encoding import build_position_encoding


TIMM_BACKBONE_ALIASES = {
    'convnext_base_384': 'convnext_base',
    'swinv2_base': 'swinv2_base_window8_256',
    'swinv2_small': 'swinv2_small_window8_256',
    'swinv2_tiny': 'swinv2_tiny_window8_256',
    'maxvit_rmlp_tiny_poly': 'maxvit_rmlp_tiny_rw_256',
    'maxvit_rmlp_tiny': 'maxvit_rmlp_tiny_rw_256',
    'maxvit_tiny': 'maxvit_tiny_rw_256',
    'maxvit_small': 'maxvit_rmlp_small_rw_256',
    'fastvit_tiny': 'fastvit_t8',
    'fastvit_small': 'fastvit_s12',
    'fastvit_medium': 'fastvit_sa24',
    'efficientvit_tiny': 'efficientvit_b0',
    'efficientvit_small': 'efficientvit_b1',
    'efficientnetv2_tiny': 'efficientnetv2_rw_t',
    'efficientnetv2_small': 'efficientnetv2_rw_s',
    'efficientnet_b0': 'tf_efficientnet_b0',
    'efficientnet_b1': 'tf_efficientnet_b1',
    'resnet50': 'resnet50',
    'resnet101': 'resnet101',
    'resnet18': 'resnet18',
    'mobilenetv4_small': 'mobilenetv4_conv_small',
    'mobilenetv4_hybrid': 'mobilenetv4_hybrid_medium',
    'hgnetv2_tiny': 'hgnetv2_b0',
    'hgnetv2_small': 'hgnetv2_b1',
    'pvtv2_b0': 'pvt_v2_b0',
    'pvtv2_b1': 'pvt_v2_b1',
    'edgenext_tiny': 'edgenext_xx_small',
    'repvit_tiny': 'repvit_m0_9',
    'repvit_small': 'repvit_m1',
}

TIMM_BACKBONE_PREFIXES = (
    'convnext_',
    'convnextv2_',
    'swinv2',
    'maxvit',
    'fastvit',
    'efficientvit_',
    'efficientnetv2_',
    'efficientnet_',
    'tf_efficientnet_',
    'tf_efficientnetv2_',
    'resnet',
    'mobilenetv4_',
    'hgnet',
    'hgnetv2_',
    'pvt_v2',
    'edgenext',
    'repvit',
)

SUPPORTED_ABLATION_BACKBONES = (
    'convnext_tiny',
    'convnext_base',
    'convnextv2_tiny',
    'convnextv2_base',
    'swinv2_tiny',
    'swinv2_base',
    'maxvit_tiny',
    'maxvit_small',
    'maxvit_rmlp_tiny',
    'fastvit_tiny',
    'fastvit_small',
    'fastvit_medium',
    'efficientvit_tiny',
    'efficientvit_small',
    'efficientnetv2_tiny',
    'efficientnetv2_small',
    'efficientnet_b0',
    'efficientnet_b1',
    'resnet18',
    'resnet50',
    'resnet101',
    'mobilenetv4_small',
    'mobilenetv4_hybrid',
    'hgnetv2_tiny',
    'hgnetv2_small',
    'pvtv2_b0',
    'pvtv2_b1',
    'edgenext_tiny',
    'edgenext_small',
    'repvit_tiny',
    'repvit_small',
)


def resolve_timm_backbone_name(name):
    return TIMM_BACKBONE_ALIASES.get(name, name)


def is_timm_backbone(name):
    resolved_name = resolve_timm_backbone_name(name)
    return any(resolved_name.startswith(prefix) for prefix in TIMM_BACKBONE_PREFIXES)


def get_supported_timm_backbones():
    return SUPPORTED_ABLATION_BACKBONES


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

        return [
            self.output_convs[0](p2),
            self.output_convs[1](p3),
            self.output_convs[2](p4),
            self.output_convs[3](p5),
        ]


class DirectFeatureAdapter(nn.Module):
    """Project timm's native 4x/8x features into PET's 256-channel contract."""
    def __init__(self, stage_channels, reduction_to_index, out_size=256):
        super().__init__()
        self.index_4x = reduction_to_index[4]
        self.index_8x = reduction_to_index[8]
        self.proj_4x = nn.Conv2d(stage_channels[self.index_4x], out_size, kernel_size=1)
        self.proj_8x = nn.Conv2d(stage_channels[self.index_8x], out_size, kernel_size=1)

    def forward(self, inputs):
        return self.proj_4x(inputs[self.index_4x]), self.proj_8x(inputs[self.index_8x])


class LiteFPNAdapter(nn.Module):
    """Top-down semantic fusion without the heavier 3x3 output convolutions."""
    def __init__(self, stage_channels, reduction_to_index, hidden_size=256):
        super().__init__()
        self.reduction_to_index = reduction_to_index
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(ch, hidden_size, kernel_size=1, stride=1, padding=0)
            for ch in stage_channels
        )

    def forward(self, inputs):
        p = [conv(feat) for conv, feat in zip(self.lateral_convs, inputs)]
        for idx in range(len(p) - 2, -1, -1):
            p[idx] = p[idx] + F.interpolate(p[idx + 1], size=p[idx].shape[-2:], mode='nearest')
        return p[self.reduction_to_index[4]], p[self.reduction_to_index[8]]


class TimmBackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, model_name: str, adapter: str = 'fpn'):
        super().__init__()
        self.backbone = backbone
        self.num_channels = num_channels
        self.model_name = model_name
        self.adapter = adapter

        self.feature_info = backbone.feature_info.get_dicts()
        self.stage_channels = [info['num_chs'] for info in self.feature_info]
        self.stage_reductions = [info['reduction'] for info in self.feature_info]
        if len(self.stage_channels) < 4:
            raise ValueError(
                f'{model_name} must expose at least 4 feature stages, got reductions {self.stage_reductions}.'
            )

        self.output_reduction_to_index = {
            reduction: idx for idx, reduction in enumerate(self.stage_reductions)
        }
        missing_reductions = [reduction for reduction in (4, 8) if reduction not in self.output_reduction_to_index]
        if missing_reductions:
            raise ValueError(
                f'{model_name} must expose PET 4x and 8x features. '
                f'Got reductions {self.stage_reductions}; missing {missing_reductions}.'
            )

        if adapter == 'fpn':
            self.fpn = BackboneFPN(self.stage_channels, hidden_size=num_channels, out_size=num_channels, out_kernel=3)
            self.lite_fpn = None
            self.direct_adapter = None
        elif adapter == 'lite_fpn':
            self.fpn = None
            self.lite_fpn = LiteFPNAdapter(
                self.stage_channels,
                self.output_reduction_to_index,
                hidden_size=num_channels,
            )
            self.direct_adapter = None
        elif adapter == 'direct':
            self.fpn = None
            self.lite_fpn = None
            self.direct_adapter = DirectFeatureAdapter(
                self.stage_channels,
                self.output_reduction_to_index,
                out_size=num_channels,
            )
        else:
            raise ValueError(f'Unsupported timm adapter: {adapter}. Use "lite_fpn", "direct", or "fpn".')

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
        if self.adapter == 'fpn':
            features_fpn = self.fpn(feats)
            features_4x = features_fpn[self.output_reduction_to_index[4]]
            features_8x = features_fpn[self.output_reduction_to_index[8]]
        elif self.adapter == 'lite_fpn':
            features_4x, features_8x = self.lite_fpn(feats)
        else:
            features_4x, features_8x = self.direct_adapter(feats)

        mask = tensor_list.mask
        assert mask is not None
        mask_4x = F.interpolate(mask[None].float(), size=features_4x.shape[-2:]).to(torch.bool)[0]
        mask_8x = F.interpolate(mask[None].float(), size=features_8x.shape[-2:]).to(torch.bool)[0]

        return {
            '4x': NestedTensor(features_4x, mask_4x),
            '8x': NestedTensor(features_8x, mask_8x),
        }


class TimmBackbone(TimmBackboneBase):
    def __init__(self, model_name='convnextv2_base', pretrained=True, allow_pretrained_fallback=False, adapter='fpn'):
        timm = importlib.import_module('timm')
        actual_model_name = resolve_timm_backbone_name(model_name)
        def _create(pretrained_flag):
            try:
                return timm.create_model(
                    actual_model_name,
                    pretrained=pretrained_flag,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                    dynamic_img_size=True,
                )
            except TypeError:
                # Fallback for older timm without dynamic_img_size support
                return timm.create_model(
                    actual_model_name,
                    pretrained=pretrained_flag,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                )

        try:
            backbone = _create(pretrained)
        except (RuntimeError, OSError) as exc:
            if not pretrained:
                raise
            if not allow_pretrained_fallback:
                raise RuntimeError(
                    f'Could not load pretrained weights for {actual_model_name}. '
                    'Install/cache the timm pretrained weights, pass --no_pretrained_backbone, '
                    'or pass --allow_random_backbone_fallback if random initialization is intentional.'
                ) from exc
            warnings.warn(
                f'Could not load pretrained weights for {actual_model_name}: {exc}. Falling back to random init.',
                RuntimeWarning,
            )
            backbone = _create(False)
        super().__init__(backbone, num_channels=256, model_name=actual_model_name, adapter=adapter)


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


def build_backbone_timm(args):
    name = getattr(args, 'backbone', 'convnextv2_base')
    if not is_timm_backbone(name):
        raise ValueError(f'Unsupported timm backbone: {name}')
    adapter = getattr(args, 'timm_adapter', 'fpn')
    position_embedding = build_position_encoding(args)
    backbone = TimmBackbone(
        model_name=name,
        pretrained=not getattr(args, 'no_pretrained_backbone', False),
        allow_pretrained_fallback=getattr(args, 'allow_random_backbone_fallback', False),
        adapter=adapter,
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_backbone_convnextv2(args):
    return build_backbone_timm(args)


def build_backbone_maxvit(args):
    return build_backbone_timm(args)


def build_backbone_swinv2(args):
    return build_backbone_timm(args)
