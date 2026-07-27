"""EBC-supervised routing components for Point quEry Transformer.

The router is intentionally independent of PET's point decoder.  It predicts
blockwise count distributions and uses a separately supervised route logit to
decide whether a quadtree cell should enter PET's dense branch.  The model
therefore retains point-query localization and does not substitute a scalar
or blockwise count for PET inference.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def _group_norm(channels):
    groups = min(32, int(channels))
    while groups > 1 and int(channels) % groups:
        groups -= 1
    return nn.GroupNorm(groups, int(channels))


class EBCQuadtreeRouter(nn.Module):
    """Enhanced-blockwise-counting router for PET quadtree cells.

    ``zero_logits`` models structural empty cells and ``bin_logits`` models
    positive integer-valued local counts.  ``route_logits`` is supervised with
    PET's existing dense-cell teacher.  A zero-initialized count-to-route
    residual lets learned EBC counts influence routing only when supported by
    the training data, preventing a new count prior from silently changing
    the initial PET routing behavior.
    """

    def __init__(
        self,
        hidden_dim,
        context_h,
        context_w,
        count_bin_centers,
        zero_prior=0.9,
        mid_dim=128,
        activation='gelu',
        route_count_threshold=2.0,
    ):
        super().__init__()
        centers = torch.as_tensor(count_bin_centers, dtype=torch.float32)
        if centers.ndim != 1 or centers.numel() < 2:
            raise ValueError('EBC router needs at least two positive count bins')
        if not bool(torch.all(centers > 0)):
            raise ValueError('EBC router count bins must be positive')
        if not bool(torch.all(centers[1:] > centers[:-1])):
            raise ValueError('EBC router count bins must be strictly increasing')
        if not bool(torch.allclose(centers, centers.round())):
            raise ValueError('EBC router count bins must be integer-valued')
        if activation == 'gelu':
            activation_layer = nn.GELU
        elif activation == 'relu':
            activation_layer = lambda: nn.ReLU(inplace=True)
        else:
            raise ValueError('EBC router activation must be "gelu" or "relu"')

        hidden_dim = int(hidden_dim)
        mid_dim = max(1, int(mid_dim))
        context_h = max(1, int(context_h))
        context_w = max(1, int(context_w))
        self.register_buffer('count_bin_centers', centers)
        self.route_count_threshold = float(route_count_threshold)
        self.context_pool = nn.AvgPool2d(
            (context_h, context_w),
            stride=(context_h, context_w),
            ceil_mode=False,
            count_include_pad=False,
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, mid_dim, 3, padding=1, bias=False),
            _group_norm(mid_dim),
            activation_layer(),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=2, dilation=2, bias=False),
            _group_norm(mid_dim),
            activation_layer(),
        )
        self.zero_head = nn.Conv2d(mid_dim, 1, 1)
        self.bin_head = nn.Conv2d(mid_dim, centers.numel(), 1)
        self.route_head = nn.Conv2d(mid_dim, 1, 1)
        # The residual is trainable but neutral at initialization.
        self.count_to_route = nn.Parameter(torch.zeros(()))

        zero_prior = min(max(float(zero_prior), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.zero_head.weight)
        nn.init.constant_(self.zero_head.bias, math.log(zero_prior / (1.0 - zero_prior)))
        nn.init.zeros_(self.bin_head.weight)
        nn.init.zeros_(self.bin_head.bias)
        # Most non-empty local cells begin near one person.
        self.bin_head.bias.data[0] = 4.0
        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.route_head.bias)

    def forward(self, x):
        pooled = self.context_pool(x)
        features = self.encoder(pooled)
        zero_logits = self.zero_head(features).squeeze(1)
        bin_logits = self.bin_head(features)
        centers = self.count_bin_centers.to(device=x.device, dtype=bin_logits.dtype)
        positive_count = (
            bin_logits.softmax(dim=1) * centers[None, :, None, None]
        ).sum(dim=1)
        expected_count = (1.0 - zero_logits.sigmoid()) * positive_count
        count_evidence = torch.log1p(expected_count) - math.log1p(self.route_count_threshold)
        route_logits = self.route_head(features).squeeze(1) + self.count_to_route * count_evidence
        return {
            'zero_logits': zero_logits,
            'bin_logits': bin_logits,
            'positive_count': positive_count,
            'expected_count': expected_count,
            'route_logits': route_logits,
            'route_prob': route_logits.sigmoid().unsqueeze(1),
        }
