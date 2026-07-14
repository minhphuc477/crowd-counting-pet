import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = _ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.proj(x)
        return self.fuse(torch.cat((skip, x), dim=1))


def _build_vgg16_bn(pretrained):
    if not pretrained:
        try:
            return models.vgg16_bn(weights=None)
        except TypeError:
            return models.vgg16_bn(pretrained=False)

    try:
        return models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
    except (AttributeError, TypeError):
        return models.vgg16_bn(pretrained=True)


class VGG16BNAnnotationRestorer(nn.Module):
    """Predict an image-space (dy, dx) annotation restoration field."""

    def __init__(self, pretrained=True):
        super().__init__()
        features = list(_build_vgg16_bn(pretrained).features.children())
        if len(features) < 43:
            raise RuntimeError('torchvision vgg16_bn has an unexpected feature layout')

        # Keep each skip before the following max-pool, matching full, 1/2,
        # 1/4, 1/8, and 1/16 image resolutions.
        self.enc1 = nn.Sequential(*features[:6])
        self.enc2 = nn.Sequential(*features[6:13])
        self.enc3 = nn.Sequential(*features[13:23])
        self.enc4 = nn.Sequential(*features[23:33])
        self.enc5 = nn.Sequential(*features[33:43])

        self.up4 = _UpBlock(512, 512, 512)
        self.up3 = _UpBlock(512, 256, 256)
        self.up2 = _UpBlock(256, 128, 128)
        self.up1 = _UpBlock(128, 64, 64)
        self.vector_head = nn.Conv2d(64, 2, 1)
        nn.init.zeros_(self.vector_head.weight)
        nn.init.zeros_(self.vector_head.bias)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('annotation restorer input must have shape [B, 3, H, W]')
        if min(images.shape[-2:]) < 16:
            raise ValueError('annotation restorer input height and width must be at least 16')

        x1 = self.enc1(images)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.vector_head(x.float())


def sample_vector_field(vector_field, points_yx):
    """Bilinearly sample a single-image (dy, dx) field at subpixel points."""
    if vector_field.ndim == 3:
        vector_field = vector_field.unsqueeze(0)
    if vector_field.ndim != 4 or vector_field.shape[0] != 1 or vector_field.shape[1] != 2:
        raise ValueError('vector_field must have shape [2, H, W] or [1, 2, H, W]')
    if points_yx.ndim != 2 or points_yx.shape[1] != 2:
        raise ValueError('points_yx must have shape [N, 2]')
    if points_yx.numel() == 0:
        return vector_field.new_empty((0, 2))

    height, width = vector_field.shape[-2:]
    points = points_yx.to(device=vector_field.device, dtype=vector_field.dtype)
    grid_x = points[:, 1] * (2.0 / max(width - 1, 1)) - 1.0
    grid_y = points[:, 0] * (2.0 / max(height - 1, 1)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        vector_field,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    return sampled[0, :, 0].transpose(0, 1)
