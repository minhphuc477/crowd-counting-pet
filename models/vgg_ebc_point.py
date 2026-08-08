"""Standalone VGG-FPN block counting and point localization model.

This branch intentionally does not inherit PET's quadtree or Transformer.
Counting is learned as blockwise classification, while localization is learned
with a fixed-capacity point-slot head on the stride-4 feature map.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import build_backbone


def _activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    return nn.GELU()


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation="gelu"):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(32, out_channels),
            _activation(activation),
        )


class ResidualContextFusion(nn.Module):
    """Fuse 8x context into 4x detail without erasing pretrained FPN features."""

    def __init__(self, channels=256, activation="gelu", init=1e-3):
        super().__init__()
        self.detail = ConvNormAct(channels, channels, 3, activation)
        self.context = ConvNormAct(channels, channels, 3, activation)
        self.mix = nn.Sequential(
            ConvNormAct(channels * 2, channels, 1, activation),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.gate = nn.Parameter(torch.tensor(float(init)))

    def forward(self, detail, context):
        context = F.interpolate(context, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        residual = self.mix(torch.cat((self.detail(detail), self.context(context)), dim=1))
        return detail + self.gate * residual


class EmptyCriterion(nn.Module):
    """Training loop compatibility; losses are computed inside the model."""

    def forward(self, *args, **kwargs):  # pragma: no cover - intentionally unused
        raise RuntimeError("VGG EBC-Point computes its losses in model.forward")


class VGGEBCPoint(nn.Module):
    def __init__(self, args):
        super().__init__()
        if getattr(args, "backbone", "vgg16_bn") != "vgg16_bn":
            raise ValueError("vgg_ebc_point currently requires --backbone vgg16_bn")

        self.backbone = build_backbone(args)
        channels = int(getattr(args, "ebc_point_hidden_dim", 256))
        if channels % 32:
            raise ValueError("--ebc_point_hidden_dim must be divisible by 32")
        activation = getattr(args, "ebc_point_activation", "gelu")
        self.block_size = int(getattr(args, "ebc_point_block_size", 32))
        self.point_slots = int(getattr(args, "ebc_point_slots", 4))
        if self.block_size < 4 or self.block_size % 4:
            raise ValueError("--ebc_point_block_size must be a positive multiple of 4")
        if self.point_slots < 1:
            raise ValueError("--ebc_point_slots must be positive")

        self.input_4 = nn.Conv2d(256, channels, 1)
        self.input_8 = nn.Conv2d(256, channels, 1)
        self.fusion = ResidualContextFusion(
            channels,
            activation=activation,
            init=float(getattr(args, "ebc_point_fusion_init", 1e-3)),
        )
        self.refine = nn.Sequential(
            ConvNormAct(channels, channels, 3, activation),
            ConvNormAct(channels, channels, 3, activation),
        )

        centers_text = getattr(
            args,
            "ebc_point_bin_centers",
            "0,1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512",
        )
        centers = torch.tensor([float(v) for v in str(centers_text).split(",")], dtype=torch.float32)
        if centers.numel() < 2 or centers[0].item() != 0 or not bool(torch.all(centers[1:] > centers[:-1])):
            raise ValueError("--ebc_point_bin_centers must be increasing and start at 0")
        self.register_buffer("count_bin_centers", centers, persistent=True)
        self.count_head = nn.Sequential(
            ConvNormAct(channels, channels, 3, activation),
            nn.Conv2d(channels, centers.numel(), 1),
        )
        self.point_head = nn.Sequential(
            ConvNormAct(channels, channels, 3, activation),
            nn.Conv2d(channels, self.point_slots * 3, 1),
        )

        prior = float(getattr(args, "ebc_point_prior", 0.01))
        prior = min(max(prior, 1e-5), 1.0 - 1e-5)
        point_bias = math.log(prior / (1.0 - prior))
        final_point = self.point_head[-1]
        zero_prior = float(getattr(args, "ebc_point_zero_prior", 0.95))
        zero_prior = min(max(zero_prior, 1e-5), 1.0 - 1e-5)
        nonzero_prior = torch.exp(-centers[1:] / 8.0)
        nonzero_prior = nonzero_prior / nonzero_prior.sum() * (1.0 - zero_prior)
        with torch.no_grad():
            nn.init.normal_(self.count_head[-1].weight, std=0.01)
            self.count_head[-1].bias[0] = math.log(zero_prior)
            self.count_head[-1].bias[1:].copy_(nonzero_prior.log())
            final_point.bias.view(self.point_slots, 3)[:, 0].fill_(point_bias)

        self.score_threshold = float(getattr(args, "score_threshold", 0.5))
        self.focal_alpha = float(getattr(args, "ebc_point_focal_alpha", 0.25))
        self.focal_gamma = float(getattr(args, "ebc_point_focal_gamma", 2.0))
        self.zero_class_weight = float(getattr(args, "ebc_point_zero_class_weight", 0.25))
        self.weight_dict = {
            "loss_ebc_ce": float(getattr(args, "ebc_point_ce_coef", 1.0)),
            "loss_ebc_local": float(getattr(args, "ebc_point_local_count_coef", 0.25)),
            "loss_ebc_global": float(getattr(args, "ebc_point_global_count_coef", 0.25)),
            "loss_ebc_multiscale": float(getattr(args, "ebc_point_multiscale_coef", 0.1)),
            "loss_point_cls": float(getattr(args, "ebc_point_cls_coef", 1.0)),
            "loss_point_coord": float(getattr(args, "ebc_point_coord_coef", 5.0)),
            "loss_count_consistency": float(getattr(args, "ebc_point_consistency_coef", 0.02)),
        }

    @staticmethod
    def _valid_image_hw(samples, index):
        if samples.mask is None:
            return samples.tensors.shape[-2:]
        valid = ~samples.mask[index]
        height = int(valid.any(dim=1).sum().item())
        width = int(valid.any(dim=0).sum().item())
        return max(height, 1), max(width, 1)

    def _count_features(self, fused, feature_mask):
        kernel = self.block_size // 4
        valid = (~feature_mask).to(dtype=fused.dtype).unsqueeze(1)
        numerator = F.avg_pool2d(
            fused * valid, kernel_size=kernel, stride=kernel, ceil_mode=True,
        )
        denominator = F.avg_pool2d(
            valid, kernel_size=kernel, stride=kernel, ceil_mode=True,
        )
        return numerator / denominator.clamp(min=1e-6)

    def _forward_heads(self, samples):
        features, _ = self.backbone(samples)
        detail = self.input_4(features["4x"].tensors)
        context = self.input_8(features["8x"].tensors)
        fused = self.refine(self.fusion(detail, context))
        count_logits = self.count_head(self._count_features(fused, features["4x"].mask))
        point_raw = self.point_head(fused)
        batch, _, height, width = point_raw.shape
        point_raw = point_raw.view(batch, self.point_slots, 3, height, width)
        return count_logits, point_raw

    def _build_targets(self, samples, targets, count_shape, point_shape, device):
        batch, count_h, count_w = count_shape
        _, _, point_h, point_w = point_shape
        block_targets = torch.zeros((batch, count_h, count_w), device=device)
        block_valid = torch.zeros((batch, count_h, count_w), dtype=torch.bool, device=device)
        point_labels = torch.zeros((batch, self.point_slots, point_h, point_w), device=device)
        point_offsets = torch.zeros((batch, self.point_slots, 2, point_h, point_w), device=device)
        point_valid = torch.zeros((batch, point_h, point_w), dtype=torch.bool, device=device)
        global_counts = torch.zeros((batch,), device=device)

        stride_y = samples.tensors.shape[-2] / float(point_h)
        stride_x = samples.tensors.shape[-1] / float(point_w)
        for index, target in enumerate(targets):
            valid_h, valid_w = self._valid_image_hw(samples, index)
            valid_count_h = min(count_h, math.ceil(valid_h / self.block_size))
            valid_count_w = min(count_w, math.ceil(valid_w / self.block_size))
            block_valid[index, :valid_count_h, :valid_count_w] = True
            valid_point_h = min(point_h, math.ceil(valid_h / stride_y))
            valid_point_w = min(point_w, math.ceil(valid_w / stride_x))
            point_valid[index, :valid_point_h, :valid_point_w] = True

            points = target["points"].to(device=device, dtype=torch.float32)
            global_counts[index] = float(points.shape[0])
            if points.numel() == 0:
                continue
            inside = (
                (points[:, 0] >= 0) & (points[:, 0] < valid_h)
                & (points[:, 1] >= 0) & (points[:, 1] < valid_w)
            )
            points = points[inside]
            if points.numel() == 0:
                continue

            block_y = torch.clamp((points[:, 0] / self.block_size).long(), 0, count_h - 1)
            block_x = torch.clamp((points[:, 1] / self.block_size).long(), 0, count_w - 1)
            flat_blocks = block_y * count_w + block_x
            block_targets[index].view(-1).scatter_add_(
                0, flat_blocks, torch.ones_like(flat_blocks, dtype=torch.float32)
            )

            feature_y = points[:, 0] / stride_y
            feature_x = points[:, 1] / stride_x
            cell_y = torch.clamp(feature_y.floor().long(), 0, point_h - 1)
            cell_x = torch.clamp(feature_x.floor().long(), 0, point_w - 1)
            flat_cells = cell_y * point_w + cell_x
            sorted_cells, order = torch.sort(flat_cells, stable=True)
            sequence = torch.arange(sorted_cells.numel(), device=device)
            starts = torch.zeros_like(sequence)
            is_start = torch.ones_like(sorted_cells, dtype=torch.bool)
            is_start[1:] = sorted_cells[1:] != sorted_cells[:-1]
            starts[is_start] = sequence[is_start]
            starts = torch.cummax(starts, dim=0).values
            sorted_slots = sequence - starts
            retained = sorted_slots < self.point_slots
            point_index = order[retained]
            slots = sorted_slots[retained]
            y = cell_y[point_index]
            x = cell_x[point_index]
            point_labels[index, slots, y, x] = 1.0
            point_offsets[index, slots, 0, y, x] = feature_y[point_index] - y
            point_offsets[index, slots, 1, y, x] = feature_x[point_index] - x
        return block_targets, block_valid, point_labels, point_offsets, point_valid, global_counts

    def _nearest_bins(self, counts):
        distances = (counts.unsqueeze(-1) - self.count_bin_centers).abs()
        return distances.argmin(dim=-1)

    def _losses(self, samples, targets, count_logits, point_raw):
        batch, _, count_h, count_w = count_logits.shape
        _, _, _, point_h, point_w = point_raw.shape
        built = self._build_targets(
            samples, targets, (batch, count_h, count_w), (batch, self.point_slots, point_h, point_w), count_logits.device
        )
        block_targets, block_valid, point_labels, point_offsets, point_valid, global_counts = built

        count_logits_cells = count_logits.permute(0, 2, 3, 1)
        bin_targets = self._nearest_bins(block_targets)
        class_weights = torch.ones_like(self.count_bin_centers)
        class_weights[0] = self.zero_class_weight
        loss_ebc_ce = F.cross_entropy(
            count_logits_cells[block_valid], bin_targets[block_valid], weight=class_weights,
        )
        probabilities = count_logits_cells.softmax(dim=-1)
        expected_blocks = (probabilities * self.count_bin_centers).sum(dim=-1)
        loss_ebc_local = F.smooth_l1_loss(
            torch.log1p(expected_blocks[block_valid]),
            torch.log1p(block_targets[block_valid]),
        )
        predicted_counts = (expected_blocks * block_valid).flatten(1).sum(dim=1)
        loss_ebc_global = F.smooth_l1_loss(torch.log1p(predicted_counts), torch.log1p(global_counts))
        multiscale_losses = []
        predicted_map = expected_blocks * block_valid
        target_map = block_targets * block_valid
        for scale in (2, 4):
            if min(count_h, count_w) < scale:
                continue
            pred_coarse = F.avg_pool2d(
                predicted_map[:, None], scale, stride=scale, ceil_mode=True,
                divisor_override=1,
            )[:, 0]
            target_coarse = F.avg_pool2d(
                target_map[:, None], scale, stride=scale, ceil_mode=True,
                divisor_override=1,
            )[:, 0]
            valid_coarse = F.max_pool2d(
                block_valid[:, None].float(), scale, stride=scale, ceil_mode=True,
            )[:, 0].bool()
            multiscale_losses.append(F.smooth_l1_loss(
                torch.log1p(pred_coarse[valid_coarse]),
                torch.log1p(target_coarse[valid_coarse]),
            ))
        loss_ebc_multiscale = (
            torch.stack(multiscale_losses).mean()
            if multiscale_losses else expected_blocks.sum() * 0.0
        )

        point_logits = point_raw[:, :, 0]
        valid_slots = point_valid[:, None].expand_as(point_logits)
        bce = F.binary_cross_entropy_with_logits(point_logits, point_labels, reduction="none")
        probabilities_point = point_logits.sigmoid()
        pt = probabilities_point * point_labels + (1.0 - probabilities_point) * (1.0 - point_labels)
        alpha = self.focal_alpha * point_labels + (1.0 - self.focal_alpha) * (1.0 - point_labels)
        positive_norm = point_labels.sum().clamp(min=1.0)
        loss_point_cls = (alpha * (1.0 - pt).pow(self.focal_gamma) * bce * valid_slots).sum() / positive_norm

        positive = point_labels.bool()
        if positive.any():
            predicted_offsets = point_raw[:, :, 1:3].sigmoid()
            positive_xy = positive[:, :, None].expand_as(predicted_offsets)
            loss_point_coord = F.smooth_l1_loss(predicted_offsets[positive_xy], point_offsets[positive_xy])
        else:
            loss_point_coord = point_raw.sum() * 0.0

        soft_point_counts = (probabilities_point * valid_slots).flatten(1).sum(dim=1)
        loss_count_consistency = F.smooth_l1_loss(
            torch.log1p(soft_point_counts), torch.log1p(predicted_counts.detach())
        )
        loss_dict = {
            "loss_ebc_ce": loss_ebc_ce,
            "loss_ebc_local": loss_ebc_local,
            "loss_ebc_global": loss_ebc_global,
            "loss_ebc_multiscale": loss_ebc_multiscale,
            "loss_point_cls": loss_point_cls,
            "loss_point_coord": loss_point_coord,
            "loss_count_consistency": loss_count_consistency,
        }
        return loss_dict

    def _inference(self, samples, count_logits, point_raw):
        count_probs = count_logits.permute(0, 2, 3, 1).softmax(dim=-1)
        expected_blocks = (count_probs * self.count_bin_centers).sum(dim=-1)
        point_logits = point_raw[:, :, 0]
        point_offsets = point_raw[:, :, 1:3].sigmoid()
        batch, slots, height, width = point_logits.shape
        if batch != 1:
            raise ValueError("VGG EBC-Point evaluation currently requires batch size 1")
        valid_h, valid_w = self._valid_image_hw(samples, 0)
        stride_y = samples.tensors.shape[-2] / float(height)
        stride_x = samples.tensors.shape[-1] / float(width)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=point_logits.device),
            torch.arange(width, device=point_logits.device),
            indexing="ij",
        )
        scores = point_logits[0].sigmoid()
        valid = ((grid_y * stride_y) < valid_h) & ((grid_x * stride_x) < valid_w)
        keep = (scores >= self.score_threshold) & valid.unsqueeze(0)
        slot_index, y_index, x_index = torch.where(keep)
        if slot_index.numel():
            y = (y_index + point_offsets[0, slot_index, 0, y_index, x_index]) * stride_y
            x = (x_index + point_offsets[0, slot_index, 1, y_index, x_index]) * stride_x
            points = torch.stack((y / samples.tensors.shape[-2], x / samples.tensors.shape[-1]), dim=-1)
            kept_logits = point_logits[0, slot_index, y_index, x_index]
            logits = torch.stack((torch.zeros_like(kept_logits), kept_logits), dim=-1).unsqueeze(0)
        else:
            points = point_logits.new_zeros((0, 2))
            logits = point_logits.new_zeros((1, 0, 2))

        count_h = min(expected_blocks.shape[1], math.ceil(valid_h / self.block_size))
        count_w = min(expected_blocks.shape[2], math.ceil(valid_w / self.block_size))
        count = expected_blocks[0, :count_h, :count_w].sum()
        return {
            "pred_logits": logits,
            "pred_points": points.unsqueeze(0),
            "count_for_mae": count,
            "eval_count_debug": {
                "ebc_block_count": float(count.detach().float().item()),
                "point_count": float(points.shape[0]),
            },
        }

    def forward(self, samples, epoch=0, train=False, criterion=None, targets=None, test=False, **kwargs):
        count_logits, point_raw = self._forward_heads(samples)
        if train:
            if targets is None:
                raise ValueError("targets are required for training")
            loss_dict = self._losses(samples, targets, count_logits, point_raw)
            losses = sum(loss_dict[name] * self.weight_dict[name] for name in loss_dict)
            return {"loss_dict": loss_dict, "weight_dict": self.weight_dict, "losses": losses}
        return self._inference(samples, count_logits, point_raw)


def build_vgg_ebc_point(args):
    model = VGGEBCPoint(args)
    criterion = EmptyCriterion()
    return model, criterion
