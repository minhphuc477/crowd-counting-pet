"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from contextlib import nullcontext
from typing import Iterable
import numpy as np
import cv2

import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F

import util.misc as utils
from util.misc import NestedTensor


def autocast_context(device, enabled=False, dtype=None):
    if not enabled or device.type != 'cuda':
        return nullcontext()
    kwargs = {}
    if dtype is not None:
        kwargs['dtype'] = dtype
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda', **kwargs)
    return torch.cuda.amp.autocast(**kwargs)


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, targets, pred, vis_dir, split_map=None):
    """
    Visualize predictions
    """
    gts = [t['points'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # draw ground-truth points (red)
        size = 2
        for t in gts[idx]:
            sample_vis = cv2.circle(sample_vis, (int(t[1]), int(t[0])), size, (0, 0, 255), -1)

        # draw predictions (green)
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
        
        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        
        # save image
        if vis_dir is not None:
            # eliminate invalid area
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]

            name = targets[idx]['image_path'].split('/')[-1].split('.')[0]
            cv2.imwrite(os.path.join(vis_dir, '{}_gt{}_pred{}.jpg'.format(name, len(gts[idx]), len(pred[idx]))), sample_vis)


# training
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    model_ema=None, model_without_ddp=None, freeze_bn: bool = False,
                    amp_enabled: bool = False, scaler=None, accum_iter: int = 1,
                    amp_dtype=None):
    model.train()
    criterion.train()
    accum_iter = max(1, int(accum_iter))
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad(set_to_none=True)
    for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt_points = [target['points'] for target in targets]

        with autocast_context(device, amp_enabled, amp_dtype):
            outputs = model(samples, epoch=epoch, train=True,
                                            criterion=criterion, targets=targets)
        loss_dict, weight_dict, losses = outputs['loss_dict'], outputs['weight_dict'], outputs['losses']

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        missing_loss_weights = sorted(set(loss_dict_reduced) - set(weight_dict))
        if missing_loss_weights:
            raise RuntimeError(f'Loss keys missing weights: {missing_loss_weights}')
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        should_step = (step + 1) % accum_iter == 0 or (step + 1) == len(data_loader)
        losses_to_backward = losses / accum_iter
        if amp_enabled and scaler is not None:
            scaler.scale(losses_to_backward).backward()
            if should_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            losses_to_backward.backward()
            if should_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if should_step and model_ema is not None:
            source_model = model_without_ddp if model_without_ddp is not None else model
            model_ema.update(source_model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _predict_count(model, samples, targets):
    outputs = model(samples, test=True, targets=targets)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    return outputs, float(len(outputs_scores))


def _ceil_to_multiple(value, multiple=256):
    return max(multiple, int(math.ceil(float(value) / multiple)) * multiple)


def _resize_nested_tensor(samples, scale):
    scale = float(scale)
    if abs(scale - 1.0) < 1e-6:
        return samples
    
    # Extract valid image dimensions from the mask
    # Mask is False for valid pixels, True for padding (padded on bottom/right)
    valid_h = (~samples.mask[0, :, 0]).sum().item()
    valid_w = (~samples.mask[0, 0, :]).sum().item()
    
    valid_img = samples.tensors[:, :, :valid_h, :valid_w]
    
    scaled_h = max(1, int(round(valid_h * scale)))
    scaled_w = max(1, int(round(valid_w * scale)))
    pad_h = _ceil_to_multiple(scaled_h)
    pad_w = _ceil_to_multiple(scaled_w)
    
    tensors = F.interpolate(valid_img, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
    
    # Recreate the padded tensor and mask
    b, c = samples.tensors.shape[:2]
    padded_tensors = torch.zeros((b, c, pad_h, pad_w), dtype=tensors.dtype, device=tensors.device)
    padded_tensors[:, :, :scaled_h, :scaled_w] = tensors
    
    padded_mask = torch.ones((b, pad_h, pad_w), dtype=torch.bool, device=tensors.device)
    padded_mask[:, :scaled_h, :scaled_w] = False
    
    return NestedTensor(padded_tensors, padded_mask)


@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None, tta_flip=False, tta_scales=None):
    """P2PNet/APGCC-style full-image crowd evaluation without crop overlap.

    The APGCC issue #7 refers to P2PNet's evaluator. PET already evaluates full
    validation images, so this wrapper preserves that protocol while accepting
    PET targets (`points`) and its thresholded `test_forward()` output.
    """
    return evaluate(model, data_loader, device, vis_dir=vis_dir, tta_flip=tta_flip, tta_scales=tta_scales)


# evaluation
@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, vis_dir=None, tta_flip=False, tta_scales=None):
    model.eval()
    if tta_scales is None:
        tta_scales = (1.0,)
    tta_scales = tuple(dict.fromkeys(float(scale) for scale in tta_scales if float(scale) > 0))
    if not tta_scales:
        tta_scales = (1.0,)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        if len(targets) != 1 or samples.tensors.shape[0] != 1:
            raise ValueError('PET evaluation expects batch_size=1; counting metrics are image-level.')
        img_h, img_w = samples.tensors.shape[-2:]

        # inference
        outputs, predict_cnt = _predict_count(model, samples, targets)
        # outputs_scores: per-query person probability, shape [N_queries]
        # test_forward() already applies score thresholding and returns only
        # surviving (person) queries in pred_logits, so len(outputs_scores) is
        # the predicted count — matching the original PET evaluation protocol.
        outputs_points = outputs['pred_points'][0]
        if tta_flip or any(abs(scale - 1.0) > 1e-6 for scale in tta_scales):
            tta_counts = []
            for scale in tta_scales:
                scaled_samples = _resize_nested_tensor(samples, scale)
                _, scaled_count = _predict_count(model, scaled_samples, targets)
                tta_counts.append(scaled_count)
                if tta_flip:
                    flipped_samples = NestedTensor(
                        torch.flip(scaled_samples.tensors, dims=[3]),
                        torch.flip(scaled_samples.mask, dims=[2]),
                    )
                    _, flipped_count = _predict_count(model, flipped_samples, targets)
                    tta_counts.append(flipped_count)
            predict_cnt = float(sum(tta_counts) / len(tta_counts))
        gt_cnt = targets[0]['points'].shape[0]

        # compute error
        mae = abs(predict_cnt - gt_cnt)
        mse_sq = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)

        # record results
        results = {}
        toTensor = lambda x: torch.tensor(x).float().to(device)
        results['mae'], results['mse_sq'] = toTensor(mae), toTensor(mse_sq)
        results['pred_cnt'], results['gt_cnt'] = toTensor(predict_cnt), toTensor(gt_cnt)
        results_reduced = utils.reduce_dict(results)
        metric_logger.update(mae=results_reduced['mae'], mse_sq=results_reduced['mse_sq'])
        metric_logger.update(pred_cnt=results_reduced['pred_cnt'], gt_cnt=results_reduced['gt_cnt'])

        # visualize predictions
        if vis_dir: 
            points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
            split_threshold = outputs.get('split_threshold', 0.5)
            if torch.is_tensor(split_threshold):
                split_threshold = float(split_threshold.detach().cpu().item())
            split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > split_threshold).float().numpy()
            visualization(samples, targets, [points], vis_dir, split_map=split_map)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['mse'] = np.sqrt(results.pop('mse_sq'))
    return results
