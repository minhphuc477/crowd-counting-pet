"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import json
from contextlib import nullcontext
from typing import Iterable
import numpy as np
import cv2

import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy is already a project dependency via matcher.py
    linear_sum_assignment = None

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


def _predict_count(model, samples, targets, epoch=0):
    outputs = model(samples, test=True, targets=targets, epoch=epoch)
    if 'count_for_mae' in outputs:
        count_value = outputs['count_for_mae']
        if torch.is_tensor(count_value):
            count_value = count_value.detach().float().reshape(-1)[0].item()
        return outputs, float(count_value)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    return outputs, float(len(outputs_scores))


def _valid_hw(samples):
    if samples.mask is None:
        return samples.tensors.shape[-2:]
    valid_h = int((~samples.mask[0]).any(dim=1).sum().item())
    valid_w = int((~samples.mask[0]).any(dim=0).sum().item())
    return max(valid_h, 1), max(valid_w, 1)


def _greedy_match_count(distances, threshold):
    valid_pairs = torch.nonzero(distances <= threshold, as_tuple=False)
    if valid_pairs.numel() == 0:
        return 0
    valid_distances = distances[valid_pairs[:, 0], valid_pairs[:, 1]]
    order = torch.argsort(valid_distances)
    pred_used = torch.zeros(distances.shape[0], dtype=torch.bool, device=distances.device)
    gt_used = torch.zeros(distances.shape[1], dtype=torch.bool, device=distances.device)
    true_positive = 0
    for pair_index in order.tolist():
        pred_index = int(valid_pairs[pair_index, 0].item())
        gt_index = int(valid_pairs[pair_index, 1].item())
        if pred_used[pred_index] or gt_used[gt_index]:
            continue
        pred_used[pred_index] = True
        gt_used[gt_index] = True
        true_positive += 1
    return true_positive


def _localization_match_counts(pred_points, gt_points, threshold):
    """Return TP/FP/FN for point localization under a fixed or per-GT threshold."""
    pred_count = int(pred_points.shape[0])
    gt_count = int(gt_points.shape[0])
    if pred_count == 0 or gt_count == 0:
        return 0, pred_count, gt_count

    distances = torch.cdist(pred_points.float(), gt_points.float(), p=2)
    if torch.is_tensor(threshold):
        threshold_tensor = threshold.to(device=distances.device, dtype=distances.dtype).reshape(1, -1)
        if threshold_tensor.numel() != gt_count:
            raise ValueError(f'per-GT localization threshold has {threshold_tensor.numel()} values for {gt_count} GT points')
        valid_pairs_mask = distances <= threshold_tensor
        invalid_cost = max(float(threshold_tensor.max().detach().cpu().item()) + 1.0, 1.0) * 1_000_000.0
    else:
        threshold = float(threshold)
        if threshold <= 0:
            true_positive = 0
            false_positive = pred_count - true_positive
            false_negative = gt_count - true_positive
            return true_positive, false_positive, false_negative
        valid_pairs_mask = distances <= threshold
        invalid_cost = max(threshold + 1.0, 1.0) * 1_000_000.0

    if not bool(valid_pairs_mask.any().item()):
        true_positive = 0
    elif linear_sum_assignment is None:
        valid_pairs = torch.nonzero(valid_pairs_mask, as_tuple=False)
        valid_distances = distances[valid_pairs[:, 0], valid_pairs[:, 1]]
        order = torch.argsort(valid_distances)
        pred_used = torch.zeros(pred_count, dtype=torch.bool, device=distances.device)
        gt_used = torch.zeros(gt_count, dtype=torch.bool, device=distances.device)
        true_positive = 0
        for pair_index in order.tolist():
            pred_index = int(valid_pairs[pair_index, 0].item())
            gt_index = int(valid_pairs[pair_index, 1].item())
            if pred_used[pred_index] or gt_used[gt_index]:
                continue
            pred_used[pred_index] = True
            gt_used[gt_index] = True
            true_positive += 1
    else:
        cost = distances.detach().cpu().numpy()
        cost = np.where(valid_pairs_mask.detach().cpu().numpy(), cost, invalid_cost)
        row_ind, col_ind = linear_sum_assignment(cost)
        true_positive = int(np.sum(cost[row_ind, col_ind] < invalid_cost))
    false_positive = pred_count - true_positive
    false_negative = gt_count - true_positive
    return true_positive, false_positive, false_negative


def _nearest_neighbor_sigma(gt_points, scale, fallback, min_value=1.0):
    gt_count = int(gt_points.shape[0])
    if gt_count <= 1:
        return gt_points.new_full((gt_count,), float(fallback))
    distances = torch.cdist(gt_points.float(), gt_points.float(), p=2)
    eye = torch.eye(gt_count, dtype=torch.bool, device=gt_points.device)
    distances = distances.masked_fill(eye, float('inf'))
    nearest = distances.min(dim=1).values
    sigma = nearest * float(scale)
    return sigma.clamp_min(float(min_value))


def _target_sigma(target, name, device, fallback_threshold, gt_points_abs, large_scale=1.0, small_scale=0.5):
    """Return threshold descriptor and tensor/scalar for localization matching.

    Official NWPU-style localization uses per-GT small/large sigma. SHA only
    ships point annotations, so `adaptive_nn` is a transparent fallback for
    point-only datasets and must not be reported as the official NWPU metric.
    """
    if 'sigma' in target:
        sigma = target['sigma'].to(device=device, dtype=torch.float32)
        if sigma.ndim == 2 and sigma.shape[1] >= 2:
            return 'target_sigma', sigma[:, 1 if name == 'large' else 0]
        if sigma.ndim == 1:
            return 'target_sigma', sigma
    return 'adaptive_nn', _nearest_neighbor_sigma(
        gt_points_abs,
        large_scale if name == 'large' else small_scale,
        fallback_threshold,
    )


def _localization_summary(true_positive, false_positive, false_negative):
    precision_denom = true_positive + false_positive
    recall_denom = true_positive + false_negative
    precision = true_positive / precision_denom if precision_denom > 0 else 0.0
    recall = true_positive / recall_denom if recall_denom > 0 else 0.0
    f1_denom = precision + recall
    f1 = (2.0 * precision * recall / f1_denom) if f1_denom > 0 else 0.0
    return precision, recall, f1


def _add_localization_result(results, name, threshold, true_positive, false_positive, false_negative):
    """Store point-localization metrics under legacy and paper-style names."""
    precision, recall, f1 = _localization_summary(true_positive, false_positive, false_negative)
    if torch.is_tensor(threshold):
        threshold_value = float(threshold.detach().float().mean().cpu().item()) if threshold.numel() else 0.0
    else:
        threshold_value = float(threshold)
    results[f'loc_threshold_{name}'] = threshold_value
    results[f'loc_tp_{name}'] = float(true_positive)
    results[f'loc_fp_{name}'] = float(false_positive)
    results[f'loc_fn_{name}'] = float(false_negative)
    results[f'loc_prec_{name}'] = precision
    results[f'loc_rec_{name}'] = recall
    results[f'loc_f1_{name}'] = f1

    sigma_name = {'large': 'sigma_l', 'small': 'sigma_s'}.get(name)
    if sigma_name is not None:
        results[f'loc_threshold_{sigma_name}'] = threshold_value
        results[f'loc_tp_{sigma_name}'] = float(true_positive)
        results[f'loc_fp_{sigma_name}'] = float(false_positive)
        results[f'loc_fn_{sigma_name}'] = float(false_negative)
        results[f'loc_prec_{sigma_name}'] = precision
        results[f'loc_rec_{sigma_name}'] = recall
        results[f'loc_f1_{sigma_name}'] = f1


def format_localization_metrics(metrics, prefix=''):
    """Compact F1/precision/recall text for crowd-localization result logs."""
    if 'loc_f1_large' not in metrics or 'loc_f1_small' not in metrics:
        return ''

    def value(key):
        raw = metrics.get(key, 0.0)
        if torch.is_tensor(raw):
            raw = raw.detach().cpu().item()
        return float(raw)

    return (
        f"{prefix}loc_sigma_l(F1/Prec/Rec)="
        f"{value('loc_f1_large'):.4f}/"
        f"{value('loc_prec_large'):.4f}/"
        f"{value('loc_rec_large'):.4f} "
        f"loc_sigma_s(F1/Prec/Rec)="
        f"{value('loc_f1_small'):.4f}/"
        f"{value('loc_prec_small'):.4f}/"
        f"{value('loc_rec_small'):.4f}"
    )


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
def evaluate_crowd_no_overlap(
    model,
    data_loader,
    device,
    epoch=0,
    vis_dir=None,
    tta_flip=False,
    tta_scales=None,
    localization_metrics=True,
    localization_large_threshold=8.0,
    localization_small_threshold=4.0,
    localization_protocol='fixed',
    localization_large_scale=1.0,
    localization_small_scale=0.5,
):
    """P2PNet/APGCC-style full-image crowd evaluation without crop overlap.

    The APGCC issue #7 refers to P2PNet's evaluator. PET already evaluates full
    validation images, so this wrapper preserves that protocol while accepting
    PET targets (`points`) and its thresholded `test_forward()` output.
    """
    return evaluate(
        model,
        data_loader,
        device,
        epoch=epoch,
        vis_dir=vis_dir,
        tta_flip=tta_flip,
        tta_scales=tta_scales,
        localization_metrics=localization_metrics,
        localization_large_threshold=localization_large_threshold,
        localization_small_threshold=localization_small_threshold,
        localization_protocol=localization_protocol,
        localization_large_scale=localization_large_scale,
        localization_small_scale=localization_small_scale,
    )


# evaluation
@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    epoch=0,
    vis_dir=None,
    tta_flip=False,
    tta_scales=None,
    localization_metrics=True,
    localization_large_threshold=8.0,
    localization_small_threshold=4.0,
    localization_protocol='fixed',
    localization_large_scale=1.0,
    localization_small_scale=0.5,
    per_image_results_file=None,
):
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
    loc_thresholds = {
        'large': float(localization_large_threshold),
        'small': float(localization_small_threshold),
    }
    localization_protocol = str(localization_protocol)
    if localization_protocol not in ('fixed', 'target_sigma', 'adaptive_nn'):
        raise ValueError("localization_protocol must be one of fixed, target_sigma, adaptive_nn")
    loc_totals = {
        name: {'tp': 0.0, 'fp': 0.0, 'fn': 0.0}
        for name in loc_thresholds
    }
    loc_threshold_sums = {name: 0.0 for name in loc_thresholds}
    loc_threshold_counts = {name: 0.0 for name in loc_thresholds}
    loc_protocol_used = {name: localization_protocol for name in loc_thresholds}
    per_image_rows = [] if per_image_results_file else None
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        if len(targets) != 1 or samples.tensors.shape[0] != 1:
            raise ValueError('PET evaluation expects batch_size=1; counting metrics are image-level.')
        img_h, img_w = _valid_hw(samples)

        # inference
        outputs, predict_cnt = _predict_count(model, samples, targets, epoch=epoch)
        # outputs_scores: per-query person probability, shape [N_queries]
        # test_forward() already applies score thresholding and returns only
        # surviving (person) queries in pred_logits, so len(outputs_scores) is
        # the predicted count — matching the original PET evaluation protocol.
        outputs_points = outputs['pred_points'][0]
        if tta_flip or any(abs(scale - 1.0) > 1e-6 for scale in tta_scales):
            tta_counts = []
            for scale in tta_scales:
                scaled_samples = _resize_nested_tensor(samples, scale)
                _, scaled_count = _predict_count(model, scaled_samples, targets, epoch=epoch)
                tta_counts.append(scaled_count)
                if tta_flip:
                    flipped_samples = NestedTensor(
                        torch.flip(scaled_samples.tensors, dims=[3]),
                        torch.flip(scaled_samples.mask, dims=[2]),
                    )
                    _, flipped_count = _predict_count(model, flipped_samples, targets, epoch=epoch)
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

        if localization_metrics:
            per_image_loc = {}
            if outputs_points.numel() == 0:
                pred_points_abs = outputs_points.detach().reshape(0, 2).to(device=device, dtype=torch.float32)
            else:
                pred_points_abs = outputs_points.detach().to(device=device, dtype=torch.float32)
                pred_points_abs = pred_points_abs * pred_points_abs.new_tensor([float(img_h), float(img_w)])
            gt_points_abs = targets[0]['points'].to(device=device, dtype=torch.float32)
            for name, threshold in loc_thresholds.items():
                match_threshold = threshold
                protocol_used = localization_protocol
                if localization_protocol == 'target_sigma':
                    protocol_used, match_threshold = _target_sigma(
                        targets[0],
                        name,
                        device,
                        threshold,
                        gt_points_abs,
                        large_scale=localization_large_scale,
                        small_scale=localization_small_scale,
                    )
                elif localization_protocol == 'adaptive_nn':
                    match_threshold = _nearest_neighbor_sigma(
                        gt_points_abs,
                        localization_large_scale if name == 'large' else localization_small_scale,
                        threshold,
                    )
                loc_protocol_used[name] = protocol_used
                if torch.is_tensor(match_threshold):
                    loc_threshold_sums[name] += float(match_threshold.detach().float().sum().cpu().item())
                    loc_threshold_counts[name] += float(match_threshold.numel())
                else:
                    loc_threshold_sums[name] += float(match_threshold)
                    loc_threshold_counts[name] += 1.0
                tp, fp, fn = _localization_match_counts(pred_points_abs, gt_points_abs, match_threshold)
                loc_totals[name]['tp'] += float(tp)
                loc_totals[name]['fp'] += float(fp)
                loc_totals[name]['fn'] += float(fn)
                per_image_loc[f'loc_tp_{name}'] = int(tp)
                per_image_loc[f'loc_fp_{name}'] = int(fp)
                per_image_loc[f'loc_fn_{name}'] = int(fn)
                denom_p = float(tp + fp)
                denom_r = float(tp + fn)
                prec = float(tp) / denom_p if denom_p > 0 else 0.0
                rec = float(tp) / denom_r if denom_r > 0 else 0.0
                per_image_loc[f'loc_prec_{name}'] = prec
                per_image_loc[f'loc_rec_{name}'] = rec
                per_image_loc[f'loc_f1_{name}'] = (
                    2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                )

        if per_image_rows is not None:
            target = targets[0]
            row = {
                'image_id': str(target.get('image_id', '')),
                'image_path': str(target.get('image_path', '')),
                'gt_cnt': int(gt_cnt),
                'pred_cnt': float(predict_cnt),
                'abs_error': float(mae),
                'sq_error': float(mse_sq),
            }
            if localization_metrics:
                row.update(per_image_loc)
            per_image_rows.append(row)

        if 'eval_count_debug' in outputs:
            metric_logger.update(**{
                f'dbg_{key}': float(value)
                for key, value in outputs['eval_count_debug'].items()
            })

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
    if localization_metrics:
        loc_reduce = {}
        for name, counts in loc_totals.items():
            for count_name, value in counts.items():
                loc_reduce[f'loc_{count_name}_{name}'] = torch.tensor(value, dtype=torch.float64, device=device)
            loc_reduce[f'loc_threshold_sum_{name}'] = torch.tensor(loc_threshold_sums[name], dtype=torch.float64, device=device)
            loc_reduce[f'loc_threshold_count_{name}'] = torch.tensor(loc_threshold_counts[name], dtype=torch.float64, device=device)
        loc_reduce = utils.reduce_dict(loc_reduce, average=False)
        results['loc_protocol'] = localization_protocol
        for name in loc_thresholds:
            tp = float(loc_reduce[f'loc_tp_{name}'].item())
            fp = float(loc_reduce[f'loc_fp_{name}'].item())
            fn = float(loc_reduce[f'loc_fn_{name}'].item())
            threshold_sum = float(loc_reduce[f'loc_threshold_sum_{name}'].item())
            threshold_count = float(loc_reduce[f'loc_threshold_count_{name}'].item())
            threshold = threshold_sum / threshold_count if threshold_count > 0 else loc_thresholds[name]
            _add_localization_result(results, name, threshold, tp, fp, fn)
            results[f'loc_protocol_{name}'] = loc_protocol_used[name]
    if per_image_rows is not None and utils.is_main_process():
        per_image_rows.sort(key=lambda row: row['sq_error'], reverse=True)
        os.makedirs(os.path.dirname(per_image_results_file) or '.', exist_ok=True)
        with open(per_image_results_file, 'w', encoding='utf-8') as handle:
            json.dump(per_image_rows, handle, indent=2)
    return results
