"""
Train and eval functions used in main.py
"""
import math
import os
from contextlib import nullcontext
from typing import Iterable
import numpy as np
import cv2

import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F

import util.misc as utils
from util.misc import NestedTensor


class NonFiniteTrainingError(RuntimeError):
    pass


def _build_threshold_values(reference_threshold, threshold_sweep=False,
                            threshold_min=0.30, threshold_max=0.70, threshold_step=0.025):
    reference_threshold = float(reference_threshold)
    if not threshold_sweep:
        return [reference_threshold]

    if threshold_step <= 0:
        raise ValueError('threshold_step must be positive')
    if threshold_max < threshold_min:
        raise ValueError('threshold_max must be >= threshold_min')

    steps = int(round((threshold_max - threshold_min) / threshold_step))
    threshold_values = [round(threshold_min + idx * threshold_step, 6) for idx in range(steps + 1)]
    threshold_values = [min(max(value, 0.0), 1.0) for value in threshold_values]

    reference_threshold = round(min(max(reference_threshold, 0.0), 1.0), 6)
    if reference_threshold not in threshold_values:
        threshold_values.append(reference_threshold)
        threshold_values.sort()
    return threshold_values


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

        # Vẽ các điểm ground-truth màu đỏ để làm mốc so sánh với kết quả dự đoán.
        size = 2
        for t in gts[idx]:
            sample_vis = cv2.circle(sample_vis, (int(t[1]), int(t[0])), size, (0, 0, 255), -1)

        # Vẽ các điểm dự đoán màu xanh để quan sát trực quan chất lượng đếm.
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
        
        # Vẽ bản đồ phân tách vùng sparse/dense để kiểm tra quyết định của quadtree splitter.
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        
        # Lưu ảnh trực quan hóa ra đĩa để phục vụ debug, báo cáo và đối chiếu mô hình.
        if vis_dir is not None:
            # Loại bỏ vùng padding/không hợp lệ trước khi lưu ảnh để kết quả hiển thị chính xác hơn.
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]

            name = targets[idx]['image_path'].split('/')[-1].split('.')[0]
            cv2.imwrite(os.path.join(vis_dir, '{}_gt{}_pred{}.jpg'.format(name, len(gts[idx]), len(pred[idx]))), sample_vis)


# Bắt đầu giai đoạn huấn luyện: cập nhật tham số mô hình dựa trên dữ liệu và hàm mất mát.
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    use_amp: bool = False, accum_iter: int = 1, scaler=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    accum_iter = max(1, int(accum_iter))
    optimizer.zero_grad(set_to_none=True)
    data_loader_length = len(data_loader) if hasattr(data_loader, '__len__') else None

    for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt_points = [target['points'] for target in targets]
        should_step = ((step + 1) % accum_iter == 0)
        if data_loader_length is not None and step + 1 == data_loader_length:
            should_step = True

        sync_context = model.no_sync if hasattr(model, 'no_sync') and not should_step else nullcontext

        with sync_context():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp and device.type == 'cuda'):
                outputs = model(samples, epoch=epoch, train=True,
                                criterion=criterion, targets=targets)
                loss_dict, weight_dict, losses = outputs['loss_dict'], outputs['weight_dict'], outputs['losses']

        # Đồng bộ và giảm các giá trị loss giữa nhiều GPU để log phản ánh đúng toàn bộ tiến trình.
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        loss_to_backward = losses_reduced_scaled / accum_iter

        if not math.isfinite(loss_value):
            optimizer.zero_grad(set_to_none=True)
            raise NonFiniteTrainingError(
                "non-finite loss at epoch {} step {}: {} | reduced_losses={}".format(
                    epoch, step, loss_value, loss_dict_reduced
                )
            )

        if use_amp and scaler is not None:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        if should_step:
            if max_norm > 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Thu thập thống kê từ mọi tiến trình phân tán rồi mới tính giá trị tổng hợp cuối cùng.
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# Thực hiện giai đoạn đánh giá: chỉ suy luận và đo chất lượng mà không cập nhật trọng số.
@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, vis_dir=None, inference_threshold=None,
             threshold_sweep=False, threshold_min=0.30, threshold_max=0.70, threshold_step=0.025):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    model_without_ddp = model.module if hasattr(model, 'module') else model
    default_threshold = getattr(model_without_ddp, 'inference_threshold', 0.5)
    if inference_threshold is not None:
        default_threshold = inference_threshold
    threshold_values = _build_threshold_values(
        default_threshold,
        threshold_sweep=threshold_sweep,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )
    threshold_tensor = torch.tensor(threshold_values, device=device, dtype=torch.float32)
    mae_sums = torch.zeros(len(threshold_values), device=device, dtype=torch.float32)
    mse_sums = torch.zeros(len(threshold_values), device=device, dtype=torch.float32)
    sample_count = torch.zeros(1, device=device, dtype=torch.float32)
    default_threshold_index = min(
        range(len(threshold_values)),
        key=lambda idx: abs(threshold_values[idx] - float(default_threshold))
    )

    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        img_h, img_w = samples.tensors.shape[-2:]
        metric_device = samples.tensors.device

        # Chạy suy luận để lấy đầu ra dự đoán từ mô hình mà không cập nhật tham số.
        outputs = model(samples, test=True, targets=targets)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0].float()
        outputs_points = outputs['pred_points'][0]
        
        # Hậu xử lý điểm dự đoán (lọc, đổi thang đo) trước khi tính metric hoặc trực quan hóa.
        gt_cnt = targets[0]['points'].shape[0]
        gt_cnt_tensor = torch.tensor(float(gt_cnt), device=metric_device, dtype=torch.float32)
        predict_cnts = (outputs_scores.unsqueeze(0) > threshold_tensor.unsqueeze(1)).sum(dim=1).float()

        # Tính sai số giữa dự đoán và ground-truth để đánh giá chất lượng mô hình.
        diff = predict_cnts - gt_cnt_tensor
        mae_sums += diff.abs()
        mse_sums += diff.square()
        sample_count += 1.0
        mae = diff[default_threshold_index].abs()
        mse = diff[default_threshold_index].square()

        # Ghi lại kết quả từng mẫu/từng batch để tổng hợp thống kê cuối cùng.
        results = {}
        results['mae'], results['mse'] = mae, mse

        results_reduced = utils.reduce_dict(results)
        metric_logger.update(mae=results_reduced['mae'], mse=results_reduced['mse'])

        # Trực quan hóa dự đoán lên ảnh để kiểm tra định tính vùng đúng/sai của mô hình.
        if vis_dir: 
            vis_mask = outputs_scores > threshold_values[default_threshold_index]
            vis_points = outputs_points[vis_mask].detach().cpu()
            points = [[point[0]*img_h, point[1]*img_w] for point in vis_points]     # recover to actual points
            split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            visualization(samples, targets, [points], vis_dir, split_map=split_map)
    
    # Thu thập thống kê từ mọi tiến trình phân tán rồi mới tính giá trị tổng hợp cuối cùng.
    metric_logger.synchronize_between_processes()
    if utils.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(mae_sums)
        torch.distributed.all_reduce(mse_sums)
        torch.distributed.all_reduce(sample_count)

    denom = max(sample_count.item(), 1.0)
    mae_values = (mae_sums / denom).detach().cpu().tolist()
    mse_values = torch.sqrt(mse_sums / denom).detach().cpu().tolist()
    best_index = min(
        range(len(threshold_values)),
        key=lambda idx: (
            mae_values[idx],
            mse_values[idx],
            abs(threshold_values[idx] - float(default_threshold)),
            threshold_values[idx],
        ),
    )
    results = {
        'mae': mae_values[best_index],
        'mse': mse_values[best_index],
        'threshold': threshold_values[best_index],
    }
    if len(threshold_values) > 1:
        results['threshold_candidates'] = threshold_values
        results['mae_by_threshold'] = mae_values
        results['mse_by_threshold'] = mse_values
    return results
