"""
Modules to compute bipartite matching
"""
from contextlib import nullcontext

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def _disable_autocast_for(device):
    if device.type != 'cuda':
        return nullcontext()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda', enabled=False)
    return torch.cuda.amp.autocast(enabled=False)


def get_query_supervision_mask(outputs, target, batch_index=0):
    """Return queries whose centers fall inside the annotated image region."""
    points_queries = outputs.get("points_queries")
    if points_queries is None:
        return None
    if points_queries.ndim == 3:
        points_queries = points_queries[batch_index]
    supervision_mask = target.get("supervision_mask")
    if supervision_mask is None:
        return torch.ones(
            points_queries.shape[0],
            dtype=torch.bool,
            device=points_queries.device,
        )

    supervision_mask = supervision_mask.to(
        device=points_queries.device,
        dtype=torch.bool,
    )
    if supervision_mask.ndim != 2:
        raise ValueError(
            "target supervision_mask must have shape [H, W], "
            f"got {tuple(supervision_mask.shape)}"
        )
    img_h, img_w = outputs["img_shape"]
    query_y = torch.floor(points_queries[:, 0] * float(img_h)).long()
    query_x = torch.floor(points_queries[:, 1] * float(img_w)).long()
    valid = (
        (query_y >= 0)
        & (query_y < supervision_mask.shape[0])
        & (query_x >= 0)
        & (query_x < supervision_mask.shape[1])
    )
    result = torch.zeros_like(valid)
    if valid.any():
        result[valid] = supervision_mask[query_y[valid], query_x[valid]]
    return result


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network
    """
    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_point: This is the relative weight of the L2 error of the point coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs can't be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, **kwargs):
        """ 
        Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, 2] with the classification logits
                 "pred_points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, _ = outputs["pred_logits"].shape[:2]
        device = outputs["pred_logits"].device

        # The matcher can be called inside AMP autocast. Keep the assignment
        # cost in fp32: fp16 cdist can overflow on 256/512 pixel distances.
        indices = []
        with _disable_autocast_for(device):
            pred_logits = outputs["pred_logits"].detach().float()
            pred_points = outputs["pred_points"].detach().float()
            if not torch.isfinite(pred_logits).all():
                raise ValueError('HungarianMatcher received non-finite pred_logits')
            if not torch.isfinite(pred_points).all():
                raise ValueError('HungarianMatcher received non-finite pred_points')
            img_h, img_w = outputs['img_shape']
            for batch_index in range(bs):
                target = targets[batch_index]
                target_count = len(target["points"])
                valid_queries = get_query_supervision_mask(
                    outputs,
                    target,
                    batch_index,
                )
                valid_query_indices = torch.nonzero(
                    valid_queries,
                    as_tuple=False,
                ).flatten()
                if target_count == 0 or valid_query_indices.numel() == 0:
                    empty = torch.empty(0, dtype=torch.int64, device=device)
                    indices.append((empty, empty))
                    continue

                target_ids = target["labels"].to(device=device)
                target_points = target["points"].to(
                    device=device,
                    dtype=torch.float32,
                )
                if not torch.isfinite(target_points).all():
                    raise ValueError(
                        'HungarianMatcher received non-finite target points'
                    )
                probabilities = pred_logits[
                    batch_index,
                    valid_query_indices,
                ].softmax(-1)
                source_points = pred_points[
                    batch_index,
                    valid_query_indices,
                ].clone()
                source_points[:, 0] *= float(img_h)
                source_points[:, 1] *= float(img_w)
                cost_class = -probabilities[:, target_ids]
                cost_point = torch.cdist(
                    source_points,
                    target_points,
                    p=2,
                )
                cost = (
                    self.cost_point * cost_point
                    + self.cost_class * cost_class
                )
                if not torch.isfinite(cost).all():
                    raise ValueError(
                        'HungarianMatcher produced a non-finite cost matrix'
                    )
                source_local, target_index = linear_sum_assignment(
                    cost.cpu()
                )
                source_local = torch.as_tensor(
                    source_local,
                    dtype=torch.int64,
                    device=device,
                )
                target_index = torch.as_tensor(
                    target_index,
                    dtype=torch.int64,
                    device=device,
                )
                indices.append((
                    valid_query_indices[source_local],
                    target_index,
                ))
        return indices


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_point=args.set_cost_point)
