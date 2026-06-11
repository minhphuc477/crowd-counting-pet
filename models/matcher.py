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
        bs, num_queries = outputs["pred_logits"].shape[:2]
        sizes = [len(v["points"]) for v in targets]
        total_targets = sum(sizes)
        device = outputs["pred_logits"].device
        if total_targets == 0:
            empty = torch.empty(0, dtype=torch.int64, device=device)
            return [(empty, empty) for _ in range(bs)]

        # The matcher can be called inside AMP autocast. Keep the assignment
        # cost in fp32: fp16 cdist can overflow on 256/512 pixel distances.
        with _disable_autocast_for(device):
            # flatten to compute the cost matrices in a batch
            pred_logits = outputs["pred_logits"].detach().float()
            pred_points = outputs["pred_points"].detach().float()
            if not torch.isfinite(pred_logits).all():
                raise ValueError('HungarianMatcher received non-finite pred_logits')
            if not torch.isfinite(pred_points).all():
                raise ValueError('HungarianMatcher received non-finite pred_points')
            out_prob = pred_logits.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, 2]
            out_points = pred_points.flatten(0, 1)  # [batch_size * num_queries, 2]

            # concat target labels and points
            tgt_ids = torch.cat([v["labels"] for v in targets]).to(device=device)
            tgt_points = torch.cat([v["points"] for v in targets]).to(device=device, dtype=torch.float32)
            if not torch.isfinite(tgt_points).all():
                raise ValueError('HungarianMatcher received non-finite target points')

            # compute the classification cost, i.e., - prob[target class]
            cost_class = -out_prob[:, tgt_ids]

            # compute the L2 cost between points
            img_h, img_w = outputs['img_shape']
            out_points_abs = out_points.clone()
            out_points_abs[:, 0] *= float(img_h)
            out_points_abs[:, 1] *= float(img_w)
            cost_point = torch.cdist(out_points_abs, tgt_points, p=2)

            # final cost matrix
            C = self.cost_point * cost_point + self.cost_class * cost_class
            if not torch.isfinite(C).all():
                raise ValueError('HungarianMatcher produced a non-finite cost matrix')
            C = C.view(bs, num_queries, total_targets).cpu()

        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            if sizes[i] == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64, device=device),
                    torch.empty(0, dtype=torch.int64, device=device),
                ))
            else:
                src_idx, tgt_idx = linear_sum_assignment(c[i])
                indices.append((
                    torch.as_tensor(src_idx, dtype=torch.int64, device=device),
                    torch.as_tensor(tgt_idx, dtype=torch.int64, device=device),
                ))
        return indices


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_point=args.set_cost_point)
