"""
PET model and criterion classes
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

from .matcher import build_matcher
from .backbones import *
from .transformer import *
from .position_encoding import build_position_encoding


def _parse_size_pair(value, default, name):
    if value is None or value == '':
        return tuple(default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.lower().replace('x', ',').split(',') if part.strip()]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f'{name} must be a pair formatted as "w,h", got {value!r}')
    pair = tuple(int(part) for part in parts)
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f'{name} values must be positive, got {pair}')
    return pair


def _parse_size_pair_list(value, default, name):
    if value is None or value == '':
        return [tuple(pair) for pair in default]
    if isinstance(value, str):
        chunks = [chunk.strip() for chunk in value.split(';') if chunk.strip()]
        if not chunks:
            return [tuple(pair) for pair in default]
        return [_parse_size_pair(chunk, None, name) for chunk in chunks]
    return [_parse_size_pair(pair, None, name) for pair in value]


class BasePETCount(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        hidden_dim = args.hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        class_prior_prob = float(getattr(args, 'class_prior_prob', -1.0))
        if class_prior_prob > 0:
            if not 0.0 < class_prior_prob < 1.0:
                raise ValueError('--class_prior_prob must be in (0, 1), or <=0 to disable')
            bias_value = math.log(class_prior_prob / (1.0 - class_prior_prob))
            nn.init.constant_(self.class_embed.bias, 0.0)
            self.class_embed.bias.data[1:] = bias_value
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'
    
    def points_queris_embed(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during training
        """
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:], device=src.device)
        shape = (image_shape + stride//2 -1) // stride
        shape_h, shape_w = int(shape[0].item()), int(shape[1].item())

        # generate point queries
        shift_x = ((torch.arange(0, shape_w, device=src.device) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape_h, device=src.device) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape

        # get point queries embedding
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]
        query_embed = query_embed.view(bs, c, h, w)

        # get point queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down,shift_x_down]
        query_feats = query_feats.view(bs, c, h, w)

        return query_embed, points_queries, query_feats
    
    def points_queris_embed_inference(self, samples, stride=8, src=None, **kwargs):
        """
        Generate point query embedding during inference
        """
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:], device=src.device)
        shape = (image_shape + stride//2 -1) // stride
        shape_h, shape_w = int(shape[0].item()), int(shape[1].item())

        # generate points queries
        shift_x = ((torch.arange(0, shape_w, device=src.device) + 0.5) * stride).long()
        shift_y = ((torch.arange(0, shape_h, device=src.device) + 0.5) * stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
        h, w = shift_x.shape

        # get points queries embedding 
        query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
        bs, c = query_embed.shape[:2]

        # get points queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
        query_feats = src[:, :, shift_y_down, shift_x_down]
        
        # window-rize
        query_embed = query_embed.reshape(bs, c, h, w)
        points_queries = points_queries.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0)
        query_feats = query_feats.reshape(bs, c, h, w)

        dec_win_w, dec_win_h = kwargs['dec_win_size']
        query_embed_win = window_partition(query_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)
        points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)
        query_feats_win = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)

        # Prune inactive windows before the decoder (matches original PET).
        div = kwargs['div']
        div_win = window_partition(div.unsqueeze(1), window_size_h=dec_win_h, window_size_w=dec_win_w)
        valid_div = (div_win > 0.5).sum(dim=0)[:, 0]
        v_idx = valid_div > 0
        query_embed_win = query_embed_win[:, v_idx]
        query_feats_win = query_feats_win[:, v_idx]
        points_queries_win = points_queries_win[:, v_idx].reshape(-1, 2)

        return query_embed_win, points_queries_win, query_feats_win, v_idx
    
    def get_point_query(self, samples, features, **kwargs):
        """
        Generate point query
        """
        src, _ = features[self.feat_name].decompose()

        # generate points queries and position embedding
        if 'train' in kwargs:
            query_embed, points_queries, query_feats = self.points_queris_embed(samples, self.pq_stride, src, **kwargs)
            query_embed = query_embed.flatten(2).permute(2,0,1) # NxCxHxW --> (HW)xNxC
            v_idx = None
        else:
            query_embed, points_queries, query_feats, v_idx = self.points_queris_embed_inference(samples, self.pq_stride, src, **kwargs)

        out = (query_embed, points_queries, query_feats, v_idx)
        return out
    
    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """
        outputs_class = self.class_embed(hs)
        # normalize to 0~1
        outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0

        # normalize point-query coordinates
        # Clone to avoid in-place mutation: predict() is called once for sparse
        # and once for dense; without clone the second call gets pre-normalized values.
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape
        points_queries = points_queries.float().to(samples.tensors.device).clone()
        points_queries[:, 0] /= img_h
        points_queries[:, 1] /= img_w

        # rescale offset range during testing
        if 'test' in kwargs:
            outputs_offsets[...,0] /= (img_h / 256)
            outputs_offsets[...,1] /= (img_w / 256)

        outputs_points = outputs_offsets[-1] + points_queries
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_points, 'img_shape': img_shape, 'pred_offsets': outputs_offsets[-1]}
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

    def forward(self, samples, features, context_info, **kwargs):
        encode_src, src_pos_embed, mask = context_info

        # get points queries for transformer
        pqs = self.get_point_query(samples, features, **kwargs)
        
        # point querying
        kwargs['pq_stride'] = self.pq_stride
        hs = self.transformer(encode_src, src_pos_embed, mask, pqs, img_shape=samples.tensors.shape[-2:], **kwargs)

        # prediction
        points_queries = pqs[1]
        outputs = self.predict(samples, points_queries, hs, **kwargs)
        return outputs
    

def _make_group_norm(num_channels):
    groups = min(32, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class QuadContextMixer(nn.Module):
    """Lightweight quadtree-aware residual context mixer for PET encoder features.

    This is intentionally not a Mamba/selective-scan block. It borrows the
    useful principle that each token should see both fine local context and
    coarser quadtree parent context, while keeping PET's decoder unchanged.
    """
    def __init__(self, hidden_dim, mid_dim=128, levels=2, shift=1, activation='gelu'):
        super().__init__()
        self.levels = max(1, int(levels))
        self.shift = max(0, int(shift))
        mid_dim = max(1, int(mid_dim))

        if activation == 'gelu':
            act_gate = nn.GELU()
            act_local = nn.GELU()
        elif activation == 'relu':
            act_gate = nn.ReLU(inplace=True)
            act_local = nn.ReLU(inplace=True)
        else:
            raise ValueError(f'Unsupported quad context activation: {activation}. Use "gelu" or "relu".')

        self.local_context = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            _make_group_norm(hidden_dim),
            act_local,
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
        )
        self.coarse_context = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim, mid_dim, 1, bias=False),
            _make_group_norm(mid_dim),
            act_gate,
            nn.Conv2d(mid_dim, 1, 1),
        )
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _pool_to_parent(self, x, pool_size):
        h, w = x.shape[-2:]
        kernel_h = min(int(pool_size), h)
        kernel_w = min(int(pool_size), w)
        pooled = F.avg_pool2d(
            x,
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w),
            ceil_mode=True,
            count_include_pad=False,
        )
        return F.interpolate(pooled, size=(h, w), mode='nearest')

    def _quadtree_parent_context(self, x):
        contexts = []
        for level in range(self.levels):
            contexts.append(self._pool_to_parent(x, 2 ** (level + 1)))
        return torch.stack(contexts, dim=0).mean(dim=0)

    def _shifted_parent_context(self, x):
        base_context = self._quadtree_parent_context(x)
        if self.shift <= 0:
            return base_context

        shift_h = min(self.shift, max(1, x.shape[-2] - 1))
        shift_w = min(self.shift, max(1, x.shape[-1] - 1))
        contexts = [base_context]
        for shift in ((shift_h, shift_w), (shift_h, -shift_w), (-shift_h, shift_w), (-shift_h, -shift_w)):
            shifted = torch.roll(x, shifts=shift, dims=(2, 3))
            shifted_context = self._quadtree_parent_context(shifted)
            contexts.append(torch.roll(shifted_context, shifts=(-shift[0], -shift[1]), dims=(2, 3)))
        return torch.stack(contexts, dim=0).mean(dim=0)

    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        local_context = self.local_context(x)
        parent_context = self.coarse_context(self._shifted_parent_context(x))
        mixed_context = gate * local_context + (1.0 - gate) * parent_context
        return x + self.out_proj(mixed_context)


class QuadtreeSplitter(nn.Module):
    def __init__(self, hidden_dim, context_h, context_w, head='pool', mid_dim=128, activation='gelu'):
        super().__init__()
        context_h = max(1, int(context_h))
        context_w = max(1, int(context_w))
        if activation == 'gelu':
            act1 = nn.GELU()
            act2 = nn.GELU()
        elif activation == 'relu':
            act1 = nn.ReLU(inplace=True)
            act2 = nn.ReLU(inplace=True)
        else:
            raise ValueError(f'Unsupported splitter activation: {activation}. Use "gelu" or "relu".')
        if head == 'pool':
            self.net = nn.Sequential(
                nn.AvgPool2d((context_h, context_w), stride=(context_h, context_w)),
                nn.Conv2d(hidden_dim, 1, 1),
                nn.Sigmoid(),
            )
        elif head == 'conv':
            mid_dim = max(1, int(mid_dim))
            self.pool_logits = nn.Sequential(
                nn.AvgPool2d((context_h, context_w), stride=(context_h, context_w)),
                nn.Conv2d(hidden_dim, 1, 1),
            )
            self.residual_logits = nn.Sequential(
                nn.Conv2d(hidden_dim, mid_dim, 3, padding=1, bias=False),
                _make_group_norm(mid_dim),
                act1,
                nn.Conv2d(mid_dim, mid_dim, 3, padding=2, dilation=2, bias=False),
                _make_group_norm(mid_dim),
                act2,
                nn.AvgPool2d((context_h, context_w), stride=(context_h, context_w)),
                nn.Conv2d(mid_dim, 1, 1),
            )
            nn.init.zeros_(self.residual_logits[-1].weight)
            nn.init.zeros_(self.residual_logits[-1].bias)
        else:
            raise ValueError(f'Unsupported splitter head: {head}. Use "pool" or "conv".')

    def forward(self, x):
        if hasattr(self, 'pool_logits'):
            return torch.sigmoid(self.pool_logits(x) + self.residual_logits(x))
        return self.net(x)


class PET(nn.Module):
    """ 
    Point quEry Transformer
    """
    def __init__(self, backbone, num_classes, args=None):
        super().__init__()
        self.backbone = backbone
        
        # positional embedding
        self.pos_embed = build_position_encoding(args)

        # feature projection
        hidden_dim = args.hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            ]
        )

        # context encoder
        self.encode_feats = '8x'
        enc_win_list = _parse_size_pair_list(
            getattr(args, 'enc_win_sizes', ''),
            [(32, 16), (32, 16), (16, 8), (16, 8)],
            'enc_win_sizes',
        )  # encoder window size
        args.enc_layers = len(enc_win_list)
        self.context_encoder = build_encoder(args, enc_win_list=enc_win_list)

        # quadtree splitter
        context_patch = _parse_size_pair(getattr(args, 'context_patch_size', ''), (128, 64), 'context_patch_size')
        context_w, context_h = context_patch[0]//int(self.encode_feats[:-1]), context_patch[1]//int(self.encode_feats[:-1])
        self.quadtree_splitter = QuadtreeSplitter(
            hidden_dim,
            context_h,
            context_w,
            head=getattr(args, 'splitter_head', 'pool'),
            mid_dim=getattr(args, 'splitter_hidden_dim', 128),
            activation=getattr(args, 'splitter_activation', 'gelu'),
        )

        # point-query quadtree
        args.sparse_stride, args.dense_stride = 8, 4    # point-query stride
        transformer = build_decoder(args)
        self.quadtree_sparse = BasePETCount(backbone, num_classes, quadtree_layer='sparse', args=args, transformer=transformer)
        self.quadtree_dense = BasePETCount(backbone, num_classes, quadtree_layer='dense', args=args, transformer=transformer)
        self.ifi_loss_coef = float(getattr(args, 'ifi_loss_coef', 0.0))
        if self.ifi_loss_coef > 0:
            self.ifi_cls_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.ifi_coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        else:
            self.ifi_cls_embed = None
            self.ifi_coord_embed = None
        self.warmup_epochs = int(getattr(args, 'warmup_epochs', 5))
        self.quadtree_loss_coef = float(getattr(args, 'quadtree_loss_coef', 0.1))
        self.quadtree_prior_coef = float(getattr(args, 'quadtree_prior_coef', 0.025))
        self.split_count_threshold = int(getattr(args, 'split_count_threshold', 2))
        self.split_pos_weight = float(getattr(args, 'split_pos_weight', 1.0))
        self.negative_loss_coef = float(getattr(args, 'negative_loss_coef', 0.1))
        self.split_threshold = float(getattr(args, 'split_threshold', -1.0))
        self.split_threshold_quantile = float(getattr(args, 'split_threshold_quantile', 0.55))
        self.score_threshold = float(getattr(args, 'score_threshold', 0.5))
        self.eval_nms_radius = float(getattr(args, 'eval_nms_radius', 0.0))
        self.eval_branch_gate = getattr(args, 'eval_branch_gate', 'none')
        if self.eval_branch_gate not in ('none', 'query', 'pred'):
            raise ValueError('eval_branch_gate must be one of "none", "query", or "pred"')
        self.eval_soft_split_gate = getattr(args, 'eval_soft_split_gate', 'none')
        if self.eval_soft_split_gate not in ('none', 'query', 'pred'):
            raise ValueError('eval_soft_split_gate must be one of "none", "query", or "pred"')
        self.pet_loss_variant = getattr(args, 'pet_loss_variant', 'paper')
        self.split_loss_variant = getattr(args, 'split_loss_variant', 'auto')
        if self.split_loss_variant == 'auto':
            self.split_loss_variant = 'paper' if self.pet_loss_variant == 'paper' else 'paper_gt'
        if self.split_loss_variant not in ('paper', 'gt', 'paper_gt'):
            raise ValueError('split_loss_variant must be one of "auto", "paper", "gt", or "paper_gt"')
        self.count_loss_coef = float(getattr(args, 'count_loss_coef', 0.0))
        self.count_loss_gate = getattr(args, 'count_loss_gate', 'detach')
        if self.count_loss_gate not in ('detach', 'soft', 'hard'):
            raise ValueError('count_loss_gate must be one of "detach", "soft", or "hard"')
        self.count_loss_type = getattr(args, 'count_loss_type', 'log_l1')
        if self.count_loss_type not in ('log_l1', 'l1', 'smooth_l1'):
            raise ValueError('count_loss_type must be one of "log_l1", "l1", or "smooth_l1"')
        self.count_loss_start_epoch = int(getattr(args, 'count_loss_start_epoch', -1))
        self.region_count_loss_coef = float(getattr(args, 'region_count_loss_coef', 0.0))
        self.region_count_grid = max(1, int(getattr(args, 'region_count_grid', 4)))
        self.region_count_gate = getattr(args, 'region_count_gate', 'detach')
        if self.region_count_gate not in ('none', 'detach', 'soft', 'hard'):
            raise ValueError('region_count_gate must be one of "none", "detach", "soft", or "hard"')
        self.region_count_type = getattr(args, 'region_count_type', 'log_l1')
        if self.region_count_type not in ('log_l1', 'l1', 'smooth_l1'):
            raise ValueError('region_count_type must be one of "log_l1", "l1", or "smooth_l1"')
        self.region_count_start_epoch = int(getattr(args, 'region_count_start_epoch', -1))
        self.region_count_end_epoch = int(getattr(args, 'region_count_end_epoch', -1))
        self.bayesian_loss_coef = float(getattr(args, 'bayesian_loss_coef', 0.0))
        self.bayesian_sigma = float(getattr(args, 'bayesian_sigma', 8.0))
        self.bayesian_bg_coef = float(getattr(args, 'bayesian_bg_coef', 0.05))
        self.bayesian_loss_gate = getattr(args, 'bayesian_loss_gate', 'detach')
        if self.bayesian_loss_gate not in ('none', 'detach', 'soft', 'hard'):
            raise ValueError('bayesian_loss_gate must be one of "none", "detach", "soft", or "hard"')
        self.bayesian_start_epoch = int(getattr(args, 'bayesian_start_epoch', -1))
        self.bayesian_end_epoch = int(getattr(args, 'bayesian_end_epoch', -1))
        self.apg_loss_coef = float(getattr(args, 'apg_loss_coef', 0.0))
        self.apg_pos_k = max(1, int(getattr(args, 'apg_pos_k', 1)))
        self.apg_point_coef = float(getattr(args, 'apg_point_coef', 5.0))
        self.apg_bg_coef = float(getattr(args, 'apg_bg_coef', 0.0))
        self.apg_bg_k = max(0, int(getattr(args, 'apg_bg_k', 0)))
        self.apg_bg_min_dist = max(0.0, float(getattr(args, 'apg_bg_min_dist', 12.0)))
        self.apg_start_epoch = int(getattr(args, 'apg_start_epoch', 0))
        self.apg_warmup_epochs = max(0, int(getattr(args, 'apg_warmup_epochs', 0)))
        self.apg_end_epoch = int(getattr(args, 'apg_end_epoch', -1))
        self.apg_contrastive_coef = float(getattr(args, 'apg_contrastive_coef', 0.0))
        self.apg_neg_k = max(0, int(getattr(args, 'apg_neg_k', 4)))
        self.apg_margin = float(getattr(args, 'apg_margin', 1.0))
        self.apg_consistency_coef = float(getattr(args, 'apg_consistency_coef', 0.0))
        self.apg_consistency_k = max(1, int(getattr(args, 'apg_consistency_k', 4)))
        self.apg_consistency_sigma = float(getattr(args, 'apg_consistency_sigma', 8.0))
        self.apg_soft_loss_coef = float(getattr(args, 'apg_soft_loss_coef', 0.0))
        self.apg_soft_pos_k = max(1, int(getattr(args, 'apg_soft_pos_k', 4)))
        self.apg_soft_sigma = float(getattr(args, 'apg_soft_sigma', 6.0))
        self.apg_soft_point_coef = float(getattr(args, 'apg_soft_point_coef', 2.0))
        self.ifi_point_coef = float(getattr(args, 'ifi_point_coef', 1.0))
        self.ifi_neg_k = max(0, int(getattr(args, 'ifi_neg_k', 4)))
        self.ifi_neg_radius = float(getattr(args, 'ifi_neg_radius', 12.0))
        self.ifi_neg_min_dist = float(getattr(args, 'ifi_neg_min_dist', 4.0))
        self.ifi_start_epoch = int(getattr(args, 'ifi_start_epoch', 0))
        self.ifi_end_epoch = int(getattr(args, 'ifi_end_epoch', -1))
        self.qd_apg_loss_coef = float(getattr(args, 'qd_apg_loss_coef', 0.0))
        self.qd_apg_point_coef = float(getattr(args, 'qd_apg_point_coef', 5.0))
        self.qd_apg_suppress_coef = float(getattr(args, 'qd_apg_suppress_coef', 0.5))
        self.qd_apg_start_epoch = int(getattr(args, 'qd_apg_start_epoch', 0))
        self.qd_apg_end_epoch = int(getattr(args, 'qd_apg_end_epoch', -1))
        self.qd_apg_route_source = getattr(args, 'qd_apg_route_source', 'gt_count')
        if self.qd_apg_route_source not in ('gt_count', 'split_map'):
            raise ValueError('qd_apg_route_source must be one of "gt_count" or "split_map"')
        self.sparse_dec_win_size = _parse_size_pair(
            getattr(args, 'sparse_dec_win_size', ''),
            (16, 8),
            'sparse_dec_win_size',
        )
        self.dense_dec_win_size = _parse_size_pair(
            getattr(args, 'dense_dec_win_size', ''),
            (8, 4),
            'dense_dec_win_size',
        )
        quad_context = getattr(args, 'quad_context_mixer', 'none')
        if quad_context == 'none':
            self.quad_context_mixer = nn.Identity()
        elif quad_context == 'lite':
            self.quad_context_mixer = QuadContextMixer(
                hidden_dim,
                mid_dim=getattr(args, 'quad_context_mid_dim', 128),
                levels=getattr(args, 'quad_context_levels', 2),
                shift=getattr(args, 'quad_context_shift', 1),
                activation=getattr(args, 'quad_context_activation', 'gelu'),
            )
        else:
            raise ValueError(f'Unsupported quad_context_mixer: {quad_context}. Use "none" or "lite".')

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = self.warmup_epochs

        # compute loss
        if epoch >= warmup_ep:
            loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_sparse'])
            loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_dense'])
        else:
            loss_dict_sparse = criterion(output_sparse, targets)
            loss_dict_dense = criterion(output_dense, targets)

        # sparse point queries loss
        loss_dict_sparse = {k+'_sp':v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k+'_sp':v for k,v in weight_dict.items()}
        loss_pq_sparse = sum(loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # dense point queries loss
        loss_dict_dense = {k+'_ds':v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k+'_ds':v for k,v in weight_dict.items()}
        loss_pq_dense = sum(loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)
    
        # point queries loss
        losses = loss_pq_sparse + loss_pq_dense 

        # update loss dict and weight dict
        loss_dict = dict()
        loss_dict.update(loss_dict_sparse)
        loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        weight_dict.update(weight_dict_sparse)
        weight_dict.update(weight_dict_dense)

        # quadtree splitter loss
        den = torch.stack([target['density'].reshape(()) for target in targets]).to(outputs['split_map_raw'].device)
        bs = len(den)
        ds_idx = den < 2 * self.quadtree_sparse.pq_stride   # dense regions index
        ds_div = outputs['split_map_raw'][ds_idx]
        sp_div = 1 - outputs['split_map_raw']

        # constrain sparse regions
        loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()

        # constrain dense regions
        if ds_idx.any():
            ds_num = ds_div.shape[0]
            loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        else:
            loss_split_ds = outputs['split_map_raw'].sum() * 0.0

        loss_split_prior = loss_split_sp + loss_split_ds
        if self.count_loss_coef > 0:
            loss_count = self.compute_count_loss(outputs, targets)
            count_start_epoch = warmup_ep if self.count_loss_start_epoch < 0 else self.count_loss_start_epoch
            weight_count = self.count_loss_coef if epoch >= count_start_epoch else 0.0
            loss_dict['loss_count'] = loss_count
            weight_dict['loss_count'] = weight_count
            losses += loss_count * weight_count
        if self.region_count_loss_coef > 0:
            region_start_epoch = warmup_ep if self.region_count_start_epoch < 0 else self.region_count_start_epoch
            region_active = epoch >= region_start_epoch and (
                self.region_count_end_epoch < 0 or epoch <= self.region_count_end_epoch
            )
            if region_active:
                loss_region_count = self.compute_region_count_loss(outputs, targets)
            else:
                loss_region_count = outputs['split_map_raw'].sum() * 0.0
            loss_dict['loss_region_count'] = loss_region_count
            weight_dict['loss_region_count'] = self.region_count_loss_coef
            losses += loss_region_count * self.region_count_loss_coef
        if self.bayesian_loss_coef > 0:
            bayesian_start_epoch = warmup_ep if self.bayesian_start_epoch < 0 else self.bayesian_start_epoch
            bayesian_active = epoch >= bayesian_start_epoch and (
                self.bayesian_end_epoch < 0 or epoch <= self.bayesian_end_epoch
            )
            if bayesian_active:
                loss_bayesian = self.compute_bayesian_point_loss(outputs, targets)
            else:
                loss_bayesian = outputs['split_map_raw'].sum() * 0.0
            loss_dict['loss_bayesian'] = loss_bayesian
            weight_dict['loss_bayesian'] = self.bayesian_loss_coef
            losses += loss_bayesian * self.bayesian_loss_coef
        if self.apg_loss_coef > 0:
            apg_active = epoch >= self.apg_start_epoch and (
                self.apg_end_epoch < 0 or epoch <= self.apg_end_epoch
            )
            if apg_active and self.apg_warmup_epochs > 0:
                apg_weight = self.apg_loss_coef * min(
                    1.0,
                    float(epoch - self.apg_start_epoch + 1) / float(self.apg_warmup_epochs),
                )
            else:
                apg_weight = self.apg_loss_coef
            if apg_active:
                loss_apg_sparse = self.compute_apg_loss(output_sparse, targets)
                loss_apg_dense = self.compute_apg_loss(output_dense, targets)
            else:
                loss_apg_sparse = output_sparse['pred_logits'].sum() * 0.0
                loss_apg_dense = output_dense['pred_logits'].sum() * 0.0
            loss_dict['loss_apg_sp'] = loss_apg_sparse
            loss_dict['loss_apg_ds'] = loss_apg_dense
            weight_dict['loss_apg_sp'] = apg_weight
            weight_dict['loss_apg_ds'] = apg_weight
            losses += (loss_apg_sparse + loss_apg_dense) * apg_weight
        if self.apg_soft_loss_coef > 0:
            apg_active = epoch >= self.apg_start_epoch and (
                self.apg_end_epoch < 0 or epoch <= self.apg_end_epoch
            )
            if apg_active:
                loss_apg_soft_sparse = self.compute_soft_apg_loss(output_sparse, targets)
                loss_apg_soft_dense = self.compute_soft_apg_loss(output_dense, targets)
            else:
                loss_apg_soft_sparse = output_sparse['pred_logits'].sum() * 0.0
                loss_apg_soft_dense = output_dense['pred_logits'].sum() * 0.0
            loss_dict['loss_apg_soft_sp'] = loss_apg_soft_sparse
            loss_dict['loss_apg_soft_ds'] = loss_apg_soft_dense
            weight_dict['loss_apg_soft_sp'] = self.apg_soft_loss_coef
            weight_dict['loss_apg_soft_ds'] = self.apg_soft_loss_coef
            losses += (loss_apg_soft_sparse + loss_apg_soft_dense) * self.apg_soft_loss_coef
        if self.ifi_loss_coef > 0:
            ifi_active = epoch >= self.ifi_start_epoch and (
                self.ifi_end_epoch < 0 or epoch <= self.ifi_end_epoch
            )
            if ifi_active:
                loss_ifi = self.compute_ifi_apg_loss(outputs, targets, samples)
            else:
                loss_ifi = outputs['split_map_raw'].sum() * 0.0
            loss_dict['loss_ifi'] = loss_ifi
            weight_dict['loss_ifi'] = self.ifi_loss_coef
            losses += loss_ifi * self.ifi_loss_coef
        if self.qd_apg_loss_coef > 0:
            qd_apg_active = epoch >= self.qd_apg_start_epoch and (
                self.qd_apg_end_epoch < 0 or epoch <= self.qd_apg_end_epoch
            )
            if qd_apg_active:
                loss_qd_apg = self.compute_qd_apg_loss(outputs, targets)
            else:
                loss_qd_apg = output_sparse['pred_logits'].sum() * 0.0
            loss_dict['loss_qd_apg'] = loss_qd_apg
            weight_dict['loss_qd_apg'] = self.qd_apg_loss_coef
            losses += loss_qd_apg * self.qd_apg_loss_coef

        if self.split_loss_variant == 'paper':
            weight_split = self.quadtree_loss_coef if epoch >= warmup_ep else 0.0
            loss_dict['loss_split'] = loss_split_prior
            weight_dict['loss_split'] = weight_split
            losses += loss_split_prior * weight_split
            return {'loss_dict':loss_dict, 'weight_dict':weight_dict, 'losses':losses}

        split_target = self.build_split_targets(targets, outputs['split_map_raw'], samples.tensors.shape[-2:])
        loss_split_quality = self.balanced_binary_loss(
            outputs['split_map_raw'],
            split_target,
            pos_weight=self.split_pos_weight,
            neg_weight=self.negative_loss_coef,
        )
        weight_split_quality = self.quadtree_loss_coef if epoch >= warmup_ep else 0.0
        loss_dict['loss_split_gt'] = loss_split_quality
        weight_dict['loss_split_gt'] = weight_split_quality

        # final loss
        losses += loss_split_quality * weight_split_quality
        if self.split_loss_variant == 'paper_gt':
            weight_split_prior = self.quadtree_prior_coef if epoch >= warmup_ep else 0.0
            loss_dict['loss_split_prior'] = loss_split_prior
            weight_dict['loss_split_prior'] = weight_split_prior
            losses += loss_split_prior * weight_split_prior
        return {'loss_dict':loss_dict, 'weight_dict':weight_dict, 'losses':losses}

    def compute_apg_loss(self, output, targets):
        """Auxiliary Point Guidance for PET point queries.

        APGCC's full method adds auxiliary proposal guidance to stabilize
        point-based matching. PET already owns a fixed point-query grid, so the
        compatible low-risk version is to directly supervise the nearest grid
        query/queries for each GT point as positive proposals.
        """
        logits = output['pred_logits']
        pred_points = output['pred_points']
        point_queries = output.get('points_queries')
        if point_queries is None:
            return logits.sum() * 0.0

        device = logits.device
        img_h, img_w = output['img_shape']
        query_abs = point_queries.to(device=device, dtype=pred_points.dtype).clone()
        query_abs[:, 0] *= img_h
        query_abs[:, 1] *= img_w

        cls_losses = []
        point_losses = []
        bg_losses = []
        contrastive_losses = []
        consistency_losses = []
        for batch_idx, target in enumerate(targets):
            gt_points = target['points'].to(device=device, dtype=pred_points.dtype)
            if gt_points.numel() == 0:
                continue
            query_dist = torch.cdist(gt_points, query_abs, p=2)
            k = min(self.apg_pos_k, query_abs.shape[0])
            nearest = query_dist.topk(k, largest=False).indices.reshape(-1)
            nearest = torch.unique(nearest)

            cls_target = torch.ones(nearest.shape[0], dtype=torch.long, device=device)
            cls_losses.append(F.cross_entropy(logits[batch_idx, nearest], cls_target, reduction='mean'))

            if self.apg_bg_coef > 0 and self.apg_bg_k > 0:
                min_query_dist = query_dist.min(dim=0).values
                positive_mask = torch.zeros(query_abs.shape[0], dtype=torch.bool, device=device)
                positive_mask[nearest] = True
                bg_mask = (~positive_mask) & (min_query_dist >= self.apg_bg_min_dist)
                bg_candidates = torch.nonzero(bg_mask, as_tuple=False).flatten()
                if bg_candidates.numel() > 0:
                    bg_count = min(bg_candidates.numel(), max(1, int(gt_points.shape[0]) * self.apg_bg_k))
                    # Use the nearest safe background queries: they are the
                    # most useful local negatives but still outside the GT zone.
                    _, order = torch.topk(min_query_dist[bg_candidates], k=bg_count, largest=False)
                    bg_idx = bg_candidates[order]
                    bg_target = torch.zeros(bg_idx.shape[0], dtype=torch.long, device=device)
                    bg_losses.append(F.cross_entropy(logits[batch_idx, bg_idx], bg_target, reduction='mean'))

            gt_for_queries = gt_points[torch.cdist(query_abs[nearest], gt_points, p=2).argmin(dim=1)]
            gt_norm = gt_for_queries.clone()
            gt_norm[:, 0] /= img_h
            gt_norm[:, 1] /= img_w
            point_losses.append(
                F.smooth_l1_loss(pred_points[batch_idx, nearest], gt_norm, reduction='none').sum(dim=-1).mean()
            )
            if self.apg_contrastive_coef > 0 and self.apg_neg_k > 0:
                candidate_k = min(k + self.apg_neg_k, query_abs.shape[0])
                candidates = torch.unique(query_dist.topk(candidate_k, largest=False).indices.reshape(-1))
                positive_mask = torch.zeros(query_abs.shape[0], dtype=torch.bool, device=device)
                positive_mask[nearest] = True
                negatives = candidates[~positive_mask[candidates]]
                if negatives.numel() > 0:
                    person_margin_logits = logits[batch_idx, :, 1] - logits[batch_idx, :, 0]
                    pos_score = person_margin_logits[nearest].mean()
                    neg_scores = person_margin_logits[negatives]
                    contrastive_losses.append(F.relu(self.apg_margin - pos_score + neg_scores).mean())
            if self.apg_consistency_coef > 0:
                local_k = min(self.apg_consistency_k, query_abs.shape[0])
                local_dist, local_idx = query_dist.topk(local_k, largest=False)
                sigma = max(float(self.apg_consistency_sigma), 1e-6)
                local_weights = torch.exp(-0.5 * (local_dist / sigma) ** 2)
                local_weights = local_weights / (local_weights.sum(dim=1, keepdim=True) + 1e-6)
                for gt_idx in range(gt_points.shape[0]):
                    idx_local = local_idx[gt_idx]
                    pred_local = pred_points[batch_idx, idx_local]
                    gt_norm = gt_points[gt_idx].clone()
                    gt_norm[0] /= img_h
                    gt_norm[1] /= img_w
                    weights = local_weights[gt_idx].unsqueeze(-1)
                    mean_pred = (pred_local * weights).sum(dim=0)
                    mean_loss = F.smooth_l1_loss(mean_pred, gt_norm, reduction='sum')
                    var_loss = ((pred_local - mean_pred).pow(2).sum(dim=-1) * local_weights[gt_idx]).sum()
                    consistency_losses.append(mean_loss + var_loss)

        if not cls_losses:
            return logits.sum() * 0.0
        loss_cls = torch.stack(cls_losses).mean()
        loss_point = torch.stack(point_losses).mean()
        loss = loss_cls + self.apg_point_coef * loss_point
        if bg_losses:
            loss = loss + self.apg_bg_coef * torch.stack(bg_losses).mean()
        if contrastive_losses:
            loss = loss + self.apg_contrastive_coef * torch.stack(contrastive_losses).mean()
        if consistency_losses:
            loss = loss + self.apg_consistency_coef * torch.stack(consistency_losses).mean()
        return loss

    def compute_soft_apg_loss(self, output, targets):
        """Gaussian APG on PET's actual point-query logits and offsets."""
        logits = output['pred_logits']
        pred_points = output['pred_points']
        point_queries = output.get('points_queries')
        if point_queries is None:
            return logits.sum() * 0.0

        device = logits.device
        dtype = pred_points.dtype
        img_h, img_w = output['img_shape']
        query_abs = point_queries.to(device=device, dtype=dtype).clone()
        query_abs[:, 0] *= img_h
        query_abs[:, 1] *= img_w

        sigma = max(float(self.apg_soft_sigma), 1e-6)
        cls_losses = []
        point_losses = []
        for batch_idx, target in enumerate(targets):
            gt_points = target['points'].to(device=device, dtype=dtype)
            if gt_points.numel() == 0 or query_abs.numel() == 0:
                continue
            k = min(self.apg_soft_pos_k, query_abs.shape[0])
            query_dist = torch.cdist(gt_points, query_abs, p=2)
            local_idx = torch.unique(query_dist.topk(k, largest=False).indices.reshape(-1))
            if local_idx.numel() == 0:
                continue
            local_query_abs = query_abs[local_idx]
            local_dist, local_gt_idx = torch.cdist(local_query_abs, gt_points, p=2).min(dim=1)
            score_target = torch.exp(-0.5 * (local_dist / sigma).pow(2)).clamp(0.0, 1.0)
            logit_margin = logits[batch_idx, local_idx, 1] - logits[batch_idx, local_idx, 0]
            cls_losses.append(
                F.binary_cross_entropy_with_logits(logit_margin, score_target, reduction='mean')
            )

            gt_norm = gt_points[local_gt_idx].clone()
            gt_norm[:, 0] /= img_h
            gt_norm[:, 1] /= img_w
            point_raw = F.smooth_l1_loss(
                pred_points[batch_idx, local_idx],
                gt_norm,
                reduction='none',
            ).sum(dim=-1)
            point_losses.append((point_raw * score_target).sum() / (score_target.sum() + 1e-6))

        if not cls_losses:
            return logits.sum() * 0.0
        loss = torch.stack(cls_losses).mean()
        if point_losses:
            loss = loss + self.apg_soft_point_coef * torch.stack(point_losses).mean()
        return loss

    def _sample_ifi_features(self, encode_src, batch_idx, points_abs, img_h, img_w):
        if points_abs.numel() == 0:
            return encode_src.new_zeros((0, encode_src.shape[1]))
        grid = points_abs.to(device=encode_src.device, dtype=encode_src.dtype).clone()
        grid_x = (grid[:, 1] + 0.5) / max(float(img_w), 1.0) * 2.0 - 1.0
        grid_y = (grid[:, 0] + 0.5) / max(float(img_h), 1.0) * 2.0 - 1.0
        sample_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
        feats = F.grid_sample(
            encode_src[batch_idx:batch_idx + 1],
            sample_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )
        return feats.squeeze(0).squeeze(-1).transpose(0, 1)

    def _build_ifi_negatives(self, gt_points, img_h, img_w):
        if self.ifi_neg_k <= 0 or gt_points.numel() == 0:
            return gt_points.new_zeros((0, 2))
        offsets = []
        radius = max(float(self.ifi_neg_radius), 1.0)
        for neg_idx in range(self.ifi_neg_k):
            angle = 2.0 * math.pi * neg_idx / max(self.ifi_neg_k, 1)
            offsets.append(gt_points.new_tensor([math.sin(angle) * radius, math.cos(angle) * radius]))
        offset_tensor = torch.stack(offsets, dim=0)
        neg_points = (gt_points[:, None, :] + offset_tensor[None, :, :]).reshape(-1, 2)
        neg_points[:, 0].clamp_(0, max(float(img_h) - 1.0, 0.0))
        neg_points[:, 1].clamp_(0, max(float(img_w) - 1.0, 0.0))
        if gt_points.shape[0] > 0 and self.ifi_neg_min_dist > 0:
            min_dist = torch.cdist(neg_points, gt_points, p=2).min(dim=1)[0]
            neg_points = neg_points[min_dist >= self.ifi_neg_min_dist]
        return neg_points

    def compute_ifi_apg_loss(self, outputs, targets, samples):
        """Interpolated Feature Guidance for APG.

        APG-lite supervises nearest fixed grid queries. IFI-lite complements it
        by sampling PET's encoded feature map at arbitrary GT and local-negative
        positions, matching APGCC's core idea without changing PET inference.
        """
        encode_src = outputs.get('encode_src')
        if encode_src is None:
            return outputs['split_map_raw'].sum() * 0.0
        img_h, img_w = samples.tensors.shape[-2:]
        cls_losses = []
        point_losses = []
        for batch_idx, target in enumerate(targets):
            gt_points = target['points'].to(device=encode_src.device, dtype=encode_src.dtype)
            if gt_points.numel() == 0:
                continue
            pos_feats = self._sample_ifi_features(encode_src, batch_idx, gt_points, img_h, img_w)
            pos_logits = self.ifi_cls_embed(pos_feats)
            pos_target = torch.ones(pos_logits.shape[0], dtype=torch.long, device=encode_src.device)
            cls_losses.append(F.cross_entropy(pos_logits, pos_target, reduction='mean'))

            pos_offsets = (self.ifi_coord_embed(pos_feats).sigmoid() - 0.5) * 2.0
            point_losses.append(F.smooth_l1_loss(pos_offsets, torch.zeros_like(pos_offsets), reduction='none').sum(dim=-1).mean())

            neg_points = self._build_ifi_negatives(gt_points, img_h, img_w)
            if neg_points.numel() > 0:
                neg_feats = self._sample_ifi_features(encode_src, batch_idx, neg_points, img_h, img_w)
                neg_logits = self.ifi_cls_embed(neg_feats)
                neg_target = torch.zeros(neg_logits.shape[0], dtype=torch.long, device=encode_src.device)
                cls_losses.append(F.cross_entropy(neg_logits, neg_target, reduction='mean'))

        if not cls_losses:
            return outputs['split_map_raw'].sum() * 0.0
        loss = torch.stack(cls_losses).mean()
        if point_losses:
            loss = loss + self.ifi_point_coef * torch.stack(point_losses).mean()
        return loss

    def _nearest_query_index(self, output, gt_point, device, dtype):
        img_h, img_w = output['img_shape']
        point_queries = output.get('points_queries')
        if point_queries is None or point_queries.numel() == 0:
            return None
        query_abs = point_queries.to(device=device, dtype=dtype).clone()
        query_abs[:, 0] *= img_h
        query_abs[:, 1] *= img_w
        return torch.cdist(gt_point.reshape(1, 2), query_abs, p=2).argmin(dim=1)[0]

    def _qd_positive_loss(self, output, batch_idx, query_idx, gt_point):
        logits = output['pred_logits']
        pred_points = output['pred_points']
        device = logits.device
        img_h, img_w = output['img_shape']
        cls_target = torch.ones(1, dtype=torch.long, device=device)
        loss_cls = F.cross_entropy(logits[batch_idx, query_idx].unsqueeze(0), cls_target, reduction='mean')

        gt_norm = gt_point.to(device=device, dtype=pred_points.dtype).clone()
        gt_norm[0] /= img_h
        gt_norm[1] /= img_w
        loss_point = F.smooth_l1_loss(
            pred_points[batch_idx, query_idx].reshape(1, 2),
            gt_norm.reshape(1, 2),
            reduction='none',
        ).sum(dim=-1).mean()
        return loss_cls + self.qd_apg_point_coef * loss_point

    def _qd_suppress_loss(self, output, batch_idx, query_idx):
        logits = output['pred_logits']
        device = logits.device
        cls_target = torch.zeros(1, dtype=torch.long, device=device)
        return F.cross_entropy(logits[batch_idx, query_idx].unsqueeze(0), cls_target, reduction='mean')

    def compute_qd_apg_loss(self, outputs, targets):
        """Quadtree-Dual APG.

        PET has two routed proposal sets: sparse 8x and dense 4x queries. Plain
        APG can accidentally encourage both branches around the same GT point.
        QD-APG chooses exactly one branch as the auxiliary positive branch and
        locally suppresses the other. The default route comes from the GT local
        count used by PET's splitter target; using the live split map as a
        teacher is available for ablations but is unsafe early in training.
        """
        output_sparse = outputs['sparse']
        output_dense = outputs['dense']
        split_map = outputs['split_map_raw']
        device = split_map.device
        dtype = output_sparse['pred_points'].dtype
        img_h, img_w = output_sparse['img_shape']
        split_h, split_w = split_map.shape[-2:]
        route_threshold = float(self.split_threshold) if self.split_threshold >= 0 else 0.5

        positive_losses = []
        suppress_losses = []
        for batch_idx, target in enumerate(targets):
            gt_points = target['points'].to(device=device, dtype=dtype)
            if gt_points.numel() == 0:
                continue
            y = torch.clamp((gt_points[:, 0] / max(float(img_h), 1.0) * split_h).long(), 0, split_h - 1)
            x = torch.clamp((gt_points[:, 1] / max(float(img_w), 1.0) * split_w).long(), 0, split_w - 1)
            if self.qd_apg_route_source == 'split_map':
                dense_routes = split_map[batch_idx, 0, y, x] > route_threshold
            else:
                linear_idx = y * split_w + x
                counts = torch.zeros(split_h * split_w, dtype=torch.long, device=device)
                counts.scatter_add_(0, linear_idx, torch.ones_like(linear_idx, dtype=torch.long))
                dense_routes = counts[linear_idx] >= self.split_count_threshold
            for point_idx, gt_point in enumerate(gt_points):
                sparse_idx = self._nearest_query_index(output_sparse, gt_point, device, dtype)
                dense_idx = self._nearest_query_index(output_dense, gt_point, device, dtype)
                if sparse_idx is None or dense_idx is None:
                    continue
                if bool(dense_routes[point_idx].item()):
                    positive_losses.append(self._qd_positive_loss(output_dense, batch_idx, dense_idx, gt_point))
                    if self.qd_apg_suppress_coef > 0:
                        suppress_losses.append(self._qd_suppress_loss(output_sparse, batch_idx, sparse_idx))
                else:
                    positive_losses.append(self._qd_positive_loss(output_sparse, batch_idx, sparse_idx, gt_point))
                    if self.qd_apg_suppress_coef > 0:
                        suppress_losses.append(self._qd_suppress_loss(output_dense, batch_idx, dense_idx))

        if not positive_losses:
            return output_sparse['pred_logits'].sum() * 0.0
        loss = torch.stack(positive_losses).mean()
        if suppress_losses:
            loss = loss + self.qd_apg_suppress_coef * torch.stack(suppress_losses).mean()
        return loss

    def compute_count_loss(self, outputs, targets):
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        device = output_sparse['pred_logits'].device
        dtype = output_sparse['pred_logits'].dtype
        target_counts = torch.as_tensor(
            [target['points'].shape[0] for target in targets],
            dtype=dtype,
            device=device,
        )

        sparse_scores = F.softmax(output_sparse['pred_logits'], -1)[..., 1]
        dense_scores = F.softmax(output_dense['pred_logits'], -1)[..., 1]

        if self.count_loss_gate == 'hard':
            threshold = outputs['split_threshold'].to(device=device, dtype=dtype)
            sparse_gate = ((1.0 - outputs['split_map_sparse'].to(device=device, dtype=dtype)) <= threshold).to(dtype)
            dense_gate = (outputs['split_map_dense'].to(device=device, dtype=dtype) > threshold).to(dtype)
        else:
            sparse_gate = outputs['split_map_sparse'].to(device=device, dtype=dtype)
            dense_gate = outputs['split_map_dense'].to(device=device, dtype=dtype)
            if self.count_loss_gate == 'detach':
                sparse_gate = sparse_gate.detach()
                dense_gate = dense_gate.detach()

        pred_counts = (sparse_scores * sparse_gate.reshape_as(sparse_scores)).sum(dim=1)
        pred_counts = pred_counts + (dense_scores * dense_gate.reshape_as(dense_scores)).sum(dim=1)
        if self.count_loss_type == 'l1':
            return F.l1_loss(pred_counts, target_counts)
        if self.count_loss_type == 'smooth_l1':
            return F.smooth_l1_loss(pred_counts, target_counts)
        return F.l1_loss(torch.log1p(pred_counts), torch.log1p(target_counts))

    def _split_gates_for_scores(self, outputs, branch_name, scores, mode):
        if mode == 'none':
            return torch.ones_like(scores)
        if branch_name == 'sparse':
            if mode == 'hard':
                threshold = outputs['split_threshold'].to(device=scores.device, dtype=scores.dtype)
                gates = ((1.0 - outputs['split_map_sparse'].to(device=scores.device, dtype=scores.dtype)) <= threshold).to(scores.dtype)
            else:
                gates = outputs['split_map_sparse'].to(device=scores.device, dtype=scores.dtype)
        elif branch_name == 'dense':
            if mode == 'hard':
                threshold = outputs['split_threshold'].to(device=scores.device, dtype=scores.dtype)
                gates = (outputs['split_map_dense'].to(device=scores.device, dtype=scores.dtype) > threshold).to(scores.dtype)
            else:
                gates = outputs['split_map_dense'].to(device=scores.device, dtype=scores.dtype)
        else:
            raise ValueError(f'Unsupported branch: {branch_name}')
        if mode == 'detach':
            gates = gates.detach()
        return gates.reshape_as(scores).clamp(0, 1)

    def _region_count_gates(self, outputs, branch_name, scores):
        return self._split_gates_for_scores(outputs, branch_name, scores, self.region_count_gate)

    def compute_region_count_loss(self, outputs, targets):
        """Local Point-to-Region count calibration on PET's real query logits."""
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        device = output_sparse['pred_logits'].device
        dtype = output_sparse['pred_logits'].dtype
        grid = self.region_count_grid
        num_regions = grid * grid
        target_counts = torch.zeros(len(targets), num_regions, dtype=dtype, device=device)
        img_h, img_w = output_sparse['img_shape']
        for batch_idx, target in enumerate(targets):
            points = target['points'].to(device=device, dtype=dtype)
            if points.numel() == 0:
                continue
            y = torch.clamp((points[:, 0] / max(float(img_h), 1.0) * grid).long(), 0, grid - 1)
            x = torch.clamp((points[:, 1] / max(float(img_w), 1.0) * grid).long(), 0, grid - 1)
            linear_idx = y * grid + x
            target_counts[batch_idx].scatter_add_(0, linear_idx, torch.ones_like(linear_idx, dtype=dtype))

        pred_counts = torch.zeros_like(target_counts)
        for branch_name, output in (('sparse', output_sparse), ('dense', output_dense)):
            scores = F.softmax(output['pred_logits'], -1)[..., 1]
            gates = self._region_count_gates(outputs, branch_name, scores)
            query_points = output['points_queries'].to(device=device, dtype=dtype).clamp(0, 1)
            y = torch.clamp((query_points[:, 0] * grid).long(), 0, grid - 1)
            x = torch.clamp((query_points[:, 1] * grid).long(), 0, grid - 1)
            linear_idx = y * grid + x
            for batch_idx in range(scores.shape[0]):
                pred_counts[batch_idx].scatter_add_(0, linear_idx, scores[batch_idx] * gates[batch_idx])

        if self.region_count_type == 'l1':
            return F.l1_loss(pred_counts, target_counts)
        if self.region_count_type == 'smooth_l1':
            return F.smooth_l1_loss(pred_counts, target_counts)
        return F.l1_loss(torch.log1p(pred_counts), torch.log1p(target_counts))

    def compute_bayesian_point_loss(self, outputs, targets):
        """Point-level expected-count loss around each GT point.

        This adapts Bayesian crowd-count supervision to PET's point outputs:
        the expected gated person probability near each GT point should be one,
        while probability far from any GT point is softly suppressed.
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        device = output_sparse['pred_logits'].device
        sigma = max(float(self.bayesian_sigma), 1e-6)
        pos_losses = []
        bg_losses = []

        for branch_name, output in (('sparse', output_sparse), ('dense', output_dense)):
            scores = F.softmax(output['pred_logits'], -1)[..., 1]
            gates = self._split_gates_for_scores(outputs, branch_name, scores, self.bayesian_loss_gate)
            weighted_scores = (scores * gates).float()
            img_h, img_w = output['img_shape']
            pred_points = output['pred_points'].clamp(0.0, 1.0).float()
            pred_abs = pred_points.clone()
            pred_abs[..., 0] *= float(img_h)
            pred_abs[..., 1] *= float(img_w)

            for batch_idx, target in enumerate(targets):
                gt_points = target['points'].to(device=device, dtype=torch.float32)
                branch_scores = weighted_scores[batch_idx]
                if gt_points.numel() == 0:
                    bg_losses.append(branch_scores.mean())
                    continue

                distances = torch.cdist(gt_points, pred_abs[batch_idx], p=2)
                weights = torch.exp(-0.5 * (distances / sigma).pow(2))
                expected = (weights * branch_scores.unsqueeze(0)).sum(dim=1)
                pos_losses.append(F.smooth_l1_loss(expected, torch.ones_like(expected)))

                if self.bayesian_bg_coef > 0:
                    background_weight = (1.0 - weights.max(dim=0).values).clamp(0.0, 1.0).detach()
                    bg_losses.append((branch_scores * background_weight).mean())

        if not pos_losses and not bg_losses:
            return output_sparse['pred_logits'].sum() * 0.0
        loss = output_sparse['pred_logits'].sum() * 0.0
        if pos_losses:
            loss = loss + torch.stack(pos_losses).mean()
        if bg_losses and self.bayesian_bg_coef > 0:
            loss = loss + self.bayesian_bg_coef * torch.stack(bg_losses).mean()
        return loss

    def build_split_targets(self, targets, split_map, img_shape):
        bs, _, split_h, split_w = split_map.shape
        img_h, img_w = img_shape
        split_target = torch.zeros_like(split_map)
        for batch_idx, target in enumerate(targets[:bs]):
            points = target['points'].to(split_map.device)
            if points.numel() == 0:
                continue
            y = torch.clamp((points[:, 0] / max(float(img_h), 1.0) * split_h).long(), 0, split_h - 1)
            x = torch.clamp((points[:, 1] / max(float(img_w), 1.0) * split_w).long(), 0, split_w - 1)
            linear_idx = y * split_w + x
            counts = torch.zeros(split_h * split_w, dtype=split_map.dtype, device=split_map.device)
            counts.scatter_add_(0, linear_idx, torch.ones_like(linear_idx, dtype=split_map.dtype))
            split_target[batch_idx, 0] = (counts.view(split_h, split_w) >= self.split_count_threshold).to(split_map.dtype)
        return split_target

    def balanced_binary_loss(self, pred, target, pos_weight=1.0, neg_weight=1.0, eps=1e-6):
        # This receives sigmoid probabilities, not logits. Avoid
        # F.binary_cross_entropy here because PyTorch rejects it under AMP.
        pred = pred.float().clamp(eps, 1.0 - eps)
        target = target.to(device=pred.device, dtype=pred.dtype)
        raw_loss = -(target * pred.log() + (1.0 - target) * (1.0 - pred).log())
        pos_mask = target >= 0.5
        neg_mask = ~pos_mask
        loss = raw_loss.sum() * 0.0
        if pos_mask.any():
            loss = loss + pos_weight * raw_loss[pos_mask].mean()
        if neg_mask.any():
            loss = loss + neg_weight * raw_loss[neg_mask].mean()
        return loss

    def forward(self, samples: NestedTensor, **kwargs):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # backbone
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        # positional embedding
        dense_input_embed = self.pos_embed(samples)
        kwargs['dense_input_embed'] = dense_input_embed

        # feature projection
        features['4x'] = NestedTensor(self.input_proj[0](features['4x'].tensors), features['4x'].mask)
        features['8x'] = NestedTensor(self.input_proj[1](features['8x'].tensors), features['8x'].mask)

        # forward
        if 'train' in kwargs:
            out = self.train_forward(samples, features, pos, **kwargs)
        else:
            out = self.test_forward(samples, features, pos, **kwargs)   
        return out

    def get_split_threshold(self, split_map):
        if self.split_threshold >= 0:
            return torch.as_tensor(self.split_threshold, dtype=split_map.dtype, device=split_map.device)
        # Match official PET behavior. Quantile-based thresholds force a large
        # fraction of windows into dense mode and are a major source of chronic
        # overcounting on this branch.
        return torch.as_tensor(0.5, dtype=split_map.dtype, device=split_map.device)

    def get_split_mask(self, split_map):
        threshold = self.get_split_threshold(split_map)
        return split_map >= threshold

    def get_score_mask(self, scores):
        flat = scores.detach().reshape(-1).float()
        if flat.numel() == 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        if self.score_threshold >= 0:
            threshold = torch.as_tensor(self.score_threshold, dtype=scores.dtype, device=scores.device)
        else:
            threshold = torch.quantile(flat, 0.95).to(dtype=scores.dtype, device=scores.device)
            threshold = threshold.clamp(0.05, 0.95)
        return scores > threshold

    def apply_eval_point_nms(self, pred_logits, pred_points, pred_offsets, points_queries, scores, img_shape):
        radius = float(self.eval_nms_radius)
        if radius <= 0 or pred_points.shape[0] <= 1:
            return pred_logits, pred_points, pred_offsets, points_queries

        # pred_points are normalized [y, x]; convert to pixels for a radius
        # that is stable across ShanghaiTech image sizes.
        img_h, img_w = img_shape
        scale = pred_points.new_tensor([float(img_h), float(img_w)])
        points_abs = pred_points * scale
        order = torch.argsort(scores, descending=True)
        suppressed = torch.zeros(pred_points.shape[0], dtype=torch.bool, device=pred_points.device)
        keep = []
        for idx in order:
            if bool(suppressed[idx].item()):
                continue
            keep.append(idx)
            dist = torch.linalg.vector_norm(points_abs - points_abs[idx], dim=1)
            suppressed |= dist <= radius
            suppressed[idx] = False
        if not keep:
            empty = torch.empty(0, dtype=torch.long, device=pred_points.device)
            return pred_logits[empty], pred_points[empty], pred_offsets[empty], points_queries[empty]
        keep_idx = torch.stack(keep)
        return pred_logits[keep_idx], pred_points[keep_idx], pred_offsets[keep_idx], points_queries[keep_idx]

    def get_eval_branch_gate_mask(self, output, split_map, branch):
        mode = self.eval_branch_gate
        if mode == 'none':
            return None
        if output is None:
            return None
        if mode == 'query':
            points = output['points_queries']
        else:
            points = output['pred_points']
        if points.numel() == 0:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=split_map.device)

        points = points.to(device=split_map.device, dtype=split_map.dtype).clamp(0.0, 1.0)
        grid = torch.stack(
            [points[:, 1] * 2.0 - 1.0, points[:, 0] * 2.0 - 1.0],
            dim=-1,
        ).view(1, -1, 1, 2)
        split_values = F.grid_sample(
            split_map[:1],
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        ).view(-1)
        threshold = self.get_split_threshold(split_map).to(dtype=split_values.dtype, device=split_values.device)
        if branch == 'sparse':
            return split_values <= threshold
        if branch == 'dense':
            return split_values > threshold
        raise ValueError(f'Unsupported branch: {branch}')

    def apply_eval_soft_split_gate(self, output, split_map, branch, scores):
        mode = self.eval_soft_split_gate
        if mode == 'none' or output is None or scores.numel() == 0:
            return scores
        if mode == 'query':
            points = output['points_queries']
        else:
            points = output['pred_points']
        points = points.to(device=split_map.device, dtype=split_map.dtype).clamp(0.0, 1.0)
        grid = torch.stack(
            [points[:, 1] * 2.0 - 1.0, points[:, 0] * 2.0 - 1.0],
            dim=-1,
        ).view(1, -1, 1, 2)
        split_values = F.grid_sample(
            split_map[:1],
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        ).view_as(scores).to(device=scores.device, dtype=scores.dtype)
        if branch == 'sparse':
            responsibility = 1.0 - split_values
        elif branch == 'dense':
            responsibility = split_values
        else:
            raise ValueError(f'Unsupported branch: {branch}')
        return scores * responsibility.clamp(0.0, 1.0)

    def pet_forward(self, samples, features, pos, **kwargs):
        # context encoding
        src, mask = features[self.encode_feats].decompose()
        src_pos_embed = pos[self.encode_feats]
        assert mask is not None
        encode_src = self.context_encoder(src, src_pos_embed, mask)
        encode_src = self.quad_context_mixer(encode_src)
        context_info = (encode_src, src_pos_embed, mask)
        
        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape
        sp_h, sp_w = src_h, src_w
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)
        split_map = self.quadtree_splitter(encode_src)
        split_map_raw_sparse = F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)
        split_map_raw_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)
        split_map_dense = split_map_raw_dense
        split_map_sparse = 1 - split_map_raw_sparse
        split_threshold = self.get_split_threshold(split_map)
        split_mask_sparse = split_map_raw_sparse <= split_threshold
        split_mask_dense = split_map_raw_dense > split_threshold
        sparse_active = 'train' in kwargs or bool(split_mask_sparse.any().item())
        dense_active = 'train' in kwargs or bool(split_mask_dense.any().item())
        if 'train' not in kwargs and not sparse_active and not dense_active:
            sparse_active = True
        
        # quadtree layer0 forward (sparse)
        if sparse_active:
            sparse_kwargs = dict(kwargs)
            sparse_kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            sparse_kwargs['dec_win_size'] = list(self.sparse_dec_win_size)
            outputs_sparse = self.quadtree_sparse(samples, features, context_info, **sparse_kwargs)
        else:
            outputs_sparse = None
        
        # quadtree layer1 forward (dense)
        if dense_active:
            dense_kwargs = dict(kwargs)
            dense_kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            dense_kwargs['dec_win_size'] = list(self.dense_dec_win_size)
            outputs_dense = self.quadtree_dense(samples, features, context_info, **dense_kwargs)
        else:
            outputs_dense = None
        
        # format outputs
        outputs = dict()
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        outputs['split_mask_sparse'] = split_mask_sparse
        outputs['split_mask_dense'] = split_mask_dense
        outputs['split_threshold'] = split_threshold.detach()
        if 'train' in kwargs:
            outputs['encode_src'] = encode_src
        return outputs
    
    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)

        # compute loss
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples)
        return losses
    
    def test_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        out_dense, out_sparse = outputs['dense'], outputs['sparse']
        pred_logits_parts = []
        pred_points_parts = []
        pred_offsets_parts = []
        points_queries_parts = []
        score_parts = []
        template_out = out_sparse if out_sparse is not None else out_dense

        if out_sparse is not None:
            out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
            out_sparse_eval_scores = self.apply_eval_soft_split_gate(out_sparse, outputs['split_map_raw'], 'sparse', out_sparse_scores)
            index_sparse = self.get_score_mask(out_sparse_eval_scores).to(out_sparse['pred_logits'].device)
            sparse_gate = self.get_eval_branch_gate_mask(out_sparse, outputs['split_map_raw'], 'sparse')
            if sparse_gate is not None:
                index_sparse = index_sparse & sparse_gate.to(device=index_sparse.device)
            pred_logits_parts.append(out_sparse['pred_logits'][index_sparse])
            pred_points_parts.append(out_sparse['pred_points'][index_sparse])
            pred_offsets_parts.append(out_sparse['pred_offsets'][index_sparse])
            points_queries_parts.append(out_sparse['points_queries'][index_sparse])
            score_parts.append(out_sparse_eval_scores[index_sparse])

        if out_dense is not None:
            out_dense_scores = torch.nn.functional.softmax(out_dense['pred_logits'], -1)[..., 1]
            out_dense_eval_scores = self.apply_eval_soft_split_gate(out_dense, outputs['split_map_raw'], 'dense', out_dense_scores)
            index_dense = self.get_score_mask(out_dense_eval_scores).to(out_dense['pred_logits'].device)
            dense_gate = self.get_eval_branch_gate_mask(out_dense, outputs['split_map_raw'], 'dense')
            if dense_gate is not None:
                index_dense = index_dense & dense_gate.to(device=index_dense.device)
            pred_logits_parts.append(out_dense['pred_logits'][index_dense])
            pred_points_parts.append(out_dense['pred_points'][index_dense])
            pred_offsets_parts.append(out_dense['pred_offsets'][index_dense])
            points_queries_parts.append(out_dense['points_queries'][index_dense])
            score_parts.append(out_dense_eval_scores[index_dense])

        if pred_logits_parts:
            pred_logits = torch.cat(pred_logits_parts, dim=0)
            pred_points = torch.cat(pred_points_parts, dim=0)
            pred_offsets = torch.cat(pred_offsets_parts, dim=0)
            points_queries_out = torch.cat(points_queries_parts, dim=0)
            scores = torch.cat(score_parts, dim=0)
        else:
            device = outputs['split_map_raw'].device
            pred_logits = torch.empty((0, 2), dtype=template_out['pred_logits'].dtype, device=device)
            pred_points = torch.empty((0, 2), dtype=template_out['pred_points'].dtype, device=device)
            pred_offsets = torch.empty((0, 2), dtype=template_out['pred_offsets'].dtype, device=device)
            points_queries_out = torch.empty((0, 2), dtype=template_out['points_queries'].dtype, device=device)
            scores = torch.empty((0,), dtype=template_out['pred_logits'].dtype, device=device)

        pred_logits, pred_points, pred_offsets, points_queries_out = self.apply_eval_point_nms(
            pred_logits,
            pred_points,
            pred_offsets,
            points_queries_out,
            scores,
            template_out['img_shape'],
        )

        div_out = dict()
        for name in list(template_out.keys()):
            if name == 'points_queries':
                continue
            if name == 'pred_logits':
                div_out[name] = pred_logits.unsqueeze(0)
            elif name == 'pred_points':
                div_out[name] = pred_points.unsqueeze(0)
            elif name == 'pred_offsets':
                div_out[name] = pred_offsets.unsqueeze(0)
            elif 'pred' in name:
                div_out[name] = template_out[name]
            else:
                div_out[name] = template_out[name]
        div_out['points_queries'] = points_queries_out.unsqueeze(0)
        div_out['split_map_raw'] = outputs['split_map_raw']
        div_out['split_threshold'] = outputs['split_threshold']
        return div_out


class SetCriterion(nn.Module):
    """ Compute the loss for PET:
        1) compute hungarian assignment between ground truth points and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction and split map
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 negative_loss_coef=0.1, non_div_loss_coef=0.25,
                 pet_loss_variant='paper', args=None):
        """
        Parameters:
            num_classes: one-class in crowd counting
            matcher: module able to compute a matching between targets and point queries
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.class_loss_type = getattr(args, 'class_loss_type', 'ce')
        if self.class_loss_type not in ('ce', 'focal'):
            raise ValueError('class_loss_type must be one of "ce" or "focal"')
        self.focal_alpha = float(getattr(args, 'focal_alpha', 0.25))
        self.focal_gamma = float(getattr(args, 'focal_gamma', 2.0))
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef    # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)
        self.negative_loss_coef = negative_loss_coef
        self.non_div_loss_coef = non_div_loss_coef
        self.pet_loss_variant = pet_loss_variant
        self.div_thrs_dict = {8: 0.0, 4: 0.5}

    def classification_loss_per_query(self, src_logits, target_classes):
        if self.class_loss_type == 'ce':
            return F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                ignore_index=-1,
                reduction='none',
            )

        valid = target_classes != -1
        safe_targets = target_classes.clamp_min(0)
        log_probs = F.log_softmax(src_logits, dim=-1)
        log_pt = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        pt = log_pt.exp()
        alpha = src_logits.new_tensor(self.focal_alpha).clamp(0.0, 1.0)
        alpha_t = torch.where(safe_targets == 1, alpha, 1.0 - alpha)
        loss = -alpha_t * (1.0 - pt).pow(self.focal_gamma) * log_pt
        return loss * valid.to(loss.dtype)

    def weighted_mean_loss(self, raw_loss, weights=None, eps=1e-6):
        if weights is None:
            return raw_loss.mean()
        weights = weights.to(raw_loss.device, dtype=raw_loss.dtype).reshape_as(raw_loss)
        valid = weights > 0
        if not valid.any():
            return raw_loss.sum() * 0.0
        return (raw_loss[valid] * weights[valid]).sum() / (weights[valid].sum() + eps)
    
    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        idx = (idx[0].to(src_logits.device), idx[1].to(src_logits.device))
        if idx[0].numel() > 0:
            target_classes_o = torch.cat([
                t["labels"][J.to(t["labels"].device)] for t, (_, J) in zip(targets, indices)
            ]).to(src_logits.device)
        else:
            target_classes_o = torch.empty(0, dtype=torch.int64, device=src_logits.device)
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # compute classification loss
        if self.pet_loss_variant == 'paper':
            if 'div' in kwargs:
                den = torch.stack([target['density'].reshape(()) for target in targets]).to(src_logits.device)
                den_sort = torch.sort(den)[1]
                ds_idx = den_sort[:len(den_sort)//2]
                sp_idx = den_sort[len(den_sort)//2:]
                eps = 1e-5

                weights = target_classes.clone().float()
                weights[weights == 0] = self.empty_weight[0]
                weights[weights == 1] = self.empty_weight[1]
                raw_ce_loss = self.classification_loss_per_query(src_logits, target_classes)

                split_map = kwargs['div'].to(src_logits.device)
                div_thrs = self.div_thrs_dict[outputs['pq_stride']]
                div_mask = split_map > div_thrs

                loss_ce_sp = (raw_ce_loss * weights * div_mask)[sp_idx].sum() / ((weights * div_mask)[sp_idx].sum() + eps)
                loss_ce_ds = (raw_ce_loss * weights * div_mask)[ds_idx].sum() / ((weights * div_mask)[ds_idx].sum() + eps)
                non_div_mask = split_map <= div_thrs
                loss_ce_nondiv = (raw_ce_loss * weights * non_div_mask).sum() / ((weights * non_div_mask).sum() + eps)
                loss_ce = loss_ce_sp + loss_ce_ds + loss_ce_nondiv
            else:
                raw_ce_loss = self.classification_loss_per_query(src_logits, target_classes)
                if self.class_loss_type == 'ce':
                    weights = target_classes.clone().float()
                    weights[weights == 0] = self.empty_weight[0]
                    weights[weights == 1] = self.empty_weight[1]
                    loss_ce = self.weighted_mean_loss(raw_ce_loss, weights)
                else:
                    loss_ce = self.weighted_mean_loss(raw_ce_loss)
            return {'loss_ce': loss_ce}

        raw_ce_loss = self.classification_loss_per_query(src_logits, target_classes)
        if self.class_loss_type == 'ce':
            class_weight = target_classes.clone().float()
            class_weight[class_weight == 0] = self.empty_weight[0]
            class_weight[class_weight == 1] = self.empty_weight[1]
            raw_ce_loss = raw_ce_loss * class_weight
        if 'div' in kwargs:
            split_weight = kwargs['div'].to(src_logits.device, dtype=raw_ce_loss.dtype).reshape_as(raw_ce_loss).clamp(0, 1)
            region_weight = split_weight + self.non_div_loss_coef * (1.0 - split_weight)
            loss_ce = self.weighted_mean_loss(raw_ce_loss, region_weight)
        else:
            loss_ce = self.weighted_mean_loss(raw_ce_loss)

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        assert 'pred_points' in outputs
        # get indices
        idx = self._get_src_permutation_idx(indices)
        idx = (idx[0].to(outputs['pred_points'].device), idx[1].to(outputs['pred_points'].device))
        src_points = outputs['pred_points'][idx]
        if idx[0].numel() > 0:
            target_points = torch.cat([
                t['points'][i.to(t['points'].device)] for t, (_, i) in zip(targets, indices)
            ], dim=0).to(outputs['pred_points'].device)
        else:
            target_points = torch.empty((0, 2), dtype=outputs['pred_points'].dtype, device=outputs['pred_points'].device)

        # compute regression loss
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        if target_points.numel() > 0:
            target_points[:, 0] /= img_h
            target_points[:, 1] /= img_w
        loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')

        if self.pet_loss_variant == 'paper':
            if loss_points_raw.numel() == 0:
                losses['loss_points'] = outputs['pred_points'].sum() * 0.0
                return losses

            if 'div' in kwargs:
                den = torch.stack([target['density'].reshape(()) for target in targets]).to(outputs['pred_points'].device)
                den_sort = torch.sort(den)[1]
                img_ds_idx = den_sort[:len(den_sort)//2]
                img_sp_idx = den_sort[len(den_sort)//2:]
                ds_parts = [torch.where(idx[0] == bs_id)[0] for bs_id in img_ds_idx]
                sp_parts = [torch.where(idx[0] == bs_id)[0] for bs_id in img_sp_idx]
                pt_ds_idx = torch.cat(ds_parts) if ds_parts else torch.empty(0, dtype=torch.int64, device=idx[0].device)
                pt_sp_idx = torch.cat(sp_parts) if sp_parts else torch.empty(0, dtype=torch.int64, device=idx[0].device)

                eps = 1e-5
                split_map = kwargs['div'].to(outputs['pred_points'].device)
                div_thrs = self.div_thrs_dict[outputs['pq_stride']]
                div_mask = split_map > div_thrs
                loss_points_div = loss_points_raw * div_mask[idx].unsqueeze(-1)
                loss_points_div_sp = loss_points_div[pt_sp_idx].sum() / (len(pt_sp_idx) + eps)
                loss_points_div_ds = loss_points_div[pt_ds_idx].sum() / (len(pt_ds_idx) + eps)

                non_div_mask = split_map <= div_thrs
                loss_points_nondiv = (loss_points_raw * non_div_mask[idx].unsqueeze(-1)).sum() / (non_div_mask[idx].sum() + eps)
                losses['loss_points'] = loss_points_div_sp + loss_points_div_ds + loss_points_nondiv
            else:
                losses['loss_points'] = loss_points_raw.sum() / num_points
            return losses

        loss_points_raw = loss_points_raw.sum(dim=-1)

        if 'div' in kwargs:
            eps = 1e-5
            split_map = kwargs['div'].to(outputs['pred_points'].device, dtype=loss_points_raw.dtype)
            split_weight = split_map[idx].clamp(0, 1)
            point_weight = split_weight + self.non_div_loss_coef * (1.0 - split_weight)
            losses['loss_points'] = (loss_points_raw * point_weight).sum() / (point_weight.sum() + eps)
        else:
            losses['loss_points'] = loss_points_raw.sum() / num_points
        
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        return losses


class MLP(nn.Module):
    """
    Multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_reduce=False, use_relu=True):
        super().__init__()
        self.num_layers = num_layers
        if is_reduce:
            h = [hidden_dim//2**i for i in range(num_layers - 1)]
        else:
            h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.use_relu = use_relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_relu:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            else:
                x = layer(x)
        return x


def build_pet(args):
    device = torch.device(args.device)

    # build model
    num_classes = 1
    backbone = build_backbone(args)
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )

    # build loss criterion
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses,
                             negative_loss_coef=getattr(args, 'negative_loss_coef', 0.1),
                             non_div_loss_coef=getattr(args, 'non_div_loss_coef', 0.25),
                             pet_loss_variant=getattr(args, 'pet_loss_variant', 'paper'),
                             args=args)
    criterion.to(device)
    return model, criterion
