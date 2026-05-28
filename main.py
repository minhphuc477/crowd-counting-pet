import argparse
import copy
import datetime
import json
import math
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models.backbones import get_supported_timm_backbones, is_timm_backbone


BASE_TRAINING_DEFAULTS = {
    'lr': 1e-4,
    'lr_backbone': 1e-5,
    'lr_backbone_adapter': -1.0,
    'batch_size': 8,
    'warmup_epochs': 5,
    'freeze_backbone_epochs': 0,
}

BACKBONE_RECIPES = {
    'heavy': {
        'batch_size': 4,
        'lr': 5e-5,
        'lr_backbone': 5e-6,
        'lr_backbone_adapter': 1e-4,
        'warmup_epochs': 10,
        'freeze_backbone_epochs': 5,
    },
    'mobile': {
        'batch_size': 8,
        'lr': 7.5e-5,
        'lr_backbone': 7.5e-6,
        'lr_backbone_adapter': 1e-4,
        'warmup_epochs': 8,
        'freeze_backbone_epochs': 3,
    },
    'vgg': {
        'batch_size': 8,
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'lr_backbone_adapter': 1e-4,
        'warmup_epochs': 5,
        'freeze_backbone_epochs': 0,
    },
}

HEAVY_BACKBONE_PREFIXES = (
    'convnext_base',
    'convnextv2_base',
    'convnextv2_large',
    'convnextv2_huge',
    'swinv2',
    'maxvit',
    'resnet',
    'pvt_v2',
    'pvtv2',
)

MOBILE_BACKBONE_PREFIXES = (
    'convnext_tiny',
    'convnextv2_tiny',
    'convnextv2_small',
    'fastvit',
    'efficientvit',
    'efficientnet_',
    'efficientnetv2',
    'tf_efficientnetv2',
    'tf_efficientnet_',
    'mobilenetv4',
    'hgnet',
    'hgnetv2',
    'edgenext',
    'repvit',
)

BACKBONE_ABLATION_PRESETS = {
    'crowd_dense': [
        'convnextv2_base',
        'convnext_base',
        'swinv2_base',
        'maxvit_small',
        'pvtv2_b1',
    ],
    'latency': [
        'convnextv2_tiny',
        'fastvit_tiny',
        'efficientvit_tiny',
        'mobilenetv4_small',
        'repvit_tiny',
        'edgenext_tiny',
    ],
    'full': list(get_supported_timm_backbones()),
}


def get_backbone_recipe(backbone_name):
    if backbone_name.startswith('vgg'):
        return BACKBONE_RECIPES['vgg']
    if any(backbone_name.startswith(prefix) for prefix in HEAVY_BACKBONE_PREFIXES):
        return BACKBONE_RECIPES['heavy']
    if any(backbone_name.startswith(prefix) for prefix in MOBILE_BACKBONE_PREFIXES):
        return BACKBONE_RECIPES['mobile']
    return None


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # training Parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_backbone_adapter', default=-1.0, type=float,
                        help='learning rate for randomly initialized backbone adapters/FPN; negative uses --lr')
    parser.add_argument('--freeze_backbone_epochs', default=0, type=int,
                        help='freeze pretrained backbone feature extractor for this many initial epochs')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--lr_scheduler', default='step', type=str,
                        choices=('step', 'warmup_hold_cosine'),
                        help='learning-rate schedule to use')
    parser.add_argument('--lr_drop', default=-1, type=int,
                        help='epoch interval for StepLR decay; negative keeps original PET behavior and drops after --epochs')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='multiplicative LR decay factor for StepLR')
    parser.add_argument('--auto_backbone_recipe', action='store_true',
                        help='opt into backbone-specific lr/batch/warmup defaults')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='number of warmup epochs')
    parser.add_argument('--hold_epochs', default=-1, type=int,
                        help='epochs to keep peak lr before cosine decay; -1 picks an automatic value')
    parser.add_argument('--min_lr', default=1e-7, type=float,
                        help='minimum learning rate reached by cosine annealing')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--list_backbones', action='store_true',
                        help='print supported timm ablation backbones and exit')
    parser.add_argument('--no_pretrained_backbone', action='store_true',
                        help='initialize the backbone randomly instead of loading timm/ImageNet weights')
    parser.add_argument('--allow_random_backbone_fallback', action='store_true',
                        help='allow timm backbones to continue with random init if pretrained weights cannot load')
    parser.add_argument('--timm_adapter', default='lite_fpn', choices=('lite_fpn', 'direct', 'fpn'),
                        help='adapter used to map timm features into PET 4x/8x features')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--transformer_activation', default='relu', choices=('relu', 'gelu'),
                        help='activation used in transformer feed-forward blocks')
    parser.add_argument('--transformer_norm_style', default='post', choices=('post', 'pre'),
                        help='post matches official PET; pre is more stable for deeper transformer stacks')
    parser.add_argument('--enc_win_sizes', default='', type=str,
                        help='encoder window sizes as "w,h;w,h;..."; empty keeps paper PET defaults')
    parser.add_argument('--sparse_dec_win_size', default='', type=str,
                        help='sparse decoder window size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--dense_dec_win_size', default='', type=str,
                        help='dense decoder window size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--context_patch_size', default='', type=str,
                        help='quadtree splitter context patch size as "w,h"; empty keeps paper PET default')
    parser.add_argument('--splitter_head', default='pool', choices=('pool', 'conv'),
                        help='quadtree splitter head; pool matches official PET, conv adds local context')
    parser.add_argument('--splitter_hidden_dim', default=128, type=int,
                        help='hidden channels for --splitter_head conv')
    parser.add_argument('--splitter_activation', default='gelu', choices=('relu', 'gelu'),
                        help='activation used by --splitter_head conv')

    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--count_loss_coef', default=0.0, type=float,
                        help='optional L1 loss on soft predicted count; 0 disables it')
    parser.add_argument('--count_loss_gate', default='detach', choices=('detach', 'soft', 'hard'),
                        help='routing gates used by count loss; detach calibrates scores without splitter gradients')
    parser.add_argument('--count_loss_type', default='log_l1', choices=('log_l1', 'l1', 'smooth_l1'),
                        help='count-loss scale; log_l1 is safer early in training')
    parser.add_argument('--count_loss_start_epoch', default=-1, type=int,
                        help='epoch to enable count loss; negative uses warmup_epochs')
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--pet_loss_variant', default='paper', choices=('paper', 'balanced'),
                        help='paper matches official PET; balanced enables experimental zero/negative-region losses')
    parser.add_argument('--negative_loss_coef', default=0.1, type=float,
                        help='extra scale for all-negative classification and split-map regions')
    parser.add_argument('--non_div_loss_coef', default=0.25, type=float,
                        help='auxiliary loss scale outside the current quadtree branch')
    parser.add_argument('--quadtree_loss_coef', default=0.1, type=float,
                        help='ground-truth split-map quality loss coefficient')
    parser.add_argument('--quadtree_prior_coef', default=0.025, type=float,
                        help='legacy image-level splitter prior coefficient')
    parser.add_argument('--split_count_threshold', default=2, type=int,
                        help='minimum people in a splitter cell before it should split')
    parser.add_argument('--split_pos_weight', default=1.0, type=float,
                        help='positive cell weight for quadtree quality loss')
    parser.add_argument('--split_threshold', default=0.5, type=float,
                        help='split-map threshold used to route sparse vs dense inference windows')
    parser.add_argument('--split_threshold_quantile', default=0.55, type=float,
                        help='legacy adaptive split-map quantile (unused when split_threshold is set)')
    parser.add_argument('--score_threshold', default=0.5, type=float,
                        help='point classification threshold; negative enables adaptive score thresholding')

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="", type=str)
    parser.add_argument('--patch_size', default=256, type=int,
                        help='training crop size for SHA')
    parser.add_argument('--crop_attempts', default=1, type=int,
                        help='number of random crop candidates tried per positive training image')
    parser.add_argument('--min_crop_points', default=0, type=int,
                        help='minimum people desired in a positive training crop')

    # misc parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_model_only', action='store_true',
                        help='load only model weights from --resume and reset optimizer/scheduler/epoch counters')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--syn_bn', default=0, type=int)
    parser.add_argument('--ema_decay', default=0.0, type=float,
                        help='exponential moving average decay for eval/checkpointing; 0 disables EMA')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true', default=True,
                        help='enable deterministic CuDNN and seeded DataLoader workers')
    parser.add_argument('--no_deterministic', dest='deterministic', action='store_false',
                        help='disable deterministic CuDNN mode')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def apply_backbone_recipe(args):
    """Apply backbone-specific fine-tuning defaults when the user left a generic setting in place."""
    recipe = get_backbone_recipe(args.backbone)
    if recipe is None:
        return

    for key, tuned_value in recipe.items():
        default_value = BASE_TRAINING_DEFAULTS.get(key)
        current_value = getattr(args, key, default_value)
        if default_value is not None and current_value == default_value:
            setattr(args, key, tuned_value)


def merge_checkpoint_args(args, checkpoint):
    checkpoint_args = checkpoint.get('args')
    if checkpoint_args is None:
        return args
    if isinstance(checkpoint_args, dict):
        checkpoint_args = argparse.Namespace(**checkpoint_args)

    merged = argparse.Namespace(**vars(checkpoint_args))
    for key, value in vars(args).items():
        if not hasattr(merged, key):
            setattr(merged, key, value)
    runtime_keys = {
        'resume', 'device', 'output_dir', 'seed', 'start_epoch',
        'resume_model_only', 'num_workers', 'world_size', 'dist_url', 'list_backbones', 'syn_bn', 'deterministic',
        # allow overriding schedule/eval settings at resume time
        'epochs', 'eval_freq', 'data_path',
    }
    if getattr(args, 'resume_model_only', False):
        runtime_keys.update({
            'lr', 'lr_backbone', 'lr_backbone_adapter', 'weight_decay',
            'lr_scheduler', 'lr_drop', 'lr_gamma', 'warmup_epochs', 'hold_epochs',
            'min_lr', 'ema_decay',
        })
    for key in runtime_keys:
        setattr(merged, key, getattr(args, key))
    return merged


def build_optimizer_param_groups(model_without_ddp, args):
    """Keep pretrained backbone weights on low LR while training new adapters at main LR."""
    use_timm = is_timm_backbone(getattr(args, 'backbone', ''))
    timm_feature_prefix = 'backbone.backbone.backbone.'
    adapter_prefixes = (
        'backbone.backbone.fpn.',  # timm Joiner -> TimmBackbone -> BackboneFPN
        'backbone.backbone.lite_fpn.',  # timm Joiner -> TimmBackbone -> LiteFPNAdapter
        'backbone.backbone.direct_adapter.',  # timm Joiner -> TimmBackbone -> DirectFeatureAdapter
        'backbone.0.fpn.',         # VGG Joiner -> Backbone_VGG -> FeatsFusion
    )
    adapter_lr = float(getattr(args, 'lr_backbone_adapter', -1.0))

    main_params, adapter_params, backbone_params = [], [], []
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if use_timm and name.startswith(timm_feature_prefix):
            backbone_params.append(param)
        elif any(name.startswith(prefix) for prefix in adapter_prefixes):
            adapter_params.append(param)
        elif 'backbone' in name:
            backbone_params.append(param)
        else:
            main_params.append(param)

    param_groups = []
    group_summary = []
    if main_params:
        param_groups.append({'params': main_params})
        group_summary.append(('main', len(main_params), sum(p.numel() for p in main_params), args.lr))
    if adapter_params:
        adapter_group = {'params': adapter_params}
        effective_adapter_lr = args.lr if adapter_lr < 0 else adapter_lr
        if adapter_lr >= 0:
            adapter_group['lr'] = adapter_lr
        param_groups.append(adapter_group)
        group_summary.append(('backbone_adapter', len(adapter_params), sum(p.numel() for p in adapter_params), effective_adapter_lr))
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.lr_backbone})
        group_summary.append(('backbone', len(backbone_params), sum(p.numel() for p in backbone_params), args.lr_backbone))

    return param_groups, group_summary


def set_raw_backbone_trainability(model_without_ddp, args, trainable):
    use_timm = is_timm_backbone(getattr(args, 'backbone', ''))
    raw_backbone_prefixes = (
        ('backbone.backbone.backbone.',) if use_timm else ('backbone.0.body',)
    )
    changed = 0
    total = 0
    for name, param in model_without_ddp.named_parameters():
        if not any(name.startswith(prefix) for prefix in raw_backbone_prefixes):
            continue
        total += 1
        if param.requires_grad != trainable:
            param.requires_grad = trainable
            changed += 1
    return total, changed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_reproducibility(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelEma:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def set(self, model):
        self.module.load_state_dict(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(self.decay).add_(
                    model_value.to(device=ema_value.device, dtype=ema_value.dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                ema_value.copy_(model_value.to(device=ema_value.device, dtype=ema_value.dtype))

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)


def main(args):
    utils.init_distributed_mode(args)
    if args.list_backbones:
        print('Supported timm ablation backbones:')
        for backbone in get_supported_timm_backbones():
            print(f'  {backbone}')
        return

    checkpoint = None
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        args = merge_checkpoint_args(args, checkpoint)
        args.no_pretrained_backbone = True

    if getattr(args, 'auto_backbone_recipe', False):
        apply_backbone_recipe(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    set_reproducibility(seed, deterministic=getattr(args, 'deterministic', True))
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)
    if args.syn_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_ema = ModelEma(model_without_ddp, args.ema_decay) if args.ema_decay > 0 else None

    # build optimizer
    param_dicts, param_group_summary = build_optimizer_param_groups(model_without_ddp, args)
    if utils.is_main_process():
        for group_name, n_tensors, n_params, lr in param_group_summary:
            print(f'optimizer group {group_name}: tensors={n_tensors}, params={n_params}, lr={lr}')
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.lr_scheduler == 'step':
        lr_drop = args.epochs if args.lr_drop <= 0 else args.lr_drop
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop, gamma=args.lr_gamma)
    else:
        warmup_epochs = min(max(0, int(args.warmup_epochs)), max(args.epochs - 1, 0))
        hold_epochs = int(args.hold_epochs)
        if hold_epochs < 0:
            hold_epochs = max(0, args.epochs // 15)
        hold_epochs = min(max(0, hold_epochs), max(args.epochs - warmup_epochs - 1, 0))
        decay_epochs = max(1, args.epochs - warmup_epochs - hold_epochs)
        min_lr_ratio = float(args.min_lr) / max(float(args.lr), 1e-12)

        def warmup_hold_cosine_factor(epoch_idx):
            if args.epochs <= 1:
                return 1.0
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                warmup_progress = float(epoch_idx + 1) / float(warmup_epochs)
                return max(1e-3, warmup_progress)
            if epoch_idx < warmup_epochs + hold_epochs:
                return 1.0
            decay_progress = min(max(epoch_idx - warmup_epochs - hold_epochs, 0), decay_epochs) / float(decay_epochs)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return max(min_lr_ratio, cosine_factor)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_hold_cosine_factor)

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, seed=args.seed)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train, generator=data_loader_generator)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                worker_init_fn=seed_worker, generator=data_loader_generator)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                worker_init_fn=seed_worker, generator=data_loader_generator)

    # output directory and log
    output_dir = Path("./outputs") / args.dataset_file / args.output_dir
    run_log_name = output_dir / 'run_log.txt'
    if utils.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        print(f'outputs will be saved to: {output_dir.resolve()}')
        with open(run_log_name, "a", encoding="utf-8") as log_file:
            log_file.write('Run Log %s\n' % time.strftime("%c"))
            log_file.write("{}\n".format(args))
            log_file.write("parameters: {}\n".format(n_parameters))

    best_mae, best_mse, best_epoch = 1e8, 1e8, 0
    if checkpoint is not None:
        model_key = 'model'
        if model_ema is not None and 'model_raw' in checkpoint and not args.resume_model_only:
            model_key = 'model_raw'
        model_without_ddp.load_state_dict(checkpoint[model_key])
        if model_ema is not None:
            if args.resume_model_only:
                model_ema.set(model_without_ddp)
            elif 'model_ema' in checkpoint:
                model_ema.load_state_dict(checkpoint['model_ema'])
            elif model_key == 'model_raw' and 'model' in checkpoint:
                model_ema.load_state_dict(checkpoint['model'])
            else:
                model_ema.set(model_without_ddp)
        if not args.resume_model_only and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint.get('best_mae', best_mae)
            best_mse = checkpoint.get('best_mse', best_mse)
            best_epoch = checkpoint.get('best_epoch', best_epoch)

    def checkpoint_payload(epoch, model_state=None, include_raw_model=False):
        payload = {
            'model': model_without_ddp.state_dict() if model_state is None else model_state,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
            'best_mae': best_mae,
            'best_mse': best_mse,
            'best_epoch': best_epoch,
        }
        if model_ema is not None:
            payload['model_ema'] = model_ema.state_dict()
            payload['ema_decay'] = args.ema_decay
        if include_raw_model:
            payload['model_raw'] = model_without_ddp.state_dict()
        return payload

    # training
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        freeze_epochs = int(getattr(args, 'freeze_backbone_epochs', 0))
        if freeze_epochs > 0:
            backbone_trainable = epoch >= freeze_epochs
            total_raw_backbone, changed_raw_backbone = set_raw_backbone_trainability(
                model_without_ddp,
                args,
                trainable=backbone_trainable,
            )
            if utils.is_main_process() and (epoch == args.start_epoch or changed_raw_backbone):
                state = 'trainable' if backbone_trainable else 'frozen'
                print(f'raw backbone is {state}: tensors={total_raw_backbone}, freeze_backbone_epochs={freeze_epochs}')
        
        t1 = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, model_ema=model_ema, model_without_ddp=model_without_ddp)
        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        
        if utils.is_main_process():
            with open(run_log_name, "a", encoding="utf-8") as log_file:
                log_file.write('[ep %d][lr %.7f][%.2fs]\n' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        lr_scheduler.step()

        # save checkpoint
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master(checkpoint_payload(epoch), checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # write log
        if utils.is_main_process():
            with open(run_log_name, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # evaluation
        if epoch % args.eval_freq == 0 and epoch > 0:
            t1 = time.time()
            eval_model = model_ema.module if model_ema is not None else model
            eval_model_name = 'ema' if model_ema is not None else 'raw'
            test_stats = evaluate(eval_model, data_loader_val, device, epoch, None)
            t2 = time.time()

            # output results
            mae, mse = test_stats['mae'], test_stats['mse']
            improved = mae < best_mae
            if improved:
                best_epoch = epoch
                best_mae = mae
                best_mse = mse
            print("\n==========================")
            print("\nepoch:", epoch, "mae:", mae, "mse:", mse, "\n\nbest mae:", best_mae, "best epoch:", best_epoch)
            print("==========================\n")
            if utils.is_main_process():
                eval_record = {
                    'epoch': epoch,
                    'test_mae': float(mae),
                    'test_mse': float(mse),
                    'pred_cnt': float(test_stats.get('pred_cnt', 0.0)),
                    'gt_cnt': float(test_stats.get('gt_cnt', 0.0)),
                    'best_epoch': int(best_epoch),
                    'best_test_mae': float(best_mae),
                    'best_test_mse': float(best_mse),
                    'improved': bool(improved),
                    'eval_time': float(t2 - t1),
                    'eval_model': eval_model_name,
                }
                with open(run_log_name, "a", encoding="utf-8") as log_file:
                    log_file.write("epoch:{}, mae:{}, mse:{}, time{}, \n\nbest mae:{}, best epoch: {}\n".format(
                                                epoch, mae, mse, t2 - t1, best_mae, best_epoch))
                    log_file.write(json.dumps(eval_record) + "\n")
                with open(output_dir / 'eval_history.jsonl', "a", encoding="utf-8") as eval_file:
                    eval_file.write(json.dumps(eval_record) + "\n")
                (output_dir / 'latest_eval_results.json').write_text(
                    json.dumps(eval_record, indent=2) + "\n",
                    encoding="utf-8",
                )

            if improved and utils.is_main_process():
                utils.save_on_master(checkpoint_payload(epoch), output_dir / 'checkpoint.pth')
                (output_dir / 'best_eval_results.json').write_text(
                    json.dumps(eval_record, indent=2) + "\n",
                    encoding="utf-8",
                )
                best_model_state = model_ema.state_dict() if model_ema is not None else model_without_ddp.state_dict()
                utils.save_on_master(
                    checkpoint_payload(
                        epoch,
                        model_state=best_model_state,
                        include_raw_model=model_ema is not None,
                    ),
                    output_dir / 'best_checkpoint.pth',
                )

    if utils.is_main_process():
        final_record = {
            'epoch': best_epoch,
            'test_mae': float(best_mae),
            'test_mse': float(best_mse),
            'best_epoch': best_epoch,
            'best_test_mae': float(best_mae),
            'best_test_mse': float(best_mse),
            'final': True,
            'eval_model': 'ema' if model_ema is not None else 'raw',
        }
        with open(run_log_name, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(final_record) + "\n")
        (output_dir / 'final_results.json').write_text(
            json.dumps(final_record, indent=2) + "\n",
            encoding="utf-8",
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
