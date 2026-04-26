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
from engine import NonFiniteTrainingError, evaluate, train_one_epoch
from models import build_model
from models.backbones import get_convnextv2_training_defaults, resolve_convnextv2_backbone_name

try:
    import optuna
except ImportError:
    optuna = None


RESUME_ARG_FIELDS = utils.CHECKPOINT_MODEL_ARG_FIELDS + (
    'lr',
    'lr_backbone',
    'batch_size',
    'weight_decay',
    'clip_max_norm',
    'accum_iter',
    'use_amp',
    'eval_freq',
)


def get_args_parser():
    arg_parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # Nhóm tham số huấn luyện: tốc độ học, số epoch, lịch giảm learning rate và clipping.
    arg_parser.add_argument('--lr', default=1e-4, type=float)
    arg_parser.add_argument('--lr_backbone', default=1e-5, type=float)
    arg_parser.add_argument('--batch_size', default=8, type=int)
    arg_parser.add_argument('--weight_decay', default=1e-4, type=float)
    arg_parser.add_argument('--epochs', default=1500, type=int)
    arg_parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Nhóm tham số mô hình: cấu hình kiến trúc chính và các siêu tham số cốt lõi.
    # - Tham số cho backbone dùng để trích xuất đặc trưng thị giác ở nhiều mức.
    arg_parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the backbone to use")
    arg_parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - Tham số cho transformer encoder/decoder để mô hình hóa quan hệ không gian-ngữ cảnh.
    arg_parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    arg_parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    arg_parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    arg_parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    arg_parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # Nhóm tham số hàm loss: điều chỉnh trọng số giữa các mục tiêu học.
    # - Tham số cho matcher (Hungarian) dùng ghép truy vấn dự đoán với ground-truth.
    arg_parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    arg_parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - Hệ số trọng số cho từng thành phần loss để cân bằng mục tiêu học.
    arg_parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    arg_parser.add_argument('--point_loss_coef', default=5.0, type=float)
    arg_parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")

    # Nhóm tham số dữ liệu: đường dẫn, tên tập dữ liệu và các tùy chọn tiền xử lý.
    arg_parser.add_argument('--dataset_file', default="SHA")
    arg_parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)

    # Nhóm tham số phụ trợ: logging, checkpoint, thiết bị chạy và các tùy chọn tiện ích.
    arg_parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    arg_parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    arg_parser.add_argument('--seed', default=42, type=int)
    arg_parser.add_argument('--resume', default='', help='resume from checkpoint')
    arg_parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    arg_parser.add_argument('--num_workers', default=2, type=int)
    arg_parser.add_argument('--eval_freq', default=5, type=int)
    arg_parser.add_argument('--target_mae', default=math.inf, type=float,
                        help='stop early once validation MAE reaches this target; default disables early stopping')
    arg_parser.add_argument('--patch_size', default=256, type=int,
                        help='training crop size for SHA')
    arg_parser.add_argument('--inference_threshold', default=0.5, type=float,
                        help='confidence threshold used to count predicted points at inference')
    arg_parser.add_argument('--threshold_sweep', action=argparse.BooleanOptionalAction, default=False,
                        help='sweep validation count thresholds and keep the best one instead of using a fixed threshold')
    arg_parser.add_argument('--threshold_min', default=0.30, type=float,
                        help='minimum threshold value to consider during validation sweep')
    arg_parser.add_argument('--threshold_max', default=0.95, type=float,
                        help='maximum threshold value to consider during validation sweep')
    arg_parser.add_argument('--threshold_step', default=0.025, type=float,
                        help='step size for validation threshold sweep')
    arg_parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True,
                        help='enable deterministic training and data loading')
    arg_parser.add_argument('--split_warmup_epochs', default=5, type=int,
                        help='epochs to wait before enabling quadtree splitter loss')
    arg_parser.add_argument('--count_loss_coef', default=0.0, type=float,
                        help='weight for the soft count consistency loss')
    arg_parser.add_argument('--accum_iter', default=1, type=int,
                        help='gradient accumulation steps')
    arg_parser.add_argument('--disable_amp', action='store_true',
                        help='disable automatic mixed precision even on CUDA')
    arg_parser.add_argument('--search_trials', default=0, type=int,
                        help='number of hyperparameter-search trials to run before final training; 0 disables search')
    arg_parser.add_argument('--search_epochs', default=8, type=int,
                        help='epochs per hyperparameter-search trial')
    arg_parser.add_argument('--search_eval_freq', default=1, type=int,
                        help='evaluation frequency during hyperparameter search')
    arg_parser.add_argument('--syn_bn', default=0, type=int)

    # Nhóm tham số huấn luyện phân tán: thiết lập tiến trình, đồng bộ và backend giao tiếp.
    arg_parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    arg_parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return arg_parser


def get_output_dir(args):
    return Path("./outputs") / args.dataset_file / args.output_dir


def append_text(file_path, text):
    existing_text = file_path.read_text(encoding='utf-8') if file_path.exists() else ''
    file_path.write_text(existing_text + text, encoding='utf-8')


def should_use_amp(args, device):
    return device.type == 'cuda' and not args.disable_amp and getattr(args, 'use_amp', True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_determinism(args):
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def get_cuda_profile(device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return {'name': '', 'memory_gb': 0.0}

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    return {
        'name': props.name.lower(),
        'memory_gb': props.total_memory / (1024 ** 3),
    }


def is_rtx_5070_ti_like(device):
    profile = get_cuda_profile(device)
    gpu_name = profile['name']
    memory_gb = profile['memory_gb']
    return ('5070' in gpu_name and 'ti' in gpu_name) or (15.0 <= memory_gb <= 18.5 and 'nvidia' in gpu_name)


def apply_rtx_5070_ti_profile(args, device):
    args.backbone = 'convnextv2_base'
    args.batch_size, args.lr, args.lr_backbone = get_convnextv2_training_defaults(args.backbone)
    args.batch_size = 2
    args.hidden_dim = max(args.hidden_dim, 384)
    args.dim_feedforward = max(args.dim_feedforward, 768)
    args.dec_layers = max(args.dec_layers, 3)
    args.accum_iter = max(args.accum_iter, 2)
    args.use_amp = should_use_amp(args, device)
    args.enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8), (8, 4)]
    args.dec_win_sizes = {
        'sparse': [32, 16],
        'dense': [16, 8],
    }
    print('[5070Ti_profile] backbone=%s batch_size=%s hidden_dim=%s dim_feedforward=%s dec_layers=%s accum_iter=%s target_mae=%s amp=%s' % (
        args.backbone, args.batch_size, args.hidden_dim, args.dim_feedforward, args.dec_layers, args.accum_iter, args.target_mae, args.use_amp
    ))


def resolve_auto_tuning_defaults(args, device):
    if args.backbone == 'auto':
        if is_rtx_5070_ti_like(device):
            apply_rtx_5070_ti_profile(args, device)
        else:
            args.backbone = resolve_convnextv2_backbone_name(args.backbone)
            args.batch_size, args.lr, args.lr_backbone = get_convnextv2_training_defaults(args.backbone)
            args.enc_win_list = getattr(args, 'enc_win_list', [(32, 16), (32, 16), (16, 8), (16, 8)])
            args.dec_win_sizes = getattr(args, 'dec_win_sizes', {
                'sparse': [16, 8],
                'dense': [8, 4],
            })
        args.accum_iter = 2 if args.batch_size <= 2 else max(1, args.accum_iter)
        args.use_amp = should_use_amp(args, device)
        print('[auto_tune] backbone=%s batch_size=%s lr=%s lr_backbone=%s accum_iter=%s amp=%s search_trials=%s' % (
            args.backbone, args.batch_size, args.lr, args.lr_backbone, args.accum_iter, args.use_amp, args.search_trials
        ))
    else:
        args.accum_iter = max(1, args.accum_iter)
        args.use_amp = should_use_amp(args, device)
        args.split_warmup_epochs = getattr(args, 'split_warmup_epochs', 5)
        args.count_loss_coef = getattr(args, 'count_loss_coef', 0.0)


def build_dataloaders(dataset_train, dataset_val, args):
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, seed=args.seed)
        sampler_val = DistributedSampler(dataset_val, shuffle=False, seed=args.seed)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train, generator=train_generator)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    worker_init_fn = seed_worker if args.deterministic else None

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                worker_init_fn=worker_init_fn, generator=train_generator)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                worker_init_fn=worker_init_fn, generator=val_generator)
    return data_loader_train, data_loader_val, sampler_train


def build_model_state(args, device, total_epochs):
    model, criterion = build_model(args)
    model.to(device)
    if args.syn_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup_epochs = min(5, max(1, total_epochs // 20))
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-7
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return model, criterion, model_without_ddp, optimizer, lr_scheduler, n_parameters


def get_backbone_candidates(reference_backbone):
    backbone_neighbors = {
        'convnextv2_nano': ['convnextv2_nano', 'convnextv2_small'],
        'convnextv2_small': ['convnextv2_nano', 'convnextv2_small', 'convnextv2_base'],
        'convnextv2_base': ['convnextv2_small', 'convnextv2_base'],
        'convnextv2_large': ['convnextv2_base', 'convnextv2_large'],
    }
    return backbone_neighbors.get(reference_backbone, [reference_backbone])


def random_log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_search_config(args, backbone_candidates, trial=None, rng=None):
    if rng is None:
        rng = random.Random(getattr(args, 'seed', 0))

    if trial is not None:
        backbone = trial.suggest_categorical('backbone', backbone_candidates)
        max_batch_size, base_lr, base_lr_backbone = get_convnextv2_training_defaults(backbone)
        batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8])
        batch_size = min(batch_size, max_batch_size)
        accum_iter = trial.suggest_categorical('accum_iter', [1, 2, 4, 8])
        lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
        lr_backbone = trial.suggest_float('lr_backbone', 1e-6, 5e-5, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        clip_max_norm = trial.suggest_categorical('clip_max_norm', [0.05, 0.1, 0.2])
        use_amp = False
        if not args.disable_amp and torch.cuda.is_available():
            use_amp = trial.suggest_categorical('use_amp', [False, True])
    else:
        backbone = rng.choice(backbone_candidates)
        max_batch_size, base_lr, base_lr_backbone = get_convnextv2_training_defaults(backbone)
        batch_candidates = sorted({max(1, max_batch_size // 2), max_batch_size})
        if max_batch_size == 1:
            batch_candidates.append(2)
        accum_candidates = [1, 2, 4]
        if max_batch_size <= 2:
            accum_candidates.append(8)
        accum_candidates = sorted(set(accum_candidates))
        amp_candidates = [False]
        if not args.disable_amp and torch.cuda.is_available():
            amp_candidates = [True, False]

        batch_size = rng.choice(batch_candidates)
        accum_iter = rng.choice(accum_candidates)
        lr = random_log_uniform(rng, base_lr / 4.0, base_lr * 4.0)
        lr_backbone = random_log_uniform(rng, base_lr_backbone / 4.0, base_lr_backbone * 4.0)
        weight_decay = random_log_uniform(rng, 1e-6, 1e-3)
        clip_max_norm = rng.choice([0.05, 0.1, 0.2])
        use_amp = rng.choice(amp_candidates)

    lr_backbone = min(lr_backbone, lr)
    return {
        'backbone': backbone,
        'batch_size': batch_size,
        'lr': lr,
        'lr_backbone': lr_backbone,
        'weight_decay': weight_decay,
        'clip_max_norm': clip_max_norm,
        'accum_iter': accum_iter,
        'use_amp': use_amp,
    }


def run_training(args, device, dataset_train, dataset_val, total_epochs, output_dir=None,
                 run_log_name=None, resume_checkpoint=None, save_checkpoints=True,
                 eval_freq=None, search_trial=None):
    model, criterion, model_without_ddp, optimizer, lr_scheduler, n_parameters = build_model_state(
        args, device, total_epochs
    )

    data_loader_train, data_loader_val, sampler_train = build_dataloaders(dataset_train, dataset_val, args)

    best_mae, best_epoch = 1e8, 0
    best_threshold = float(getattr(args, 'inference_threshold', 0.5))
    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint.get('best_mae', best_mae)
            best_epoch = checkpoint.get('best_epoch', best_epoch)
            best_threshold = checkpoint.get('best_threshold', best_threshold)
            args.inference_threshold = best_threshold
        if math.isfinite(args.target_mae) and best_mae <= args.target_mae:
            print('[auto_tune] resume checkpoint already meets target_mae=%.2f (best_mae=%.2f); stopping.' % (
                args.target_mae, best_mae
            ))
            return {'best_mae': best_mae, 'best_epoch': best_epoch, 'best_threshold': best_threshold, 'n_parameters': n_parameters}

    use_amp = getattr(args, 'use_amp', False) and device.type == 'cuda' and not args.disable_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    eval_freq = args.eval_freq if eval_freq is None else eval_freq
    eval_freq = max(1, int(eval_freq))

    if save_checkpoints and output_dir is not None and utils.is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, total_epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        t1 = time.time()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, use_amp=use_amp, accum_iter=args.accum_iter, scaler=scaler)
        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % (
              epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        if utils.is_main_process() and run_log_name is not None:
            append_text(run_log_name, '\n[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        lr_scheduler.step()

        if save_checkpoints and output_dir is not None:
            checkpoint_path = output_dir / 'checkpoint.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_mae': best_mae,
                'best_epoch': best_epoch,
                'best_threshold': best_threshold,
            }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if utils.is_main_process() and run_log_name is not None:
            append_text(run_log_name, json.dumps(log_stats) + "\n")

        if epoch % eval_freq == 0 and epoch > 0:
            t1 = time.time()
            test_stats = evaluate(
                model,
                data_loader_val,
                device,
                epoch,
                None,
                inference_threshold=args.inference_threshold,
                threshold_sweep=args.threshold_sweep,
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                threshold_step=args.threshold_step,
            )
            t2 = time.time()

            mae, mse = test_stats['mae'], test_stats['mse']
            eval_threshold = test_stats.get('threshold', args.inference_threshold)
            is_best = mae < best_mae or (math.isclose(mae, best_mae) and eval_threshold != best_threshold)
            if is_best:
                best_epoch = epoch
                best_mae = mae
                best_threshold = eval_threshold

            print("\n==========================")
            print("\nepoch:", epoch, "mae:", mae, "mse:", mse, "threshold:", eval_threshold,
                  "\n\nbest mae:", best_mae, "best epoch:", best_epoch, "best threshold:", best_threshold)
            print("==========================\n")
            if utils.is_main_process() and run_log_name is not None:
                append_text(run_log_name, "\nepoch:{}, mae:{}, mse:{}, threshold:{}, time{}, \n\nbest mae:{}, best epoch: {}, best threshold:{}\n\n".format(
                                                epoch, mae, mse, eval_threshold, t2 - t1, best_mae, best_epoch, best_threshold))

            if is_best and save_checkpoints and utils.is_main_process() and output_dir is not None:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'best_mae': best_mae,
                    'best_epoch': best_epoch,
                    'best_threshold': best_threshold,
                }, output_dir / 'best_checkpoint.pth')

            if math.isfinite(args.target_mae) and mae <= args.target_mae:
                print('[auto_tune] target_mae=%.2f reached with mae=%.2f; stopping early.' % (
                    args.target_mae, mae
                ))
                break

            if search_trial is not None and optuna is not None:
                search_trial.report(mae, epoch)
                if search_trial.should_prune():
                    raise optuna.TrialPruned()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return {'best_mae': best_mae, 'best_epoch': best_epoch, 'best_threshold': best_threshold, 'n_parameters': n_parameters}


def run_hyperparameter_search(args, device, dataset_train, dataset_val, output_dir):
    if args.search_trials <= 0:
        return {}

    backbone_candidates = get_backbone_candidates(args.backbone)
    rng = random.Random(args.seed)
    best_result = {}
    best_score = float('inf')

    def evaluate_config(trial=None):
        trial_args = copy.deepcopy(args)
        config = sample_search_config(trial_args, backbone_candidates, trial=trial, rng=rng)
        trial_args.backbone = config['backbone']
        trial_args.batch_size = config['batch_size']
        trial_args.lr = config['lr']
        trial_args.lr_backbone = config['lr_backbone']
        trial_args.weight_decay = config['weight_decay']
        trial_args.clip_max_norm = config['clip_max_norm']
        trial_args.accum_iter = config['accum_iter']
        trial_args.use_amp = config['use_amp']
        trial_args.start_epoch = 0
        trial_args.eval_freq = max(1, args.search_eval_freq)
        trial_args.epochs = args.search_epochs

        try:
            result = run_training(
                trial_args, device, dataset_train, dataset_val, args.search_epochs,
                output_dir=None, run_log_name=None, resume_checkpoint=None,
                save_checkpoints=False, eval_freq=args.search_eval_freq, search_trial=trial
            )
        except (NonFiniteTrainingError, RuntimeError, ValueError) as exc:
            exc_message = str(exc).lower()
            if (
                'out of memory' in exc_message
                or 'non-finite loss' in exc_message
                or 'invalid numeric entries' in exc_message
            ):
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                result = {'best_mae': float('inf'), 'best_epoch': -1, 'n_parameters': 0}
            else:
                raise

        result.update(config)
        return result

    if optuna is not None:
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1),
        )

        def objective(trial):
            result = evaluate_config(trial)
            trial.set_user_attr('config', {k: result[k] for k in ('backbone', 'batch_size', 'lr', 'lr_backbone', 'weight_decay', 'clip_max_norm', 'accum_iter', 'use_amp')})
            trial.set_user_attr('best_epoch', result['best_epoch'])
            trial.set_user_attr('best_threshold', result.get('best_threshold', args.inference_threshold))
            return result['best_mae']

        study.optimize(objective, n_trials=args.search_trials)
        best_trial = study.best_trial
        best_result = dict(best_trial.user_attrs['config'])
        best_result['best_mae'] = best_trial.value
        best_result['best_epoch'] = best_trial.user_attrs.get('best_epoch', -1)
        best_result['best_threshold'] = best_trial.user_attrs.get('best_threshold', args.inference_threshold)
    else:
        for _ in range(args.search_trials):
            result = evaluate_config(None)
            if result['best_mae'] < best_score:
                best_score = result['best_mae']
                best_result = result

    if best_result and output_dir is not None and utils.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        search_summary_path = output_dir / 'search_best_params.json'
        search_summary_path.write_text(json.dumps(best_result, indent=2, default=str), encoding='utf-8')

    if best_result:
        print('[auto_tune] best search config:', best_result)
    return best_result


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # Cố định seed ngẫu nhiên để kết quả thực nghiệm có thể tái lập giữa các lần chạy.
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_determinism(args)

    resume_checkpoint = None
    if args.resume:
        if args.resume.startswith('https'):
            resume_checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            resume_checkpoint = utils.load_checkpoint(args.resume, map_location='cpu')
        utils.restore_args_from_checkpoint(args, resume_checkpoint, fields=RESUME_ARG_FIELDS)
        if 'best_threshold' in resume_checkpoint:
            args.inference_threshold = resume_checkpoint['best_threshold']

    resolve_auto_tuning_defaults(args, device)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    output_dir = get_output_dir(args)
    run_log_name = output_dir / 'run_log.txt'
    if utils.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        append_text(run_log_name, 'Run Log ' + time.strftime("%c") + '\n' + str(args))

    if args.search_trials > 0 and not args.resume:
        best_search = run_hyperparameter_search(args, device, dataset_train, dataset_val, output_dir)
        if best_search:
            for key in ('backbone', 'batch_size', 'lr', 'lr_backbone', 'weight_decay', 'clip_max_norm', 'accum_iter', 'use_amp'):
                if key in best_search:
                    setattr(args, key, best_search[key])
            args.backbone = best_search.get('backbone', args.backbone)
            args.batch_size = int(best_search.get('batch_size', args.batch_size))
            args.lr = float(best_search.get('lr', args.lr))
            args.lr_backbone = float(best_search.get('lr_backbone', args.lr_backbone))
            args.weight_decay = float(best_search.get('weight_decay', args.weight_decay))
            args.clip_max_norm = float(best_search.get('clip_max_norm', args.clip_max_norm))
            args.accum_iter = max(1, int(best_search.get('accum_iter', args.accum_iter)))
            args.use_amp = bool(best_search.get('use_amp', args.use_amp))
            args.inference_threshold = float(best_search.get('best_threshold', args.inference_threshold))
            args.eval_freq = 1
            print('[auto_tune] using best search config:', best_search)

    train_result = run_training(
        args, device, dataset_train, dataset_val, args.epochs,
        output_dir=output_dir, run_log_name=run_log_name,
        resume_checkpoint=resume_checkpoint, save_checkpoints=True,
        eval_freq=args.eval_freq
    )
    print('[auto_tune] final result:', train_result)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser('PET training and evaluation script', parents=[get_args_parser()])
    cli_args = cli_parser.parse_args()
    try:
        main(cli_args)
    finally:
        utils.cleanup_distributed_mode()
