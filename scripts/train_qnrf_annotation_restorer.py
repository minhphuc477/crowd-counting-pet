import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as torch_mp
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.point_restoration import (  # noqa: E402
    QNRFShiftRestorationDataset,
    build_holdout_indices,
    qnrf_train_samples,
    restoration_collate,
)
from models.annotation_restorer import (  # noqa: E402
    VGG16BNAnnotationRestorer,
    sample_vector_field,
)

try:
    torch_mp.set_sharing_strategy('file_system')
except RuntimeError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a count-preserving shifted annotation restorer on QNRF.',
    )
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--crop_size', default=512, type=int)
    parser.add_argument('--crop_attempts', default=8, type=int)
    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--scale_min', default=0.7, type=float)
    parser.add_argument('--scale_max', default=1.3, type=float)
    parser.add_argument('--holdout_fraction', default=0.1, type=float)
    parser.add_argument('--holdout_seed', default=42, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no_pretrained_backbone', action='store_true')
    parser.add_argument('--resume', default='')
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--max_grad_norm', default=5.0, type=float)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def batch_restoration_loss(vector_fields, batch, device):
    image_losses = []
    point_count = 0
    for batch_index, sample in enumerate(batch):
        points = sample['shifted_points_yx'].to(device=device, dtype=torch.float32)
        target = sample['inverse_shift_yx'].to(device=device, dtype=torch.float32)
        prediction = sample_vector_field(vector_fields[batch_index].float(), points)
        image_losses.append((prediction - target).square().sum(dim=1).mean())
        point_count += int(points.shape[0])
    if not image_losses:
        raise RuntimeError('annotation restoration batch contains no supervised points')
    return torch.stack(image_losses).mean(), point_count


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.accum_iter <= 0:
        raise ValueError('epochs, batch_size, and accum_iter must be positive')
    if args.save_freq <= 0:
        raise ValueError('save_freq must be positive')
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples = qnrf_train_samples(args.data_path)
    train_indices, holdout_indices = build_holdout_indices(
        len(all_samples),
        args.holdout_fraction,
        args.holdout_seed,
    )
    split_manifest = {
        'dataset': 'QNRF',
        'data_path': str(Path(args.data_path).resolve()),
        'holdout_fraction': float(args.holdout_fraction),
        'holdout_seed': int(args.holdout_seed),
        'train_images': [Path(all_samples[index][0]).name for index in train_indices],
        'holdout_images': [Path(all_samples[index][0]).name for index in holdout_indices],
    }
    (output_dir / 'split_manifest.json').write_text(
        json.dumps(split_manifest, indent=2),
        encoding='utf-8',
    )

    dataset = QNRFShiftRestorationDataset(
        args.data_path,
        indices=train_indices,
        crop_size=args.crop_size,
        alpha=args.alpha,
        scale_range=(args.scale_min, args.scale_max),
        crop_attempts=args.crop_attempts,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=restoration_collate,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    model = VGG16BNAnnotationRestorer(
        pretrained=not args.no_pretrained_backbone,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(args.amp and device.type == 'cuda')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('split_manifest') != split_manifest:
            raise ValueError('resume checkpoint uses a different QNRF holdout split')
        checkpoint_args = checkpoint.get('args', {})
        for key in ('crop_size', 'alpha', 'scale_min', 'scale_max'):
            if checkpoint_args.get(key, getattr(args, key)) != getattr(args, key):
                raise ValueError(f'resume checkpoint uses a different {key}')
        if checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = int(checkpoint['epoch']) + 1

    def checkpoint_payload(epoch, train_loss):
        return {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': int(epoch),
            'train_loss': float(train_loss),
            'args': vars(args),
            'split_manifest': split_manifest,
            'vector_order': 'yx',
        }

    history_path = output_dir / 'train_history.jsonl'
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_images = 0
        running_points = 0
        started = time.time()
        for step, (images, batch) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                vector_fields = model(images)
                loss, point_count = batch_restoration_loss(vector_fields, batch, device)
            group_start = (step // args.accum_iter) * args.accum_iter
            group_size = min(args.accum_iter, len(loader) - group_start)
            scaler.scale(loss / group_size).backward()
            should_step = (step + 1) % args.accum_iter == 0 or step + 1 == len(loader)
            if should_step:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_images = len(batch)
            running_loss += float(loss.detach()) * batch_images
            running_images += batch_images
            running_points += point_count

        train_loss = running_loss / max(running_images, 1)
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'images': running_images,
            'points': running_points,
            'seconds': time.time() - started,
        }
        with history_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        payload = checkpoint_payload(epoch, train_loss)
        atomic_torch_save(payload, output_dir / 'checkpoint.pth')
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            atomic_torch_save(payload, output_dir / f'checkpoint_{epoch + 1:03d}.pth')

    final = {
        'epoch': args.epochs - 1,
        'checkpoint': str(output_dir / 'checkpoint.pth'),
        'train_images': len(train_indices),
        'holdout_images': len(holdout_indices),
        'holdout_fraction': args.holdout_fraction,
        'holdout_seed': args.holdout_seed,
    }
    (output_dir / 'final_results.json').write_text(
        json.dumps(final, indent=2),
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
