import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.io import savemat


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.image_io import load_rgb_image  # noqa: E402
from datasets.point_restoration import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    qnrf_train_samples,
    restoration_radii,
)
from datasets.QNRF import load_raw_points_xy  # noqa: E402
from models.annotation_restorer import (  # noqa: E402
    VGG16BNAnnotationRestorer,
    sample_vector_field,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply a trained QNRF annotation restorer without changing counts.',
    )
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tile_size', default=1024, type=int)
    parser.add_argument('--tile_overlap', default=128, type=int)
    parser.add_argument('--max_radius_scale', default=1.0, type=float)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--print_freq', default=10, type=int)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError('tile_overlap must be smaller than tile_size')
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def assign_points_to_tiles(points_xy, height, width, tile_size, overlap):
    tiles = [
        (top, left, min(tile_size, height - top), min(tile_size, width - left))
        for top in tile_starts(height, tile_size, overlap)
        for left in tile_starts(width, tile_size, overlap)
    ]
    owners = np.full(points_xy.shape[0], -1, dtype=np.int64)
    best_margin = np.full(points_xy.shape[0], -np.inf, dtype=np.float32)
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    for tile_index, (top, left, tile_height, tile_width) in enumerate(tiles):
        inside = (
            (x >= left)
            & (x <= left + tile_width - 1)
            & (y >= top)
            & (y <= top + tile_height - 1)
        )
        if not np.any(inside):
            continue
        margin = np.minimum.reduce((
            x - left,
            left + tile_width - 1 - x,
            y - top,
            top + tile_height - 1 - y,
        ))
        update = inside & (margin > best_margin)
        owners[update] = tile_index
        best_margin[update] = margin[update]
    if np.any(owners < 0):
        raise RuntimeError('at least one QNRF point was not assigned to a restoration tile')
    return tiles, owners


def clamp_displacements(displacements_yx, radii, max_radius_scale):
    if max_radius_scale <= 0:
        return displacements_yx, 0
    limits = torch.as_tensor(
        radii * float(max_radius_scale),
        device=displacements_yx.device,
        dtype=displacements_yx.dtype,
    )
    norms = torch.linalg.vector_norm(displacements_yx, dim=1)
    scale = torch.minimum(torch.ones_like(norms), limits / norms.clamp_min(1e-6))
    clipped = int((scale < 1.0).sum().item())
    return displacements_yx * scale[:, None], clipped


def atomic_savemat(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    savemat(str(temporary), payload, appendmat=False)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.tile_size < 32:
        raise ValueError('tile_size must be at least 32')
    if args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
        raise ValueError('tile_overlap must be in [0, tile_size)')
    if args.print_freq <= 0:
        raise ValueError('print_freq must be positive')

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    if checkpoint.get('vector_order', 'yx') != 'yx':
        raise ValueError('unsupported annotation-restorer vector order')
    model = VGG16BNAnnotationRestorer(pretrained=False)
    model.load_state_dict(checkpoint['model'])
    device = torch.device(args.device)
    model.to(device).eval()
    amp_enabled = bool(args.amp and device.type == 'cuda')
    alpha = float(checkpoint.get('args', {}).get('alpha', 0.4))

    samples = qnrf_train_samples(args.data_path)
    split_manifest = checkpoint.get('split_manifest')
    if not isinstance(split_manifest, dict):
        raise ValueError('restorer checkpoint is missing its training split manifest')
    checkpoint_images = set(split_manifest.get('train_images', ())) | set(
        split_manifest.get('holdout_images', ())
    )
    current_images = {Path(image_path).name for image_path, _ in samples}
    if checkpoint_images != current_images:
        raise ValueError('restorer checkpoint and QNRF training image sets differ')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total_points = 0
    total_clipped = 0
    total_displacement = 0.0
    maximum_displacement = 0.0
    with torch.inference_mode():
        for sample_index, (image_path, annotation_path) in enumerate(samples):
            image = load_rgb_image(image_path)
            width, height = image.size
            original = load_raw_points_xy(annotation_path)
            if original.shape[0] == 0:
                raise ValueError(f'QNRF restoration sample has no points: {annotation_path}')
            points = original.copy()
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            radii = restoration_radii(points, alpha=alpha)
            tiles, owners = assign_points_to_tiles(
                points,
                height,
                width,
                args.tile_size,
                args.tile_overlap,
            )
            displacement = np.zeros_like(points, dtype=np.float32)
            sample_clipped = 0
            for tile_index, (top, left, tile_height, tile_width) in enumerate(tiles):
                point_indices = np.flatnonzero(owners == tile_index)
                if point_indices.size == 0:
                    continue
                tile = image.crop((left, top, left + tile_width, top + tile_height))
                image_tensor = TF.normalize(
                    TF.to_tensor(tile),
                    IMAGENET_MEAN,
                    IMAGENET_STD,
                ).unsqueeze(0).to(device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    vector_field = model(image_tensor)
                local_yx = points[point_indices][:, ::-1].copy()
                local_yx[:, 0] -= top
                local_yx[:, 1] -= left
                local_yx_tensor = torch.from_numpy(local_yx).to(device=device, dtype=torch.float32)
                predicted_yx = sample_vector_field(vector_field.float(), local_yx_tensor)
                predicted_yx, clipped = clamp_displacements(
                    predicted_yx,
                    radii[point_indices],
                    args.max_radius_scale,
                )
                sample_clipped += clipped
                displacement[point_indices] = predicted_yx.cpu().numpy()[:, ::-1]

            restored = points + displacement
            restored[:, 0] = np.clip(restored[:, 0], 0, width - 1)
            restored[:, 1] = np.clip(restored[:, 1], 0, height - 1)
            if restored.shape[0] != original.shape[0] or not np.isfinite(restored).all():
                raise RuntimeError(f'invalid restored annotations for {image_path}')
            norms = np.linalg.norm(restored - points, axis=1)
            output_path = output_dir / f'{Path(image_path).stem}_ann.mat'
            atomic_savemat(
                output_path,
                {
                    'annPoints': restored.astype(np.float32),
                    'originalPoints': original.astype(np.float32),
                    'restorationVector': (restored - points).astype(np.float32),
                },
            )
            total_points += int(restored.shape[0])
            total_clipped += sample_clipped
            total_displacement += float(norms.sum())
            sample_max_displacement = float(norms.max())
            maximum_displacement = max(maximum_displacement, sample_max_displacement)
            rows.append({
                'image': Path(image_path).name,
                'annotation': Path(annotation_path).name,
                'output': output_path.name,
                'output_sha256': file_sha256(output_path),
                'points': int(restored.shape[0]),
                'mean_displacement': float(norms.mean()),
                'max_displacement': sample_max_displacement,
                'clipped_vectors': sample_clipped,
            })
            if (sample_index + 1) % args.print_freq == 0 or sample_index + 1 == len(samples):
                print(
                    f'[{sample_index + 1}/{len(samples)}] {Path(image_path).name}: '
                    f'points={restored.shape[0]} mean_shift={norms.mean():.3f} '
                    f'clipped={sample_clipped}',
                    flush=True,
                )

    manifest = {
        'dataset': 'QNRF',
        'method': 'shifted_annotation_restoration',
        'data_path': str(Path(args.data_path).resolve()),
        'restorer_checkpoint': str(Path(args.resume).resolve()),
        'restorer_checkpoint_sha256': file_sha256(args.resume),
        'alpha': alpha,
        'tile_size': args.tile_size,
        'tile_overlap': args.tile_overlap,
        'max_radius_scale': args.max_radius_scale,
        'holdout_fraction': float(split_manifest['holdout_fraction']),
        'holdout_seed': int(split_manifest['holdout_seed']),
        'train_images': split_manifest['train_images'],
        'holdout_images': split_manifest['holdout_images'],
        'images': len(rows),
        'points': total_points,
        'mean_displacement': total_displacement / max(total_points, 1),
        'max_displacement': maximum_displacement,
        'clipped_vectors': total_clipped,
        'count_preserved': True,
        'samples': rows,
    }
    (output_dir / 'annotation_override_manifest.json').write_text(
        json.dumps(manifest, indent=2),
        encoding='utf-8',
    )
    print(f'Restored {total_points} points in {len(rows)} images to {output_dir}')


if __name__ == '__main__':
    main()
