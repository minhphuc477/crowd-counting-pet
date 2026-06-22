#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset
from datasets.NWPU import load_annotation


def inspect_split(dataset, image_set, max_examples):
    total_points = 0
    sigma_samples = 0
    box_derived_samples = 0
    zero_point_samples = 0
    rows = []

    for image_id, img_path, json_path, mat_path in dataset.samples:
        ann = load_annotation(json_path=json_path, mat_path=mat_path)
        points = ann['points']
        point_count = int(points.shape[0])
        total_points += point_count
        if point_count == 0:
            zero_point_samples += 1
        sigma = ann.get('sigma')
        sigma_source = ann.get('sigma_source', 'missing')
        if sigma is not None:
            sigma_samples += 1
            if sigma_source == 'box_derived':
                box_derived_samples += 1
        if len(rows) < max_examples:
            if sigma is not None and sigma.shape[0] > 0:
                sigma_summary = (
                    f'{sigma_source}, mean_s={float(sigma[:, 0].mean()):.2f}, '
                    f'mean_l={float(sigma[:, 1].mean()):.2f}'
                )
            else:
                sigma_summary = 'missing'
            rows.append((image_id, Path(img_path).name, point_count, sigma_summary))

    print(f'{image_set}:')
    print(f'  root: {dataset.root_path}')
    print(f'  images: {dataset.images_dir}')
    print(f'  samples: {len(dataset.samples)}')
    print(f'  total GT points: {total_points}')
    print(f'  zero-point samples: {zero_point_samples}')
    print(f'  samples with sigma: {sigma_samples}')
    print(f'  sigma derived from boxes: {box_derived_samples}')
    if rows:
        print('  examples:')
        for image_id, img_name, point_count, sigma_summary in rows:
            print(f'    {image_id} {img_name}: {point_count} points, sigma={sigma_summary}')

    return total_points, len(dataset.samples), sigma_samples


def main():
    parser = argparse.ArgumentParser(description='Validate NWPU-Crowd annotation loading.')
    parser.add_argument('--dataset_file', default='NWPU')
    parser.add_argument('--data_path', default='./data/NWPU-Crowd')
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--crop_attempts', default=1, type=int)
    parser.add_argument('--min_crop_points', default=0, type=int)
    parser.add_argument('--eval_max_size', default=1536, type=int)
    parser.add_argument('--nwpu_eval_split', default='val', choices=('val', 'test', 'train'))
    parser.add_argument('--max_examples', default=5, type=int)
    args = parser.parse_args()

    train = build_dataset('train', args)
    val = build_dataset('val', args)

    train_points, train_samples, train_sigma = inspect_split(train, 'train', args.max_examples)
    val_points, val_samples, val_sigma = inspect_split(val, args.nwpu_eval_split, args.max_examples)

    if train_samples == 0 or val_samples == 0:
        raise SystemExit('Invalid NWPU dataset: train or validation split is empty.')
    if train_points == 0 or val_points == 0:
        print('WARNING: train or validation split has zero total points. NWPU test labels may be unavailable.')
    if val_sigma == 0:
        print(
            'WARNING: no sigma thresholds found for validation. '
            'Use --localization_protocol adaptive_nn or prepare official NWPU localization GT.'
        )
    else:
        print('NWPU annotation check passed.')


if __name__ == '__main__':
    main()
