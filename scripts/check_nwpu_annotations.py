#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset
from datasets.NWPU import load_annotation


def inspect_split(dataset, image_set, max_examples, sigma_mode):
    total_points = 0
    sigma_samples = 0
    box_derived_samples = 0
    zero_point_samples = 0
    rows = []

    for image_id, img_path, json_path, mat_path in dataset.samples:
        ann = load_annotation(json_path=json_path, mat_path=mat_path, sigma_mode=sigma_mode)
        points = ann['points']
        point_count = int(points.shape[0])
        total_points += point_count
        if point_count == 0:
            zero_point_samples += 1
        sigma = ann.get('sigma')
        sigma_source = ann.get('sigma_source', 'missing')
        if sigma is not None:
            sigma_samples += 1
            if sigma_source.startswith('box_derived'):
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
    print(f'  nwpu sigma mode: {sigma_mode}')
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
    parser.add_argument('--eval_max_size', default=-1, type=int)
    parser.add_argument('--nwpu_eval_split', default='val', choices=('val', 'test', 'train'))
    parser.add_argument('--nwpu_sigma_mode', default='official', choices=('area', 'diag', 'min_diag', 'official'),
                        help='fallback localization sigma derived from boxes when annotation sigma is absent')
    parser.add_argument('--splits', default='train,val',
                        help='comma-separated logical splits to check; use val to skip train')
    parser.add_argument('--max_examples', default=5, type=int)
    args = parser.parse_args()

    requested_splits = [split.strip() for split in args.splits.split(',') if split.strip()]
    if not requested_splits:
        requested_splits = ['val']

    totals = {}
    for split in requested_splits:
        if split == 'train':
            dataset = build_dataset('train', args)
            display = 'train'
        elif split in ('val', 'test'):
            old_eval_split = args.nwpu_eval_split
            args.nwpu_eval_split = split
            dataset = build_dataset('val', args)
            args.nwpu_eval_split = old_eval_split
            display = split
        else:
            raise SystemExit(f'Unsupported split in --splits: {split}')
        totals[split] = inspect_split(dataset, display, args.max_examples, args.nwpu_sigma_mode)

    if any(sample_count == 0 for _points, sample_count, _sigma in totals.values()):
        raise SystemExit('Invalid NWPU dataset: at least one requested split is empty.')
    if any(point_count == 0 for point_count, _samples, _sigma in totals.values()):
        print('WARNING: at least one requested split has zero total points. NWPU test labels may be unavailable.')
    checked_sigma = sum(sigma_count for _points, _samples, sigma_count in totals.values())
    if checked_sigma == 0:
        print(
            'WARNING: no sigma thresholds found for requested split(s). '
            'Use --localization_protocol adaptive_nn or prepare official NWPU localization GT.'
        )
    else:
        print('NWPU annotation check passed.')


if __name__ == '__main__':
    main()
