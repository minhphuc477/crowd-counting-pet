#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset
from datasets.QNRF import find_annotation_path, load_points


def inspect_split(dataset, image_set, max_examples):
    split_dir = Path(dataset.split_dir)

    rows = []
    missing = []
    total_points = 0
    zero_point_files = 0
    for img_path_str in dataset.img_list:
        img_path = Path(img_path_str)
        gt_path = Path(dataset.gt_list[img_path_str])
        resolved_gt = find_annotation_path(str(img_path))
        if resolved_gt is None or not gt_path.exists():
            missing.append(gt_path)
            point_count = 0
        else:
            points = load_points(str(gt_path))
            point_count = int(points.shape[0])
            total_points += point_count
            if point_count == 0:
                zero_point_files += 1
        if len(rows) < max_examples:
            rows.append((img_path.name, gt_path.name, point_count, gt_path.exists()))

    print(f'{image_set}:')
    print(f'  root: {dataset.root_path}')
    print(f'  images/annotations: {split_dir}')
    print(f'  image files: {len(dataset.img_list)}')
    print(f'  missing annotation files: {len(missing)}')
    print(f'  zero-point annotation files: {zero_point_files}')
    print(f'  total GT points: {total_points}')
    if rows:
        print('  examples:')
        for img_name, gt_name, point_count, exists in rows:
            status = 'ok' if exists else 'missing'
            print(f'    {img_name} -> {gt_name}: {point_count} points ({status})')
    if missing:
        print('  first missing annotations:')
        for gt_path in missing[:max_examples]:
            print(f'    {gt_path}')

    return total_points, len(missing), zero_point_files


def main():
    parser = argparse.ArgumentParser(description='Validate UCF-QNRF annotation loading.')
    parser.add_argument('--dataset_file', default='QNRF', choices=('QNRF', 'UCF'))
    parser.add_argument('--data_path', default='./data/UCF-QNRF_ECCV18')
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--crop_attempts', default=1, type=int)
    parser.add_argument('--min_crop_points', default=0, type=int)
    parser.add_argument('--max_examples', default=5, type=int)
    args = parser.parse_args()

    train = build_dataset('train', args)
    val = build_dataset('val', args)

    train_points, train_missing, train_zero = inspect_split(train, 'train', args.max_examples)
    val_points, val_missing, val_zero = inspect_split(val, 'val', args.max_examples)

    if train_missing or val_missing:
        raise SystemExit('Invalid dataset: at least one annotation file is missing.')
    if train_points == 0 or val_points == 0:
        raise SystemExit('Invalid dataset: total GT points is zero for train or val.')
    if train_zero or val_zero:
        raise SystemExit('Invalid dataset: at least one annotation file has zero points.')

    print('Annotation check passed.')


if __name__ == '__main__':
    main()
