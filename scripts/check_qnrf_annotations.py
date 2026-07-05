#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset
from datasets.QNRF import find_annotation_path, load_points, load_raw_points_xy


EXPECTED_SPLIT_IMAGES = {'train': 1201, 'val': 334}
EXPECTED_TOTAL_POINTS = 1_251_642


def inspect_split(dataset, image_set, max_examples):
    split_dir = Path(dataset.split_dir)

    rows = []
    missing = []
    total_points = 0
    raw_total_points = 0
    zero_point_files = 0
    outside_points = 0
    count_mismatches = 0
    for index, img_path_str in enumerate(dataset.img_list, start=1):
        img_path = Path(img_path_str)
        gt_path = Path(dataset.gt_list[img_path_str])
        resolved_gt = find_annotation_path(str(img_path))
        if resolved_gt is None or not gt_path.exists():
            missing.append(gt_path)
            point_count = 0
        else:
            with Image.open(img_path) as image:
                width, height = image.size
            raw_points = load_raw_points_xy(str(gt_path))
            points = load_points(str(gt_path), image_size=(height, width))
            raw_count = int(raw_points.shape[0])
            point_count = int(points.shape[0])
            raw_total_points += raw_count
            total_points += point_count
            count_mismatches += int(raw_count != point_count)
            outside_points += int(
                (
                    (raw_points[:, 0] < 0)
                    | (raw_points[:, 0] >= width)
                    | (raw_points[:, 1] < 0)
                    | (raw_points[:, 1] >= height)
                ).sum()
            )
            if point_count == 0:
                zero_point_files += 1
        if len(rows) < max_examples:
            rows.append((img_path.name, gt_path.name, point_count, gt_path.exists()))
        if index % 100 == 0 or index == len(dataset.img_list):
            print(
                f'  scanning {image_set}: {index}/{len(dataset.img_list)}',
                end='\r',
                flush=True,
            )
    print(' ' * 72, end='\r')

    print(f'{image_set}:')
    print(f'  root: {dataset.root_path}')
    print(f'  images/annotations: {split_dir}')
    print(f'  image files: {len(dataset.img_list)}')
    print(f'  missing annotation files: {len(missing)}')
    print(f'  zero-point annotation files: {zero_point_files}')
    print(f'  raw annPoints: {raw_total_points}')
    print(f'  total GT points: {total_points}')
    print(f'  raw/loaded count mismatches: {count_mismatches}')
    print(f'  finite points outside image bounds: {outside_points}')
    if rows:
        print('  examples:')
        for img_name, gt_name, point_count, exists in rows:
            status = 'ok' if exists else 'missing'
            print(f'    {img_name} -> {gt_name}: {point_count} points ({status})')
    if missing:
        print('  first missing annotations:')
        for gt_path in missing[:max_examples]:
            print(f'    {gt_path}')

    return {
        'images': len(dataset.img_list),
        'raw_points': raw_total_points,
        'loaded_points': total_points,
        'missing': len(missing),
        'zero': zero_point_files,
        'count_mismatches': count_mismatches,
        'outside_points': outside_points,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate UCF-QNRF annotation loading.')
    parser.add_argument('--dataset_file', default='QNRF', choices=('QNRF',))
    parser.add_argument('--data_path', default='./data/UCF-QNRF_ECCV18')
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--crop_attempts', default=1, type=int)
    parser.add_argument('--min_crop_points', default=0, type=int)
    parser.add_argument('--max_examples', default=5, type=int)
    parser.add_argument(
        '--allow_noncanonical',
        action='store_true',
        help='report official split/count mismatches without failing',
    )
    args = parser.parse_args()

    train = build_dataset('train', args)
    val = build_dataset('val', args)

    train_stats = inspect_split(train, 'train', args.max_examples)
    val_stats = inspect_split(val, 'val', args.max_examples)

    if train_stats['missing'] or val_stats['missing']:
        raise SystemExit('Invalid dataset: at least one annotation file is missing.')
    if train_stats['loaded_points'] == 0 or val_stats['loaded_points'] == 0:
        raise SystemExit('Invalid dataset: total GT points is zero for train or val.')
    if train_stats['zero'] or val_stats['zero']:
        raise SystemExit('Invalid dataset: at least one annotation file has zero points.')
    if train_stats['count_mismatches'] or val_stats['count_mismatches']:
        raise SystemExit('Invalid dataset: raw annPoints counts changed during loading.')

    canonical_errors = []
    for split_name, stats in (('train', train_stats), ('val', val_stats)):
        expected_images = EXPECTED_SPLIT_IMAGES[split_name]
        if stats['images'] != expected_images:
            canonical_errors.append(
                f'{split_name} has {stats["images"]} images, expected {expected_images}'
            )
    combined_points = train_stats['raw_points'] + val_stats['raw_points']
    if combined_points != EXPECTED_TOTAL_POINTS:
        canonical_errors.append(
            f'combined annotations={combined_points}, expected {EXPECTED_TOTAL_POINTS}'
        )
    if canonical_errors:
        message = 'Noncanonical UCF-QNRF data: ' + '; '.join(canonical_errors)
        if not args.allow_noncanonical:
            raise SystemExit(message)
        print(f'WARNING: {message}')

    print('Annotation check passed.')


if __name__ == '__main__':
    main()
