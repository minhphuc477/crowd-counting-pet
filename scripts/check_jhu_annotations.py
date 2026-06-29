#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset
from datasets.JHU import load_jhu_annotation


def inspect_split(dataset, name, max_examples):
    total_points = 0
    zero_point_images = 0
    examples = []
    for image_path, annotation_path in dataset.samples:
        points, sigma = load_jhu_annotation(annotation_path)
        count = int(points.shape[0])
        total_points += count
        zero_point_images += int(count == 0)
        if sigma.shape != points.shape:
            raise ValueError(
                f'JHU sigma shape mismatch for {annotation_path}: '
                f'{sigma.shape} versus {points.shape}'
            )
        if len(examples) < max_examples:
            examples.append((image_path.name, annotation_path.name, count))

    print(f'{name}:')
    print(f'  root: {dataset.root_path}')
    print(f'  images: {dataset.image_dir}')
    print(f'  annotations: {dataset.gt_dir}')
    print(f'  samples: {len(dataset)}')
    print(f'  total GT points: {total_points}')
    print(f'  zero-point distractors: {zero_point_images}')
    print('  examples:')
    for image_name, annotation_name, count in examples:
        print(f'    {image_name} -> {annotation_name}: {count} points')
    if total_points == 0:
        raise ValueError(f'JHU split {name} has no points')


def main():
    parser = argparse.ArgumentParser(
        description='Validate the official JHU-Crowd++ six-field annotations.'
    )
    parser.add_argument('--dataset_file', default='JHU')
    parser.add_argument('--data_path', default='./data/jhu_crowd_v2.0')
    parser.add_argument('--jhu_eval_split', default='val', choices=('val', 'test', 'train'))
    parser.add_argument('--eval_max_size', default=-1, type=int)
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--patch_size_choices', default='')
    parser.add_argument('--crop_attempts', default=1, type=int)
    parser.add_argument('--min_crop_points', default=0, type=int)
    parser.add_argument('--max_examples', default=5, type=int)
    args = parser.parse_args()

    train = build_dataset('train', args)
    validation = build_dataset('val', args)
    inspect_split(train, 'train', args.max_examples)
    inspect_split(validation, args.jhu_eval_split, args.max_examples)
    print('JHU-Crowd++ annotation check passed.')


if __name__ == '__main__':
    main()
