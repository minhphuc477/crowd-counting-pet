import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.SHA import (  # noqa: E402
    IMAGE_EXTENSIONS,
    find_ground_truth_dir,
    fixed_partial_annotation_mask,
    load_points,
)


def select_completed_points(
    gt_points,
    pred_points,
    pred_scores,
    bounds,
    image_shape,
    band_pixels,
    min_score,
    dedup_radius,
):
    height, width = image_shape
    top, left, bottom, right = bounds
    gt_points = np.asarray(gt_points, dtype=np.float32).reshape(-1, 2)
    pred_points = np.asarray(pred_points, dtype=np.float32).reshape(-1, 2)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    if pred_points.shape[0] != pred_scores.shape[0]:
        raise ValueError('prediction point and score counts differ')

    annotated = (
        (gt_points[:, 0] >= top)
        & (gt_points[:, 0] < bottom)
        & (gt_points[:, 1] >= left)
        & (gt_points[:, 1] < right)
    )
    observed_gt = gt_points[annotated]

    outside = ~(
        (pred_points[:, 0] >= top)
        & (pred_points[:, 0] < bottom)
        & (pred_points[:, 1] >= left)
        & (pred_points[:, 1] < right)
    )
    inside_image = (
        (pred_points[:, 0] >= 0)
        & (pred_points[:, 0] < height)
        & (pred_points[:, 1] >= 0)
        & (pred_points[:, 1] < width)
    )
    keep = outside & inside_image & (pred_scores >= float(min_score))
    band_pixels = int(band_pixels)
    if band_pixels > 0:
        band_top = max(0, top - band_pixels)
        band_left = max(0, left - band_pixels)
        band_bottom = min(height, bottom + band_pixels)
        band_right = min(width, right + band_pixels)
        keep &= (
            (pred_points[:, 0] >= band_top)
            & (pred_points[:, 0] < band_bottom)
            & (pred_points[:, 1] >= band_left)
            & (pred_points[:, 1] < band_right)
        )

    pseudo = pred_points[keep]
    pseudo_scores = pred_scores[keep]
    if observed_gt.shape[0] and pseudo.shape[0] and dedup_radius > 0:
        distances = np.sqrt(
            ((pseudo[:, None, :] - observed_gt[None, :, :]) ** 2).sum(axis=2)
        )
        distinct = distances.min(axis=1) > float(dedup_radius)
        pseudo = pseudo[distinct]
        pseudo_scores = pseudo_scores[distinct]

    completed = np.concatenate([observed_gt, pseudo], axis=0)
    is_pseudo = np.concatenate([
        np.zeros(observed_gt.shape[0], dtype=np.uint8),
        np.ones(pseudo.shape[0], dtype=np.uint8),
    ])
    scores = np.concatenate([
        np.ones(observed_gt.shape[0], dtype=np.float32),
        pseudo_scores,
    ])
    return completed, is_pseudo, scores


def generate_predictions(args, predictions_path):
    command = [
        sys.executable,
        REPO_ROOT / 'eval.py',
        '--resume', args.resume,
        '--dataset_file', args.dataset_file,
        '--data_path', args.data_path,
        '--eval_image_set', 'train_eval',
        '--device', args.device,
        '--num_workers', args.num_workers,
        '--no_localization_metrics',
        '--per_image_predictions_file', predictions_path,
        '--results_file', predictions_path.with_suffix('.metrics.json'),
    ]
    if args.score_threshold is not None:
        command.extend([
            '--override_score_threshold',
            str(args.score_threshold),
        ])
    print('+', ' '.join(str(part) for part in command), flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=REPO_ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Complete ShanghaiTech partial annotations using a trained PET '
            'checkpoint and write auditable MAT files for stage-two training'
        ),
    )
    parser.add_argument('--resume', required=True)
    parser.add_argument('--dataset_file', choices=('SHA', 'SHB'), required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--predictions_json', default='')
    parser.add_argument('--partial_annotation_ratio', default=0.1, type=float)
    parser.add_argument('--partial_annotation_seed', default=0, type=int)
    parser.add_argument('--partial_annotation_height_ratio', default=0.5, type=float)
    parser.add_argument(
        '--inference_band_pixels',
        default=256,
        type=int,
        help='expand the annotated rectangle by this many pixels; 0 uses all unannotated regions',
    )
    parser.add_argument('--score_threshold', default=None, type=float)
    parser.add_argument('--min_pseudo_score', default=0.0, type=float)
    parser.add_argument('--dedup_radius', default=4.0, type=float)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=2, type=int)
    args = parser.parse_args()

    if not 0.0 < args.partial_annotation_ratio < 1.0:
        raise ValueError('--partial_annotation_ratio must be in (0, 1)')
    if args.inference_band_pixels < 0:
        raise ValueError('--inference_band_pixels must be non-negative')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        Path(args.predictions_json)
        if args.predictions_json
        else output_dir / 'stage1_predictions.json'
    )
    if not args.predictions_json:
        generate_predictions(args, predictions_path)

    with predictions_path.open('r', encoding='utf-8') as handle:
        rows = json.load(handle)
    predictions = {
        Path(row['image_path']).stem: row
        for row in rows
    }

    split_dir = Path(args.data_path) / 'train_data'
    images_dir = split_dir / 'images'
    gt_dir = Path(find_ground_truth_dir(str(split_dir)))
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    summary = []
    for image_path in image_paths:
        stem = image_path.stem
        if stem not in predictions:
            raise ValueError(f'Missing stage-one prediction for {image_path}')
        gt_path = gt_dir / f'GT_{stem}.mat'
        with Image.open(image_path) as image:
            width, height = image.size
        gt_points = load_points(gt_path, image_size=(height, width))
        _, bounds = fixed_partial_annotation_mask(
            (height, width),
            image_path,
            ratio=args.partial_annotation_ratio,
            seed=args.partial_annotation_seed,
            height_ratio=args.partial_annotation_height_ratio,
        )
        row = predictions[stem]
        if (
            int(row['image_height']) != height
            or int(row['image_width']) != width
        ):
            raise ValueError(
                f'Prediction/image size mismatch for {image_path}: '
                f"{row['image_height']}x{row['image_width']} vs {height}x{width}"
            )
        completed, is_pseudo, scores = select_completed_points(
            gt_points,
            row['pred_points_yx'],
            row['pred_scores'],
            bounds,
            (height, width),
            args.inference_band_pixels,
            args.min_pseudo_score,
            args.dedup_radius,
        )
        output_path = output_dir / f'GT_{stem}.mat'
        savemat(
            output_path,
            {
                # MAT crowd datasets use (x, y); PET tensors use (y, x).
                'annPoints': completed[:, ::-1].astype(np.float32),
                'isPseudo': is_pseudo.reshape(-1, 1),
                'pointConfidence': scores.reshape(-1, 1),
                'partialBoundsYX': np.asarray(bounds, dtype=np.int32),
                'partialAnnotationRatio': np.asarray(
                    [[args.partial_annotation_ratio]],
                    dtype=np.float32,
                ),
                'partialAnnotationSeed': np.asarray(
                    [[args.partial_annotation_seed]],
                    dtype=np.int32,
                ),
                'inferenceBandPixels': np.asarray(
                    [[args.inference_band_pixels]],
                    dtype=np.int32,
                ),
            },
        )
        pseudo_count = int(is_pseudo.sum())
        summary.append({
            'image': image_path.name,
            'observed_gt': int(completed.shape[0] - pseudo_count),
            'pseudo': pseudo_count,
            'completed': int(completed.shape[0]),
        })

    (output_dir / 'completion_manifest.json').write_text(
        json.dumps(
            {
                'dataset': args.dataset_file,
                'checkpoint': args.resume,
                'predictions': str(predictions_path),
                'partial_annotation_ratio': args.partial_annotation_ratio,
                'partial_annotation_seed': args.partial_annotation_seed,
                'partial_annotation_height_ratio': args.partial_annotation_height_ratio,
                'inference_band_pixels': args.inference_band_pixels,
                'min_pseudo_score': args.min_pseudo_score,
                'dedup_radius': args.dedup_radius,
                'samples': summary,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(f'Wrote {len(summary)} completed annotation files to {output_dir}')


if __name__ == '__main__':
    main()
