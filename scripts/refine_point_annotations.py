import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Use original crowd annotations as PET decoder queries and write '
            'refined MAT annotations for a new scratch training run'
        ),
    )
    parser.add_argument('--resume', required=True)
    parser.add_argument('--dataset_file', choices=('SHA', 'SHB'), required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--min_person_score', default=0.0, type=float)
    parser.add_argument(
        '--low_score_policy',
        default='keep_original',
        choices=('keep_original', 'keep_refined'),
        help='handling for annotations classified below --min_person_score',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / 'refinement_predictions.json'
    command = [
        sys.executable,
        REPO_ROOT / 'eval.py',
        '--resume', args.resume,
        '--dataset_file', args.dataset_file,
        '--data_path', args.data_path,
        '--eval_image_set', 'train_eval',
        '--device', args.device,
        '--num_workers', args.num_workers,
        '--refinement_predictions_file', predictions_path,
    ]
    print('+', ' '.join(str(part) for part in command), flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=REPO_ROOT,
        check=True,
    )

    with predictions_path.open('r', encoding='utf-8') as handle:
        rows = json.load(handle)
    manifest_rows = []
    for row in rows:
        original = np.asarray(
            row['original_points_yx'],
            dtype=np.float32,
        ).reshape(-1, 2)
        refined = np.asarray(
            row['refined_points_yx'],
            dtype=np.float32,
        ).reshape(-1, 2)
        scores = np.asarray(
            row['person_scores'],
            dtype=np.float32,
        ).reshape(-1)
        if not (
            original.shape == refined.shape
            and scores.shape[0] == original.shape[0]
        ):
            raise ValueError(f'Invalid refinement lengths for {row["image_path"]}')
        accepted = scores >= float(args.min_person_score)
        final = refined.copy()
        if args.low_score_policy == 'keep_original':
            final[~accepted] = original[~accepted]
        stem = Path(row['image_path']).stem
        savemat(
            output_dir / f'GT_{stem}.mat',
            {
                'annPoints': final[:, ::-1],
                'originalPoints': original[:, ::-1],
                'refinedPoints': refined[:, ::-1],
                'pointConfidence': scores.reshape(-1, 1),
                'refinementAccepted': accepted.astype(np.uint8).reshape(-1, 1),
            },
        )
        manifest_rows.append({
            'image': Path(row['image_path']).name,
            'points': int(final.shape[0]),
            'accepted': int(accepted.sum()),
            'mean_displacement': float(row['mean_displacement']),
            'max_displacement': float(row['max_displacement']),
        })
    (output_dir / 'refinement_manifest.json').write_text(
        json.dumps(
            {
                'dataset': args.dataset_file,
                'checkpoint': args.resume,
                'min_person_score': args.min_person_score,
                'low_score_policy': args.low_score_policy,
                'samples': manifest_rows,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(f'Wrote {len(rows)} refined annotation files to {output_dir}')


if __name__ == '__main__':
    main()
