import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.UCFCC50 import build_folds, find_images_dir  # noqa: E402
from datasets.SHA import IMAGE_EXTENSIONS  # noqa: E402

RESERVED_FORWARD_FLAGS = {
    '--dataset_file',
    '--data_path',
    '--model_recipe',
    '--ucfcc50_fold',
    '--ucfcc50_fold_seed',
    '--ucfcc50_fold_manifest',
    '--validation_protocol',
    '--output_dir',
    '--resume',
    '--resume_model_only',
    '--resume_allow_arch_change',
}


def run(command):
    print('+', ' '.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=REPO_ROOT, check=True)


def result_metrics(path):
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    mae = payload.get(
        'test_mae',
        payload.get('eval_mae', payload.get('mae')),
    )
    mse = payload.get(
        'test_mse',
        payload.get('eval_mse', payload.get('mse')),
    )
    if mae is None or mse is None:
        raise ValueError(f'Evaluation result has no MAE/MSE: {path}')
    return float(mae), float(mse)


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Leakage-safe UCF-CC-50 five-fold training and evaluation. '
            'Unknown arguments are forwarded to main.py.'
        ),
    )
    parser.add_argument('--data_path', default='data/UCF_CC_50')
    parser.add_argument('--output_root', default='outputs/UCFCC50')
    parser.add_argument('--results_dir', default='eval_results/UCFCC50/five_fold')
    parser.add_argument('--model_recipe', default='vgg_pet_apg_rifi')
    parser.add_argument('--fold_seed', default=42, type=int)
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--resume_existing', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    args, forwarded = parser.parse_known_args()

    if any(fold not in range(5) for fold in args.folds):
        raise ValueError('--folds values must be in [0, 4]')
    forwarded_flags = {
        token.split('=', 1)[0]
        for token in forwarded
        if token.startswith('--')
    }
    forbidden = sorted(forwarded_flags & RESERVED_FORWARD_FLAGS)
    if forbidden:
        raise ValueError(
            'UCF-CC-50 fold isolation owns these arguments and will not '
            f'forward them: {", ".join(forbidden)}'
        )

    images_dir = find_images_dir(args.data_path)
    stems = [
        path.stem
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    folds = build_folds(stems, seed=args.fold_seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / 'fold_manifest.json'
    manifest_path.write_text(
        json.dumps(
            {
                'dataset': 'UCF-CC-50',
                'seed': args.fold_seed,
                'folds': folds,
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    fold_results = []
    for fold in args.folds:
        run_name = f'{args.model_recipe}_seed{args.fold_seed}_fold{fold}'
        output_dir = Path(args.output_root) / run_name
        checkpoint = output_dir / 'best_checkpoint.pth'
        if not args.eval_only:
            train_command = [
                sys.executable,
                REPO_ROOT / 'main.py',
                *forwarded,
                '--dataset_file', 'UCFCC50',
                '--data_path', args.data_path,
                '--model_recipe', args.model_recipe,
                '--allow_experimental_model_recipe',
                '--ucfcc50_fold', fold,
                '--ucfcc50_fold_seed', args.fold_seed,
                '--ucfcc50_fold_manifest', manifest_path,
                '--validation_protocol', 'train_holdout',
                '--output_dir', output_dir,
                '--device', args.device,
                '--num_workers', args.num_workers,
            ]
            latest = output_dir / 'checkpoint.pth'
            if args.resume_existing and latest.is_file():
                train_command.extend(['--resume', latest])
            run(train_command)

        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing fold checkpoint: {checkpoint}')
        result_path = results_dir / f'fold_{fold}.json'
        run([
            sys.executable,
            REPO_ROOT / 'eval.py',
            '--dataset_file', 'UCFCC50',
            '--data_path', args.data_path,
            '--resume', checkpoint,
            '--ucfcc50_fold', fold,
            '--ucfcc50_fold_seed', args.fold_seed,
            '--ucfcc50_fold_manifest', manifest_path,
            '--eval_image_set', 'val',
            '--results_file', result_path,
            '--device', args.device,
            '--num_workers', args.num_workers,
        ])
        mae, mse = result_metrics(result_path)
        fold_results.append({'fold': fold, 'mae': mae, 'mse': mse})

    mean_mae = sum(item['mae'] for item in fold_results) / len(fold_results)
    aggregate_mse = math.sqrt(
        sum(item['mse'] ** 2 for item in fold_results) / len(fold_results)
    )
    summary = {
        'dataset': 'UCF-CC-50',
        'protocol': 'five_fold_cross_validation',
        'fold_seed': args.fold_seed,
        'checkpoint_selection': 'train_holdout',
        'fold_results': fold_results,
        'mae': mean_mae,
        'mse': aggregate_mse,
    }
    (results_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
