import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_test_ids(data_root):
    path = Path(data_root) / 'test.txt'
    if not path.is_file():
        raise FileNotFoundError(f'Missing official NWPU test manifest: {path}')
    ids = []
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        fields = line.split()
        if fields:
            ids.append(Path(fields[0]).stem)
    if len(ids) != 1500 or len(set(ids)) != 1500:
        raise ValueError(
            f'NWPU test manifest must contain 1,500 unique IDs, found {len(ids)}'
        )
    return ids


def main():
    parser = argparse.ArgumentParser(
        description='Export an official 1,500-line NWPU-Crowd count submission',
    )
    parser.add_argument('--resume', required=True)
    parser.add_argument('--data_path', default='data/NWPU-Crowd')
    parser.add_argument('--output', required=True)
    parser.add_argument('--predictions_json', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--eval_max_size', default=-1, type=int)
    parser.add_argument('--override_score_threshold', default=None, type=float)
    args, forwarded = parser.parse_known_args()

    test_ids = read_test_ids(args.data_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        Path(args.predictions_json)
        if args.predictions_json
        else output.with_suffix('.predictions.json')
    )
    metrics_path = output.with_suffix('.diagnostic.json')
    command = [
        sys.executable,
        REPO_ROOT / 'eval.py',
        '--resume', args.resume,
        '--dataset_file', 'NWPU',
        '--data_path', args.data_path,
        '--nwpu_eval_split', 'test',
        '--eval_image_set', 'val',
        '--device', args.device,
        '--num_workers', args.num_workers,
        '--eval_max_size', args.eval_max_size,
        '--no_localization_metrics',
        '--per_image_predictions_file', predictions_path,
        '--results_file', metrics_path,
        *forwarded,
    ]
    if args.override_score_threshold is not None:
        command.extend([
            '--override_score_threshold',
            args.override_score_threshold,
        ])
    print('+', ' '.join(str(part) for part in command), flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=REPO_ROOT,
        check=True,
    )

    with predictions_path.open('r', encoding='utf-8') as handle:
        rows = json.load(handle)
    by_id = {str(row['image_id']): row for row in rows}
    missing = [image_id for image_id in test_ids if image_id not in by_id]
    extra = sorted(set(by_id) - set(test_ids))
    if missing or extra:
        raise ValueError(
            f'NWPU prediction IDs do not match test.txt: '
            f'missing={missing[:5]} extra={extra[:5]}'
        )
    lines = [
        f"{image_id} {float(by_id[image_id]['pred_cnt']):.6f}"
        for image_id in test_ids
    ]
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {len(lines)} NWPU test predictions to {output}')


if __name__ == '__main__':
    main()
