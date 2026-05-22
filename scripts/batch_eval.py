#!/usr/bin/env python3
"""
Batch evaluation script for PET checkpoints.

By default this scans outputs/<dataset_file> and evaluates one final checkpoint
per completed run directory. The checkpoint preference is:

  best_checkpoint.pth -> final_checkpoint.pth -> checkpoint.pth

Examples:
  python scripts/batch_eval.py --dataset_file SHA --data_path ./data/ShanghaiTech/part_A

  python scripts/batch_eval.py --dataset_file SHA --data_path ./data/ShanghaiTech/part_A --verbose

  python scripts/batch_eval.py --checkpoint_root /path/to/runs --dataset_file SHA --data_path ./data/ShanghaiTech/part_A

  python scripts/batch_eval.py --dataset_file SHA --data_path ./data/ShanghaiTech/part_A --dry_run
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_GET_SUPPORTED_TIMM_BACKBONES = None
_RESOLVE_TIMM_BACKBONE_NAME = lambda name: name
_BACKBONE_HELPERS_LOADED = False


DEFAULT_CHECKPOINT_NAMES = (
    'best_checkpoint.pth',
    'final_checkpoint.pth',
    'checkpoint.pth',
)

EVAL_METRICS_RE = re.compile(
    r'epoch:\s*(?P<epoch>-?\d+).*?mae:\s*(?P<mae>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*?mse:\s*(?P<mse>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
    re.IGNORECASE,
)


def get_backbone_from_dirname(dirname):
    """Infer backbone name from a run directory name."""
    global _BACKBONE_HELPERS_LOADED, _GET_SUPPORTED_TIMM_BACKBONES, _RESOLVE_TIMM_BACKBONE_NAME

    dirname_lower = dirname.lower()
    normalized_dirname = dirname_lower.replace('-', '_')

    if not _BACKBONE_HELPERS_LOADED:
        try:
            from models.backbones import get_supported_timm_backbones, resolve_timm_backbone_name

            _GET_SUPPORTED_TIMM_BACKBONES = get_supported_timm_backbones
            _RESOLVE_TIMM_BACKBONE_NAME = resolve_timm_backbone_name
        except ImportError:
            _GET_SUPPORTED_TIMM_BACKBONES = None
            _RESOLVE_TIMM_BACKBONE_NAME = lambda name: name
        _BACKBONE_HELPERS_LOADED = True

    if _GET_SUPPORTED_TIMM_BACKBONES is not None:
        known_backbones = sorted(
            set(_GET_SUPPORTED_TIMM_BACKBONES()),
            key=len,
            reverse=True,
        )
        for backbone in known_backbones:
            candidates = {
                backbone.lower(),
                _RESOLVE_TIMM_BACKBONE_NAME(backbone).lower(),
            }
            if any(candidate in normalized_dirname for candidate in candidates):
                return backbone

    if 'convnextv2' in dirname_lower:
        return 'convnextv2_base'
    if 'convnext' in dirname_lower:
        return 'convnext_base'
    if 'fastvit' in dirname_lower:
        return 'fastvit_tiny'
    if 'efficientvit' in dirname_lower:
        return 'efficientvit_tiny'
    if 'efficientnetv2' in dirname_lower:
        return 'efficientnetv2_tiny'
    if 'mobilenetv4' in dirname_lower:
        return 'mobilenetv4_small'
    if 'hgnetv2' in dirname_lower or 'hgnet' in dirname_lower:
        return 'hgnetv2_tiny'
    if 'pvt_v2' in dirname_lower or 'pvtv2' in dirname_lower:
        return 'pvtv2_b0'
    if 'edgenext' in dirname_lower:
        return 'edgenext_tiny'
    if 'repvit' in dirname_lower:
        return 'repvit_tiny'

    if 'swinv2' in dirname_lower:
        if 'swinv2_base_window8_256' in dirname_lower:
            return 'swinv2_base_window8_256'
        if 'swinv2_small' in dirname_lower:
            return 'swinv2_small_window8_256'
        return 'swinv2_base_window8_256'

    if 'maxvit' in dirname_lower:
        if 'poly' in dirname_lower:
            return 'maxvit_rmlp_tiny'
        if 'maxvit_rmlp_tiny_rw_256' in dirname_lower:
            return 'maxvit_rmlp_tiny'
        if 'maxvit_rmlp_small' in dirname_lower:
            return 'maxvit_small'
        if 'maxvit_small' in dirname_lower:
            return 'maxvit_small'
        if 'maxvit_tiny' in dirname_lower:
            return 'maxvit_tiny'
        return 'maxvit_tiny'

    if 'vgg' in dirname_lower:
        return 'vgg16_bn'

    return None


def should_skip_path(relative_parent, include_optuna):
    if include_optuna:
        return False
    parts = [p.lower() for p in relative_parent.parts]
    return any(p.startswith('opt_') for p in parts)


def result_key(relative_parent):
    key = relative_parent.as_posix().strip('/')
    return key.replace('/', '__').replace('\\', '__') or 'root'


def discover_checkpoints(scan_root, checkpoint_names, backbone_filter='', include_optuna=False):
    """Return one preferred checkpoint per run directory."""
    scan_root = Path(scan_root)
    selected = {}
    checkpoint_rank = {name: rank for rank, name in enumerate(checkpoint_names)}

    for checkpoint_name in checkpoint_names:
        for checkpoint_path in sorted(scan_root.rglob(checkpoint_name)):
            if not checkpoint_path.is_file():
                continue
            relative_parent = checkpoint_path.parent.relative_to(scan_root)
            candidate_name = relative_parent.as_posix()
            if should_skip_path(relative_parent, include_optuna):
                continue
            if backbone_filter and backbone_filter.lower() not in candidate_name.lower():
                continue

            current = selected.get(relative_parent)
            if current is not None:
                current_rank = checkpoint_rank[current.name]
                if checkpoint_rank[checkpoint_name] >= current_rank:
                    continue
            selected[relative_parent] = checkpoint_path

    checkpoints = []
    for relative_parent, checkpoint_path in sorted(selected.items(), key=lambda item: item[0].as_posix()):
        candidate_name = relative_parent.as_posix()
        backbone = get_backbone_from_dirname(candidate_name)
        if backbone:
            checkpoints.append({
                'dir': candidate_name,
                'key': result_key(relative_parent),
                'checkpoint': checkpoint_path,
                'checkpoint_name': checkpoint_path.name,
                'backbone': backbone,
            })
        else:
            print(f"Warning: Could not infer backbone for {candidate_name}")
    return checkpoints


def parse_eval_metrics(output_text):
    """Parse the final eval.py metrics line."""
    for match in reversed(list(EVAL_METRICS_RE.finditer(output_text))):
        return {
            'epoch': int(match.group('epoch')),
            'mae': float(match.group('mae')),
            'mse': float(match.group('mse')),
        }
    return None


def run_eval(item, args, log_path):
    """Run eval.py for a single checkpoint and return parsed metrics."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / 'eval.py'),
        '--backbone', item['backbone'],
        '--dataset_file', args.dataset_file,
        '--data_path', args.data_path,
        '--resume', str(item['checkpoint']),
        '--device', args.device,
        '--num_workers', str(args.num_workers),
    ]
    if args.override_score_threshold is not None:
        cmd.extend(['--override_score_threshold', str(args.override_score_threshold)])
    if args.override_split_threshold is not None:
        cmd.extend(['--override_split_threshold', str(args.override_split_threshold)])
    if args.override_split_threshold_quantile is not None:
        cmd.extend(['--override_split_threshold_quantile', str(args.override_split_threshold_quantile)])

    started_at = datetime.now().isoformat(timespec='seconds')
    output_lines = []
    header = [
        f"started_at: {started_at}",
        f"command: {' '.join(cmd)}",
        "",
    ]

    with log_path.open('w', encoding='utf-8', errors='replace') as log_file:
        for line in header:
            log_file.write(line + '\n')

        show_eval_output = args.verbose or (
            args.verbose_for and args.verbose_for.lower() in item['dir'].lower()
        )

        if show_eval_output:
            print(f"    Command: {' '.join(cmd)}")

        start_time = time.monotonic()
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )

        try:
            assert process.stdout is not None
            for line in process.stdout:
                output_lines.append(line)
                log_file.write(line)
                if show_eval_output:
                    print('    ' + line.rstrip())
                if args.timeout and time.monotonic() - start_time > args.timeout:
                    process.kill()
                    raise subprocess.TimeoutExpired(cmd, args.timeout)
            return_code = process.wait()
        except subprocess.TimeoutExpired:
            process.kill()
            message = f"\nERROR: evaluation timed out after {args.timeout} seconds\n"
            output_lines.append(message)
            log_file.write(message)
            return {
                'ok': False,
                'returncode': None,
                'metrics': None,
                'error': f'timeout after {args.timeout} seconds',
                'log_path': str(log_path),
            }

    output_text = ''.join(output_lines)
    metrics = parse_eval_metrics(output_text)
    if return_code != 0:
        return {
            'ok': False,
            'returncode': return_code,
            'metrics': metrics,
            'error': f'eval.py exited with return code {return_code}',
            'log_path': str(log_path),
        }
    if metrics is None:
        return {
            'ok': False,
            'returncode': return_code,
            'metrics': None,
            'error': 'could not parse eval metrics from eval.py output',
            'log_path': str(log_path),
        }
    return {
        'ok': True,
        'returncode': return_code,
        'metrics': metrics,
        'error': None,
        'log_path': str(log_path),
    }


def parse_args():
    parser = argparse.ArgumentParser('Batch evaluation for PET checkpoints')
    parser.add_argument('--dataset_file', default='SHA')
    parser.add_argument('--data_path', default='./data/ShanghaiTech/part_A')
    parser.add_argument('--output_dir', default='outputs', help='Root outputs directory')
    parser.add_argument(
        '--checkpoint_root',
        default='',
        help='Directory to scan for checkpoints. Defaults to <output_dir>/<dataset_file>.',
    )
    parser.add_argument(
        '--checkpoint_names',
        nargs='+',
        default=list(DEFAULT_CHECKPOINT_NAMES),
        help='Checkpoint filenames to consider, in preference order.',
    )
    parser.add_argument('--results_dir', default='', help='Where to write eval logs/results. Defaults under scan root.')
    parser.add_argument('--skip_existing', action='store_true', help='Skip runs with an existing eval_results.json')
    parser.add_argument('--force', action='store_true', help='Re-run even if eval_results.json already exists')
    parser.add_argument('--backbone_filter', default='', help='Only eval dirs containing this string')
    parser.add_argument('--include_optuna', action='store_true', help='Include checkpoints under opt_* trial directories')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--timeout', default=3600, type=int, help='Per-checkpoint timeout in seconds; 0 disables timeout')
    parser.add_argument('--verbose', action='store_true', help='Stream eval.py output while each checkpoint runs')
    parser.add_argument('--verbose_for', default='', help='Stream eval.py output only for dirs containing this string')
    parser.add_argument('--dry_run', action='store_true', help='Only list discovered checkpoints')
    parser.add_argument('--override_score_threshold', default=None, type=float)
    parser.add_argument('--override_split_threshold', default=None, type=float)
    parser.add_argument('--override_split_threshold_quantile', default=None, type=float)
    return parser.parse_args()


def main():
    args = parse_args()

    scan_root = Path(args.checkpoint_root) if args.checkpoint_root else Path(args.output_dir) / args.dataset_file
    if not scan_root.exists():
        print(f"Checkpoint directory not found: {scan_root}")
        return 1

    results_dir = Path(args.results_dir) if args.results_dir else scan_root / 'results'
    checkpoints_to_eval = discover_checkpoints(
        scan_root=scan_root,
        checkpoint_names=args.checkpoint_names,
        backbone_filter=args.backbone_filter,
        include_optuna=args.include_optuna,
    )

    if not checkpoints_to_eval:
        print(f"No checkpoints found under {scan_root}")
        return 0

    print(f"Scanning: {scan_root}")
    print(f"Results:  {results_dir}")
    print(f"Found {len(checkpoints_to_eval)} run checkpoint(s)")

    for item in checkpoints_to_eval:
        print(f"  - {item['dir']} | {item['backbone']} | {item['checkpoint_name']}")

    if args.dry_run:
        print("\nDry run only; no evaluations were started.")
        return 0

    results = defaultdict(list)
    failures = []
    results_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(checkpoints_to_eval, start=1):
        run_results_dir = results_dir / item['key']
        run_results_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_results_dir / 'eval_results.json'
        log_path = run_results_dir / 'eval_log.txt'

        if args.skip_existing and not args.force and result_path.exists():
            print(f"\n[{i}/{len(checkpoints_to_eval)}] Skipping {item['dir']} (existing result)")
            try:
                saved = json.loads(result_path.read_text(encoding='utf-8'))
                if saved.get('ok') and saved.get('eval_mae') is not None:
                    results[item['backbone']].append({
                        'dir': item['dir'],
                        'mae': float(saved['eval_mae']),
                        'mse': float(saved['eval_mse']) if saved.get('eval_mse') is not None else None,
                        'checkpoint': saved.get('checkpoint', str(item['checkpoint'])),
                    })
            except Exception as exc:
                print(f"  Warning: could not read existing result: {exc}")
            continue

        print(f"\n[{i}/{len(checkpoints_to_eval)}] Evaluating {item['dir']} ({item['backbone']})")
        print(f"  Checkpoint: {item['checkpoint']}")
        eval_result = run_eval(item, args, log_path)

        metrics = eval_result['metrics'] or {}
        result_doc = {
            'ok': eval_result['ok'],
            'backbone': item['backbone'],
            'dir': item['dir'],
            'checkpoint': str(item['checkpoint']),
            'checkpoint_name': item['checkpoint_name'],
            'eval_epoch': metrics.get('epoch'),
            'eval_mae': metrics.get('mae'),
            'eval_mse': metrics.get('mse'),
            'returncode': eval_result['returncode'],
            'error': eval_result['error'],
            'log_path': eval_result['log_path'],
            'evaluated_at': datetime.now().isoformat(timespec='seconds'),
        }
        result_path.write_text(json.dumps(result_doc, indent=2) + '\n', encoding='utf-8')

        if eval_result['ok']:
            mae = float(metrics['mae'])
            mse = float(metrics['mse'])
            results[item['backbone']].append({
                'dir': item['dir'],
                'mae': mae,
                'mse': mse,
                'checkpoint': str(item['checkpoint']),
            })
            print(f"  Eval MAE: {mae:.4f} | Eval MSE: {mse:.4f}")
        else:
            failures.append(result_doc)
            print(f"  Failed: {eval_result['error']}")
            print(f"  Log: {log_path}")

    summary = {}
    for backbone in sorted(results.keys()):
        maes = [r['mae'] for r in results[backbone]]
        mses = [r['mse'] for r in results[backbone] if r['mse'] is not None]
        summary[backbone] = {
            'count': len(maes),
            'mean_mae': float(np.mean(maes)),
            'std_mae': float(np.std(maes)),
            'min_mae': float(np.min(maes)),
            'max_mae': float(np.max(maes)),
            'mean_mse': float(np.mean(mses)) if mses else None,
            'results': results[backbone],
        }

    summary_doc = {
        'scan_root': str(scan_root),
        'results_dir': str(results_dir),
        'dataset_file': args.dataset_file,
        'data_path': args.data_path,
        'checkpoint_names': args.checkpoint_names,
        'success_count': sum(s['count'] for s in summary.values()),
        'failure_count': len(failures),
        'summary': summary,
        'failures': failures,
    }
    summary_path = results_dir / 'EVAL_SUMMARY.json'
    summary_path.write_text(json.dumps(summary_doc, indent=2) + '\n', encoding='utf-8')

    print(f"\nSummary saved to {summary_path}")
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    if not summary:
        print("No successful evaluations.")
    for backbone in sorted(summary.keys()):
        s = summary[backbone]
        mean_mse = 'N/A' if s['mean_mse'] is None else f"{s['mean_mse']:.4f}"
        print(f"\n{backbone}:")
        print(f"  Count: {s['count']}")
        print(f"  Mean MAE: {s['mean_mae']:.4f} +/- {s['std_mae']:.4f}")
        print(f"  Mean MSE: {mean_mse}")
        print(f"  MAE range: [{s['min_mae']:.4f}, {s['max_mae']:.4f}]")
        for row in s['results']:
            mse = 'N/A' if row['mse'] is None else f"{row['mse']:.4f}"
            print(f"    - {row['dir']}: MAE {row['mae']:.4f}, MSE {mse}")

    if failures:
        print(f"\nFailures: {len(failures)}")
        for failure in failures:
            print(f"  - {failure['dir']}: {failure['error']} (log: {failure['log_path']})")

    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
