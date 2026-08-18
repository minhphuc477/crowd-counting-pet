#!/usr/bin/env python3
"""Cross-fit a frozen two-expert count router without reading test labels."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.train_qnrf_reliability_router import (
        FEATURE_NAMES,
        align_experts,
        build_features,
        choose_threshold,
        load_predictions,
        make_stratified_folds,
        metrics,
        train_fold,
    )
except ModuleNotFoundError:
    from train_qnrf_reliability_router import (
        FEATURE_NAMES,
        align_experts,
        build_features,
        choose_threshold,
        load_predictions,
        make_stratified_folds,
        metrics,
        train_fold,
    )


def read_submission_ids(path):
    ids = []
    for line in Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
        fields = line.split()
        if fields:
            ids.append(Path(fields[0]).stem)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('submission manifest must contain unique image IDs')
    return ids


def write_count_submission(path, rows, manifest):
    by_id = {str(row['image_id']): row for row in rows}
    ids = read_submission_ids(manifest)
    missing = [image_id for image_id in ids if image_id not in by_id]
    extra = sorted(set(by_id) - set(ids))
    if missing or extra:
        raise ValueError(
            f'router predictions do not match submission manifest: '
            f'missing={missing[:5]} extra={extra[:5]}'
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        '\n'.join(
            f"{image_id} {float(by_id[image_id]['pred_cnt']):.6f}"
            for image_id in ids
        ) + '\n',
        encoding='utf-8',
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Cross-fit a reliability router on an approved selection split, '
            'then apply it to frozen test predictions without reading test GT'
        ),
    )
    parser.add_argument('--selection_reference', required=True)
    parser.add_argument('--selection_candidate', required=True)
    parser.add_argument('--test_reference', required=True)
    parser.add_argument('--test_candidate', required=True)
    parser.add_argument('--reference_name', default='pet')
    parser.add_argument('--candidate_name', default='measure')
    parser.add_argument(
        '--selection_protocol',
        required=True,
        choices=('official_val', 'oof_train'),
    )
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--model_output', required=True)
    parser.add_argument('--submission_output', default='')
    parser.add_argument('--submission_manifest', default='')
    parser.add_argument('--folds', default=5, type=int)
    parser.add_argument('--hidden_dim', default=8, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError('--folds must be at least 2')
    if args.hidden_dim < 1 or args.epochs < 1:
        raise ValueError('--hidden_dim and --epochs must be positive')
    if bool(args.submission_output) != bool(args.submission_manifest):
        raise ValueError(
            '--submission_output and --submission_manifest must be supplied together'
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    selection_reference_rows = load_predictions(
        args.selection_reference,
        require_gt=True,
    )
    selection_candidate_rows = load_predictions(
        args.selection_candidate,
        require_gt=True,
    )
    selection_keys = align_experts(
        selection_reference_rows,
        selection_candidate_rows,
        'selection',
        require_gt=True,
    )
    if len(selection_keys) < args.folds:
        raise ValueError('selection samples must be at least --folds')

    # Test ground truth is intentionally discarded even if input rows contain it.
    test_reference_rows = load_predictions(args.test_reference, require_gt=False)
    test_candidate_rows = load_predictions(args.test_candidate, require_gt=False)
    test_keys = align_experts(
        test_reference_rows,
        test_candidate_rows,
        'test',
        require_gt=False,
    )

    selection_reference = np.asarray(
        [selection_reference_rows[key]['pred'] for key in selection_keys],
        dtype=np.float64,
    )
    selection_candidate = np.asarray(
        [selection_candidate_rows[key]['pred'] for key in selection_keys],
        dtype=np.float64,
    )
    selection_gt = np.asarray(
        [selection_reference_rows[key]['gt'] for key in selection_keys],
        dtype=np.float64,
    )
    test_reference = np.asarray(
        [test_reference_rows[key]['pred'] for key in test_keys],
        dtype=np.float64,
    )
    test_candidate = np.asarray(
        [test_candidate_rows[key]['pred'] for key in test_keys],
        dtype=np.float64,
    )

    selection_features = build_features(selection_reference, selection_candidate)
    test_features = build_features(test_reference, test_candidate)
    reference_error = np.abs(selection_reference - selection_gt)
    candidate_error = np.abs(selection_candidate - selection_gt)
    labels = (candidate_error < reference_error).astype(np.float32)
    regret = np.abs(reference_error - candidate_error)
    positive_regret = regret[regret > 0]
    regret_scale = float(np.median(positive_regret)) if positive_regret.size else 1.0
    weights = np.clip(regret / max(regret_scale, 1e-6), 0.1, 10.0).astype(np.float32)

    fold_ids = make_stratified_folds(selection_gt, args.folds, args.seed)
    oof_probability = np.zeros(len(selection_keys), dtype=np.float64)
    test_probability_sum = np.zeros(len(test_keys), dtype=np.float64)
    states = []
    scalers = []
    for fold in range(args.folds):
        validation_mask = fold_ids == fold
        training_mask = ~validation_mask
        mean = selection_features[training_mask].mean(axis=0)
        std = selection_features[training_mask].std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        x_train = ((selection_features[training_mask] - mean) / std).astype(np.float32)
        x_validation = ((selection_features[validation_mask] - mean) / std).astype(np.float32)
        x_test = ((test_features - mean) / std).astype(np.float32)

        validation_probability, state = train_fold(
            x_train,
            labels[training_mask],
            weights[training_mask],
            x_validation,
            args.hidden_dim,
            args.epochs,
            args.learning_rate,
            args.weight_decay,
            args.seed + fold,
        )
        test_probability, _ = train_fold(
            x_train,
            labels[training_mask],
            weights[training_mask],
            x_test,
            args.hidden_dim,
            args.epochs,
            args.learning_rate,
            args.weight_decay,
            args.seed + fold,
        )
        oof_probability[validation_mask] = validation_probability
        test_probability_sum += test_probability
        states.append(state)
        scalers.append({
            'mean': mean.astype(np.float32),
            'std': std.astype(np.float32),
        })

    test_probability = test_probability_sum / float(args.folds)
    best, threshold_candidates = choose_threshold(
        oof_probability,
        selection_reference,
        selection_candidate,
        selection_gt,
    )
    threshold = best['threshold']
    test_use_candidate = test_probability >= threshold
    test_prediction = np.where(
        test_use_candidate,
        test_candidate,
        test_reference,
    )

    output_rows = []
    for index, key in enumerate(test_keys):
        source = test_reference_rows[key]
        output_rows.append({
            'image_id': source['image_id'],
            'image_path': source['image_path'],
            'reference_pred_cnt': float(test_reference[index]),
            'candidate_pred_cnt': float(test_candidate[index]),
            'candidate_probability': float(test_probability[index]),
            'selected_source': (
                args.candidate_name if test_use_candidate[index] else args.reference_name
            ),
            'pred_cnt': float(test_prediction[index]),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_rows, indent=2) + '\n', encoding='utf-8')

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'format_version': 1,
        'feature_names': FEATURE_NAMES,
        'hidden_dim': args.hidden_dim,
        'folds': args.folds,
        'threshold': threshold,
        'states': states,
        'scalers': scalers,
        'reference_name': args.reference_name,
        'candidate_name': args.candidate_name,
    }, model_path)

    oracle = np.where(
        candidate_error < reference_error,
        selection_candidate,
        selection_reference,
    )
    report = {
        'selection_protocol': args.selection_protocol,
        'test_ground_truth_used': False,
        'features': FEATURE_NAMES,
        'configuration': {
            'folds': args.folds,
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'seed': args.seed,
        },
        'selection_samples': len(selection_keys),
        'test_samples': len(test_keys),
        'selection_reference': metrics(selection_reference, selection_gt),
        'selection_candidate': metrics(selection_candidate, selection_gt),
        'selection_oracle': metrics(oracle, selection_gt),
        'oof_router': best,
        'threshold_candidates': threshold_candidates,
        'test_selection': {
            args.reference_name: int((~test_use_candidate).sum()),
            args.candidate_name: int(test_use_candidate.sum()),
        },
        'inputs': {
            'selection_reference': args.selection_reference,
            'selection_candidate': args.selection_candidate,
            'test_reference': args.test_reference,
            'test_candidate': args.test_candidate,
        },
        'outputs': {
            'predictions': str(output_path),
            'model': str(model_path),
            'submission': args.submission_output or None,
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    if args.submission_output:
        write_count_submission(
            args.submission_output,
            output_rows,
            args.submission_manifest,
        )

    print('OOF reference:', json.dumps(report['selection_reference'], indent=2))
    print('OOF candidate:', json.dumps(report['selection_candidate'], indent=2))
    print('OOF oracle:', json.dumps(report['selection_oracle'], indent=2))
    print('selected OOF router:', json.dumps(best, indent=2))
    print('test selection:', json.dumps(report['test_selection'], indent=2))
    print(f'wrote predictions: {output_path}')
    print(f'wrote router model: {model_path}')
    print(f'wrote report: {report_path}')
    if args.submission_output:
        print(f'wrote count submission: {args.submission_output}')


if __name__ == '__main__':
    main()
