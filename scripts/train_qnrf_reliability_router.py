#!/usr/bin/env python3
"""Train a train-only PET/ZIP reliability router and freeze test predictions.

The router learns which of two frozen counting experts has lower absolute
error. Model selection uses cross-fitted QNRF training predictions only. Test
ground truth is deliberately ignored even when it is present in eval JSON.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_NAMES = (
    'log1p_pet',
    'log1p_zip',
    'signed_normalized_difference',
    'absolute_normalized_difference',
    'log_count_ratio',
)


class ReliabilityRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(1)


def row_key(row):
    image_id = str(row.get('image_id') or '')
    image_path = str(row.get('image_path') or '')
    key = image_id or image_path
    if not key:
        raise ValueError('per-image row has neither image_id nor image_path')
    return key


def load_predictions(path, require_gt):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path} must contain a non-empty JSON list')
    keyed = {}
    for row in rows:
        key = row_key(row)
        if key in keyed:
            raise ValueError(f'{path} contains duplicate key {key!r}')
        pred = float(row['pred_cnt'])
        if not math.isfinite(pred) or pred < 0:
            raise ValueError(f'{path} has invalid prediction for {key}: {pred}')
        item = {
            'image_id': str(row.get('image_id') or ''),
            'image_path': str(row.get('image_path') or ''),
            'pred': pred,
        }
        if require_gt:
            gt = float(row['gt_cnt'])
            if not math.isfinite(gt) or gt < 0:
                raise ValueError(f'{path} has invalid GT for {key}: {gt}')
            item['gt'] = gt
        keyed[key] = item
    return keyed


def align_experts(pet, zip_counts, label, require_gt):
    if set(pet) != set(zip_counts):
        missing_pet = sorted(set(zip_counts) - set(pet))[:5]
        missing_zip = sorted(set(pet) - set(zip_counts))[:5]
        raise ValueError(
            f'{label} expert keys differ; missing_pet={missing_pet} '
            f'missing_zip={missing_zip}'
        )
    keys = sorted(pet)
    if require_gt:
        for key in keys:
            if pet[key]['gt'] != zip_counts[key]['gt']:
                raise ValueError(f'{label} GT mismatch for {key}')
    return keys


def build_features(pet_count, zip_count):
    pet_count = np.asarray(pet_count, dtype=np.float32)
    zip_count = np.asarray(zip_count, dtype=np.float32)
    log_pet = np.log1p(pet_count)
    log_zip = np.log1p(zip_count)
    scale = np.sqrt(pet_count + zip_count + 1.0)
    signed_difference = (zip_count - pet_count) / scale
    return np.stack(
        (
            log_pet,
            log_zip,
            signed_difference,
            np.abs(signed_difference),
            log_zip - log_pet,
        ),
        axis=1,
    ).astype(np.float32)


def metrics(prediction, target):
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        'mae': float(np.abs(error).mean()),
        'mse': float(np.sqrt(np.square(error).mean())),
        'bias': float(error.mean()),
    }


def make_stratified_folds(target, folds, seed):
    target = np.asarray(target, dtype=np.float64)
    order = np.argsort(target, kind='stable')
    # Shuffle within count-local groups before round-robin assignment. This
    # retains tail coverage in every fold without deriving any test statistic.
    rng = np.random.default_rng(seed)
    group_size = max(folds * 4, folds)
    ordered_groups = []
    for start in range(0, len(order), group_size):
        group = order[start:start + group_size].copy()
        rng.shuffle(group)
        ordered_groups.extend(group.tolist())
    fold_ids = np.empty(len(target), dtype=np.int64)
    for index, sample_index in enumerate(ordered_groups):
        fold_ids[sample_index] = index % folds
    return fold_ids


def train_fold(
    train_features,
    train_labels,
    train_weights,
    test_features,
    hidden_dim,
    epochs,
    learning_rate,
    weight_decay,
    seed,
):
    torch.manual_seed(seed)
    model = ReliabilityRouter(train_features.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    x = torch.from_numpy(train_features)
    y = torch.from_numpy(train_labels.astype(np.float32))
    weights = torch.from_numpy(train_weights.astype(np.float32))
    for _ in range(epochs):
        logits = model(x)
        loss = (
            F.binary_cross_entropy_with_logits(logits, y, reduction='none') * weights
        ).sum() / weights.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(test_features))).cpu().numpy()
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    return probabilities.astype(np.float64), state


def choose_threshold(oof_probability, pet, zip_counts, gt):
    candidates = []
    # threshold > 1 is the explicit PET-only fallback.
    for threshold in [*np.linspace(0.05, 0.95, 19).tolist(), 1.01]:
        use_zip = oof_probability >= threshold
        prediction = np.where(use_zip, zip_counts, pet)
        row = {
            'threshold': float(threshold),
            'zip_images': int(use_zip.sum()),
            **metrics(prediction, gt),
        }
        candidates.append(row)
    candidates.sort(key=lambda row: (row['mae'], row['mse'], row['zip_images']))
    return candidates[0], candidates


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Cross-fit a QNRF PET/ZIP reliability router on train predictions '
            'and apply the frozen ensemble to test predictions'
        ),
    )
    parser.add_argument('--train_pet', required=True)
    parser.add_argument('--train_zip', required=True)
    parser.add_argument('--test_pet', required=True)
    parser.add_argument('--test_zip', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--model_output', required=True)
    parser.add_argument('--folds', default=5, type=int)
    parser.add_argument('--hidden_dim', default=8, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--seed', default=7, type=int)
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError('--folds must be at least 2')
    if args.hidden_dim < 1 or args.epochs < 1:
        raise ValueError('--hidden_dim and --epochs must be positive')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_pet_rows = load_predictions(args.train_pet, require_gt=True)
    train_zip_rows = load_predictions(args.train_zip, require_gt=True)
    train_keys = align_experts(
        train_pet_rows,
        train_zip_rows,
        'train',
        require_gt=True,
    )
    if len(train_keys) < args.folds:
        raise ValueError('number of train samples must be at least --folds')

    test_pet_rows = load_predictions(args.test_pet, require_gt=False)
    test_zip_rows = load_predictions(args.test_zip, require_gt=False)
    test_keys = align_experts(
        test_pet_rows,
        test_zip_rows,
        'test',
        require_gt=False,
    )

    train_pet = np.asarray([train_pet_rows[key]['pred'] for key in train_keys], dtype=np.float64)
    train_zip = np.asarray([train_zip_rows[key]['pred'] for key in train_keys], dtype=np.float64)
    train_gt = np.asarray([train_pet_rows[key]['gt'] for key in train_keys], dtype=np.float64)
    test_pet = np.asarray([test_pet_rows[key]['pred'] for key in test_keys], dtype=np.float64)
    test_zip = np.asarray([test_zip_rows[key]['pred'] for key in test_keys], dtype=np.float64)

    features = build_features(train_pet, train_zip)
    test_features_raw = build_features(test_pet, test_zip)
    pet_error = np.abs(train_pet - train_gt)
    zip_error = np.abs(train_zip - train_gt)
    labels = (zip_error < pet_error).astype(np.float32)
    regret = np.abs(pet_error - zip_error)
    positive_regret = regret[regret > 0]
    regret_scale = float(np.median(positive_regret)) if positive_regret.size else 1.0
    weights = np.clip(regret / max(regret_scale, 1e-6), 0.1, 10.0).astype(np.float32)

    fold_ids = make_stratified_folds(train_gt, args.folds, args.seed)
    oof_probability = np.zeros(len(train_keys), dtype=np.float64)
    test_probability_sum = np.zeros(len(test_keys), dtype=np.float64)
    model_states = []
    scaler_rows = []
    for fold in range(args.folds):
        validation_mask = fold_ids == fold
        training_mask = ~validation_mask
        mean = features[training_mask].mean(axis=0)
        std = features[training_mask].std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        x_train = ((features[training_mask] - mean) / std).astype(np.float32)
        x_validation = ((features[validation_mask] - mean) / std).astype(np.float32)
        x_test = ((test_features_raw - mean) / std).astype(np.float32)

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
        model_states.append(state)
        scaler_rows.append({
            'mean': mean.astype(np.float32),
            'std': std.astype(np.float32),
        })

    test_probability = test_probability_sum / float(args.folds)
    best, threshold_candidates = choose_threshold(
        oof_probability,
        train_pet,
        train_zip,
        train_gt,
    )
    threshold = best['threshold']
    test_use_zip = test_probability >= threshold
    test_prediction = np.where(test_use_zip, test_zip, test_pet)

    output_rows = []
    for index, key in enumerate(test_keys):
        source_row = test_pet_rows[key]
        output_rows.append({
            'image_id': source_row['image_id'],
            'image_path': source_row['image_path'],
            'pet_pred_cnt': float(test_pet[index]),
            'zip_pred_cnt': float(test_zip[index]),
            'zip_probability': float(test_probability[index]),
            'selected_source': 'zip' if test_use_zip[index] else 'pet',
            'pred_cnt': float(test_prediction[index]),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_rows, indent=2) + '\n', encoding='utf-8')

    artifact = {
        'format_version': 1,
        'feature_names': FEATURE_NAMES,
        'hidden_dim': args.hidden_dim,
        'folds': args.folds,
        'threshold': threshold,
        'states': model_states,
        'scalers': scaler_rows,
    }
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, model_path)

    oracle_prediction = np.where(zip_error < pet_error, train_zip, train_pet)
    report = {
        'selection_data': 'QNRF train_eval only',
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
        'train_samples': len(train_keys),
        'test_samples': len(test_keys),
        'train_pet': metrics(train_pet, train_gt),
        'train_zip': metrics(train_zip, train_gt),
        'train_oracle': metrics(oracle_prediction, train_gt),
        'oof_router': best,
        'threshold_candidates': threshold_candidates,
        'test_selection': {
            'pet_images': int((~test_use_zip).sum()),
            'zip_images': int(test_use_zip.sum()),
        },
        'inputs': {
            'train_pet': args.train_pet,
            'train_zip': args.train_zip,
            'test_pet': args.test_pet,
            'test_zip': args.test_zip,
        },
        'outputs': {
            'predictions': str(output_path),
            'model': str(model_path),
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    print('OOF PET:', json.dumps(report['train_pet'], indent=2))
    print('OOF ZIP:', json.dumps(report['train_zip'], indent=2))
    print('OOF oracle:', json.dumps(report['train_oracle'], indent=2))
    print('selected OOF router:', json.dumps(best, indent=2))
    print('test selection:', json.dumps(report['test_selection'], indent=2))
    print(f'wrote predictions: {output_path}')
    print(f'wrote router model: {model_path}')
    print(f'wrote report: {report_path}')


if __name__ == '__main__':
    main()
