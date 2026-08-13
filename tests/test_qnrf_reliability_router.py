import json
import subprocess
import sys
from pathlib import Path


def write_rows(path, predictions, targets=None, start=1):
    rows = []
    for offset, prediction in enumerate(predictions):
        row = {
            'image_id': str(start + offset),
            'image_path': f'img_{start + offset:04d}.jpg',
            'pred_cnt': prediction,
        }
        if targets is not None:
            row['gt_cnt'] = targets[offset]
        rows.append(row)
    path.write_text(json.dumps(rows), encoding='utf-8')


def test_router_uses_train_gt_but_ignores_test_gt(tmp_path):
    train_gt = [10, 20, 30, 40, 100, 200, 500, 1000] * 2
    train_pet_pred = [value + (15 if index % 2 == 0 else 0) for index, value in enumerate(train_gt)]
    train_zip_pred = [value if index % 2 == 0 else value + 15 for index, value in enumerate(train_gt)]
    test_pet_pred = [10, 100, 1000, 2000]
    test_zip_pred = [12, 90, 900, 2500]

    train_pet = tmp_path / 'train_pet.json'
    train_zip = tmp_path / 'train_zip.json'
    test_pet = tmp_path / 'test_pet.json'
    test_zip = tmp_path / 'test_zip.json'
    write_rows(train_pet, train_pet_pred, train_gt)
    write_rows(train_zip, train_zip_pred, train_gt)
    # Conflicting GT fields are intentionally present. Test loaders must ignore
    # them completely rather than use them for threshold/model selection.
    write_rows(test_pet, test_pet_pred, [9999] * 4, start=100)
    write_rows(test_zip, test_zip_pred, [0] * 4, start=100)

    output = tmp_path / 'predictions.json'
    report = tmp_path / 'report.json'
    model = tmp_path / 'router.pth'
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'train_qnrf_reliability_router.py'
    subprocess.run([
        sys.executable,
        str(script),
        '--train_pet', str(train_pet),
        '--train_zip', str(train_zip),
        '--test_pet', str(test_pet),
        '--test_zip', str(test_zip),
        '--output', str(output),
        '--report', str(report),
        '--model_output', str(model),
        '--folds', '2',
        '--epochs', '5',
        '--seed', '7',
    ], check=True)

    payload = json.loads(report.read_text(encoding='utf-8'))
    predictions = json.loads(output.read_text(encoding='utf-8'))
    assert payload['selection_data'] == 'QNRF train_eval only'
    assert payload['test_ground_truth_used'] is False
    assert payload['train_samples'] == 16
    assert payload['test_samples'] == 4
    assert all('gt_cnt' not in row for row in predictions)
    assert model.is_file()


def test_scorer_requires_explicit_benchmark_acknowledgement(tmp_path):
    predictions = tmp_path / 'predictions.json'
    reference = tmp_path / 'reference.json'
    output = tmp_path / 'score.json'
    write_rows(predictions, [10, 20])
    write_rows(reference, [0, 0], [10, 20])
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'score_frozen_count_predictions.py'
    result = subprocess.run([
        sys.executable,
        str(script),
        '--predictions', str(predictions),
        '--reference', str(reference),
        '--output', str(output),
    ])
    assert result.returncode != 0
    assert not output.exists()

    subprocess.run([
        sys.executable,
        str(script),
        '--predictions', str(predictions),
        '--reference', str(reference),
        '--output', str(output),
        '--allow_benchmark_test_scoring',
    ], check=True)
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['mae'] == 0.0
    assert payload['mse'] == 0.0
