import json
import subprocess
import sys
from pathlib import Path


def write_val(path, predictions, targets):
    rows = [
        {'image_id': str(index + 1), 'pred_cnt': pred, 'gt_cnt': gt}
        for index, (pred, gt) in enumerate(zip(predictions, targets))
    ]
    path.write_text(json.dumps(rows), encoding='utf-8')


def write_test(path, predictions):
    path.write_text(
        ''.join(f'{index + 1} {pred}\n' for index, pred in enumerate(predictions)),
        encoding='utf-8',
    )


def test_adaptive_selector_uses_only_validation_to_route_test(tmp_path):
    targets = [0, 60, 300, 900]
    reference = [20, 60, 300, 700]
    conservative = [0, 50, 280, 650]
    high_recall = [30, 70, 320, 900]
    files = {}
    for name, values in (
        ('reference', reference),
        ('conservative', conservative),
        ('high_recall', high_recall),
    ):
        files[f'val_{name}'] = tmp_path / f'val_{name}.json'
        write_val(files[f'val_{name}'], values, targets)
        files[f'test_{name}'] = tmp_path / f'test_{name}.txt'
        write_test(files[f'test_{name}'], values * 375)

    output = tmp_path / 'adaptive.txt'
    report = tmp_path / 'report.json'
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'select_nwpu_adaptive_count.py'
    subprocess.run([
        sys.executable,
        str(script),
        '--val_reference', str(files['val_reference']),
        '--val_conservative', str(files['val_conservative']),
        '--val_high_recall', str(files['val_high_recall']),
        '--test_reference', str(files['test_reference']),
        '--test_conservative', str(files['test_conservative']),
        '--test_high_recall', str(files['test_high_recall']),
        '--low_gates', '25',
        '--high_gates', '500',
        '--output', str(output),
        '--report', str(report),
    ], check=True)

    payload = json.loads(report.read_text(encoding='utf-8'))
    assert payload['best']['mae'] == 0.0
    assert payload['test_usage'] == {
        'conservative': 375,
        'reference': 750,
        'high_recall': 375,
    }
    lines = output.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 1500
    assert lines[0] == '1 0.000000'
