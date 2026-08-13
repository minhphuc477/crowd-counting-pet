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


def test_zip_gate_is_selected_on_validation_and_applied_to_test(tmp_path):
    targets = [0, 50, 300, 900]
    pet = [20, 40, 300, 900]
    zip_counts = [0, 50, 280, 800]
    val_pet = tmp_path / 'val_pet.json'
    val_zip = tmp_path / 'val_zip.json'
    test_pet = tmp_path / 'test_pet.txt'
    test_zip = tmp_path / 'test_zip.txt'
    write_val(val_pet, pet, targets)
    write_val(val_zip, zip_counts, targets)
    write_test(test_pet, pet * 375)
    write_test(test_zip, zip_counts * 375)

    output = tmp_path / 'hybrid.txt'
    report = tmp_path / 'report.json'
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'select_nwpu_zip_low_gate.py'
    subprocess.run([
        sys.executable,
        str(script),
        '--val_pet', str(val_pet),
        '--val_zip', str(val_zip),
        '--test_pet', str(test_pet),
        '--test_zip', str(test_zip),
        '--gates', '100,500',
        '--alphas', '1.0',
        '--output', str(output),
        '--report', str(report),
    ], check=True)

    payload = json.loads(report.read_text(encoding='utf-8'))
    assert payload['best']['gate'] == 100.0
    assert payload['best']['alpha'] == 1.0
    assert payload['best']['mae'] == 0.0
    assert payload['test_zip_selected_images'] == 750
    lines = output.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 1500
    assert lines[:4] == [
        '1 0.000000',
        '2 50.000000',
        '3 300.000000',
        '4 900.000000',
    ]


def test_zip_gate_retains_pet_only_when_zip_does_not_help(tmp_path):
    targets = [10, 100]
    pet = [10, 100]
    zip_counts = [30, 130]
    val_pet = tmp_path / 'val_pet.json'
    val_zip = tmp_path / 'val_zip.json'
    test_pet = tmp_path / 'test_pet.txt'
    test_zip = tmp_path / 'test_zip.txt'
    write_val(val_pet, pet, targets)
    write_val(val_zip, zip_counts, targets)
    write_test(test_pet, pet * 750)
    write_test(test_zip, zip_counts * 750)

    output = tmp_path / 'hybrid.txt'
    report = tmp_path / 'report.json'
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'select_nwpu_zip_low_gate.py'
    subprocess.run([
        sys.executable,
        str(script),
        '--val_pet', str(val_pet),
        '--val_zip', str(val_zip),
        '--test_pet', str(test_pet),
        '--test_zip', str(test_zip),
        '--gates', '100',
        '--alphas', '0.5,1.0',
        '--output', str(output),
        '--report', str(report),
    ], check=True)

    payload = json.loads(report.read_text(encoding='utf-8'))
    assert payload['best']['gate'] == -1.0
    assert payload['best']['alpha'] == 0.0
    assert output.read_text(encoding='utf-8').splitlines()[0] == '1 10.000000'
