import json

from scripts.train_count_reliability_router import write_count_submission
from scripts.train_qnrf_reliability_router import load_predictions


def test_test_prediction_loader_discards_ground_truth(tmp_path):
    path = tmp_path / 'test.json'
    path.write_text(json.dumps([
        {
            'image_id': '3001',
            'image_path': 'images/3001.jpg',
            'pred_cnt': 17.5,
            'gt_cnt': 999999,
        },
    ]), encoding='utf-8')

    rows = load_predictions(path, require_gt=False)

    assert rows['3001']['pred'] == 17.5
    assert 'gt' not in rows['3001']


def test_router_submission_follows_official_manifest_order(tmp_path):
    manifest = tmp_path / 'test.txt'
    manifest.write_text('images/3002.jpg\nimages/3001.jpg\n', encoding='utf-8')
    output = tmp_path / 'submission.txt'
    rows = [
        {'image_id': '3001', 'pred_cnt': 11.0},
        {'image_id': '3002', 'pred_cnt': 22.5},
    ]

    write_count_submission(output, rows, manifest)

    assert output.read_text(encoding='utf-8').splitlines() == [
        '3002 22.500000',
        '3001 11.000000',
    ]
