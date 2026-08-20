import pytest

from scripts.select_jhu_pml_blend import alpha_grid, blend_metrics


def test_jhu_blend_selector_finds_complementary_midpoint():
    rows = [
        {'gt_cnt': 10, 'pet_pred_cnt': 8, 'measure_pred_cnt': 12},
        {'gt_cnt': 20, 'pet_pred_cnt': 24, 'measure_pred_cnt': 16},
    ]

    assert blend_metrics(rows, 0.0)['mae'] == pytest.approx(3.0)
    assert blend_metrics(rows, 1.0)['mae'] == pytest.approx(3.0)
    assert blend_metrics(rows, 0.5)['mae'] == pytest.approx(0.0)
    assert 0.5 in alpha_grid(0.25)


def test_jhu_blend_selector_rejects_invalid_grid():
    with pytest.raises(ValueError, match='alpha_step'):
        alpha_grid(0.0)
