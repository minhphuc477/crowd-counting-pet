import numpy as np

from scripts.sweep_measure_inference import (
    count_metrics,
    inclusive_grid,
    parse_scale_set,
    search_candidates,
)


def test_parse_scale_set_deduplicates_and_validates():
    assert parse_scale_set("0.9,1.0,0.9") == (0.9, 1.0)


def test_inclusive_grid_contains_endpoints():
    assert inclusive_grid(0.9, 1.0, 0.05) == [0.9, 0.95, 1.0]


def test_count_metrics_uses_crowd_mae_and_rmse():
    metrics = count_metrics(np.asarray([8.0, 14.0]), np.asarray([10.0, 10.0]))
    assert metrics["mae"] == 3.0
    assert np.isclose(metrics["mse"], np.sqrt(10.0))
    assert metrics["bias"] == 1.0


def test_search_includes_baseline_and_finds_better_view_blend():
    targets = np.asarray([10.0, 20.0, 30.0])
    views = {
        "under": np.asarray([8.0, 18.0, 28.0]),
        "over": np.asarray([12.0, 22.0, 32.0]),
    }
    result = search_candidates(
        views,
        targets,
        ensemble_alphas=[0.5],
        calibration_scales=[1.0],
        calibration_biases=[0.0],
    )
    winner = result["winners"]["uncalibrated"]
    assert winner["mae"] == 0.0
    assert winner["kind"] == "view_blend"


def test_calibration_search_cannot_drop_the_identity_candidate():
    targets = np.asarray([10.0, 20.0])
    views = {"base": np.asarray([9.0, 19.0])}
    result = search_candidates(
        views,
        targets,
        ensemble_alphas=[],
        calibration_scales=[0.5],
        calibration_biases=[-10.0],
    )
    assert result["winners"]["affine_calibrated"]["mae"] <= 1.0
