import unittest

import numpy as np

from scripts.complete_partial_annotations import select_completed_points


class PartialCompletionContractTest(unittest.TestCase):
    def test_keeps_observed_gt_and_only_band_pseudo_points(self):
        gt = np.asarray([
            [20.0, 20.0],
            [80.0, 80.0],
        ])
        pred = np.asarray([
            [20.5, 20.5],  # inside annotated region
            [45.0, 20.0],  # outside, inside expanded band
            [90.0, 90.0],  # outside expanded band
            [42.0, 22.0],  # low confidence
        ])
        scores = np.asarray([0.9, 0.8, 0.9, 0.2])
        completed, is_pseudo, confidence = select_completed_points(
            gt,
            pred,
            scores,
            bounds=(10, 10, 40, 40),
            image_shape=(100, 100),
            band_pixels=10,
            min_score=0.5,
            dedup_radius=2.0,
        )
        np.testing.assert_allclose(
            completed,
            np.asarray([[20.0, 20.0], [45.0, 20.0]]),
        )
        np.testing.assert_array_equal(is_pseudo, np.asarray([0, 1]))
        np.testing.assert_allclose(confidence, np.asarray([1.0, 0.8]))

    def test_zero_band_uses_all_unannotated_regions(self):
        completed, is_pseudo, _ = select_completed_points(
            np.asarray([[20.0, 20.0]]),
            np.asarray([[90.0, 90.0]]),
            np.asarray([0.9]),
            bounds=(10, 10, 40, 40),
            image_shape=(100, 100),
            band_pixels=0,
            min_score=0.5,
            dedup_radius=2.0,
        )
        self.assertEqual(completed.shape[0], 2)
        self.assertEqual(int(is_pseudo.sum()), 1)


if __name__ == '__main__':
    unittest.main()
