import tempfile
import unittest
from pathlib import Path

import torch

from datasets.JHU import load_jhu_annotation
from datasets.SHA import fixed_partial_annotation_mask
from models.matcher import HungarianMatcher, get_query_supervision_mask


class PartialAnnotationContractTests(unittest.TestCase):
    def test_fixed_region_is_deterministic_and_has_requested_area(self):
        first, first_bounds = fixed_partial_annotation_mask(
            (100, 200),
            'IMG_1.jpg',
            ratio=0.1,
            seed=42,
            height_ratio=0.5,
        )
        second, second_bounds = fixed_partial_annotation_mask(
            (100, 200),
            'IMG_1.jpg',
            ratio=0.1,
            seed=42,
            height_ratio=0.5,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_bounds, second_bounds)
        self.assertEqual(first.sum().item(), 2000)

    def test_matcher_never_uses_queries_outside_annotated_region(self):
        outputs = {
            'pred_logits': torch.tensor([[
                [0.0, 4.0],
                [0.0, 3.0],
                [0.0, 2.0],
                [0.0, 1.0],
            ]]),
            'pred_points': torch.tensor([[
                [0.25, 0.25],
                [0.25, 0.75],
                [0.75, 0.25],
                [0.75, 0.75],
            ]]),
            'points_queries': torch.tensor([
                [0.25, 0.25],
                [0.25, 0.75],
                [0.75, 0.25],
                [0.75, 0.75],
            ]),
            'img_shape': (100, 100),
        }
        mask = torch.zeros(100, 100, dtype=torch.bool)
        mask[:, :50] = True
        target = {
            'points': torch.tensor([[25.0, 25.0], [75.0, 25.0]]),
            'labels': torch.ones(2, dtype=torch.long),
            'supervision_mask': mask,
        }

        valid = get_query_supervision_mask(outputs, target)
        self.assertEqual(valid.tolist(), [True, False, True, False])
        indices = HungarianMatcher(1.0, 1.0)(outputs, [target])
        self.assertTrue(set(indices[0][0].tolist()).issubset({0, 2}))


class JHUContractTests(unittest.TestCase):
    def test_six_field_annotation_and_empty_distractor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.txt'
            path.write_text(
                '10 20 8 18 1 0\n30 40 10 16 2 1\n',
                encoding='utf-8',
            )
            points, sigma = load_jhu_annotation(
                path,
                image_size=(100, 100),
            )
            self.assertEqual(
                points.tolist(),
                [[20.0, 10.0], [40.0, 30.0]],
            )
            self.assertEqual(sigma.shape, (2, 2))
            self.assertTrue((sigma[:, 1] >= sigma[:, 0]).all())

            path.write_text('', encoding='utf-8')
            points, sigma = load_jhu_annotation(
                path,
                image_size=(100, 100),
            )
            self.assertEqual(points.shape, (0, 2))
            self.assertEqual(sigma.shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
