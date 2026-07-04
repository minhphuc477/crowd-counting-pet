import unittest

import torch
from torch import nn

from engine import (
    _pred_points_to_image_pixels,
    _predict_count_tiled,
    _valid_hw,
)
from util.misc import NestedTensor


class EvaluationCoordinateContractTest(unittest.TestCase):
    def test_predictions_are_denormalized_with_model_padding(self):
        tensors = torch.zeros(1, 3, 256, 512)
        mask = torch.ones(1, 256, 512, dtype=torch.bool)
        mask[:, :120, :300] = False
        samples = NestedTensor(tensors, mask)
        normalized = torch.tensor([
            [60.0 / 256.0, 150.0 / 512.0],
        ])

        self.assertEqual(_valid_hw(samples), (120, 300))
        pixels = _pred_points_to_image_pixels(normalized, samples)

        torch.testing.assert_close(
            pixels,
            torch.tensor([[60.0, 150.0]]),
        )

    def test_tiled_predictions_preserve_padded_coordinate_contract(self):
        class CenterModel(nn.Module):
            def forward(self, samples, **_kwargs):
                model_h, model_w = samples.tensors.shape[-2:]
                point = samples.tensors.new_tensor([
                    100.0 / float(model_h),
                    100.0 / float(model_w),
                ])
                return {
                    'pred_points': point.reshape(1, 1, 2),
                    'pred_logits': samples.tensors.new_tensor(
                        [[[0.0, 1.0]]],
                    ),
                }

        tensors = torch.zeros(1, 3, 512, 512)
        mask = torch.ones(1, 512, 512, dtype=torch.bool)
        mask[:, :300, :300] = False
        samples = NestedTensor(tensors, mask)
        outputs, count = _predict_count_tiled(
            CenterModel(),
            samples,
            [{'points': torch.empty(0, 2)}],
            tile_size=200,
        )
        pixels = _pred_points_to_image_pixels(
            outputs['pred_points'][0],
            samples,
        )

        self.assertEqual(count, 4.0)
        torch.testing.assert_close(
            pixels,
            torch.tensor([
                [100.0, 100.0],
                [100.0, 200.0],
                [200.0, 100.0],
                [200.0, 200.0],
            ]),
        )

    def test_resized_trigger_predictions_map_back_to_original_image(self):
        source_tensors = torch.zeros(1, 3, 256, 256)
        source_mask = torch.ones(1, 256, 256, dtype=torch.bool)
        source_mask[:, :150, :200] = False
        source = NestedTensor(source_tensors, source_mask)

        reference_tensors = torch.zeros(1, 3, 512, 512)
        reference_mask = torch.ones(1, 512, 512, dtype=torch.bool)
        reference_mask[:, :300, :400] = False
        reference = NestedTensor(reference_tensors, reference_mask)
        normalized = torch.tensor([
            [75.0 / 256.0, 100.0 / 256.0],
        ])

        pixels = _pred_points_to_image_pixels(
            normalized,
            source,
            reference_samples=reference,
        )

        torch.testing.assert_close(
            pixels,
            torch.tensor([[150.0, 200.0]]),
        )


if __name__ == '__main__':
    unittest.main()
