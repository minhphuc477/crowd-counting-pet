import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

import main
from models import build_model
from models.pet import ImplicitFeatureInterpolator


class _KeepFeature(nn.Module):
    def forward(self, value):
        return value[..., :1]


def _branch_ifi_args():
    argv = [
        '--model_recipe', 'vgg_pet_branch_ifi',
        '--dataset_file', 'SHB',
        '--data_path', 'unused',
        '--output_dir', 'outputs/_ifi_test',
        '--device', 'cpu',
        '--no_pretrained_backbone',
        '--hidden_dim', '64',
        '--dim_feedforward', '128',
        '--nheads', '8',
    ]
    args = main.get_args_parser().parse_args(argv)
    args._explicit_args = main.get_explicit_arg_names(argv)
    main.apply_model_recipe(args)
    return main.sanitize_unstable_training_args(args)


class IFIContractTest(unittest.TestCase):
    def test_image_centers_map_exactly_to_feature_centers(self):
        interpolator = ImplicitFeatureInterpolator(1, pos_dim=2, mlp_hidden_dim=1)
        interpolator.mlp = _KeepFeature()
        feature = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        points = torch.tensor([[4.0, 4.0], [12.0, 12.0]])

        sampled = interpolator.sample_points(feature, 0, points, 32, 32).squeeze(1)

        torch.testing.assert_close(sampled, torch.tensor([0.0, 5.0]), rtol=0.0, atol=0.0)

    def test_branch_recipe_is_a_controlled_scratch_ablation(self):
        args = _branch_ifi_args()
        self.assertEqual(args.epochs, 1500)
        self.assertEqual(args.apg_loss_coef, 0.0)
        self.assertEqual(args.count_head_loss_coef, 0.0)
        self.assertEqual(args.ifi_loss_coef, 0.02)
        self.assertEqual(args.ifi_end_epoch, -1)
        self.assertEqual(args.query_ifi_sharing, 'independent')
        self.assertEqual(args.ifi_feature_source, 'branch')
        self.assertEqual(args.branch_target_routing, 'none')
        self.assertEqual(args.split_loss_variant, 'paper')

    def test_independent_branch_ifi_is_used_by_training_loss(self):
        torch.manual_seed(0)
        args = _branch_ifi_args()
        model, criterion = build_model(args)
        model.train()
        image = torch.rand(3, 128, 128)
        points = torch.tensor([[30.0, 30.0], [34.0, 34.0], [90.0, 80.0]])
        target = {
            'points': points,
            'labels': torch.ones(points.shape[0], dtype=torch.long),
            'density': torch.tensor(float(points.shape[0])),
        }

        output = model([image], train=True, criterion=criterion, targets=[target], epoch=0)

        self.assertTrue(torch.isfinite(output['losses']))
        self.assertIn('loss_ifi', output['loss_dict'])
        self.assertNotIn('loss_apg_sp', output['loss_dict'])
        self.assertNotIn('loss_apg_ds', output['loss_dict'])
        self.assertIsNone(model.ifi_interpolator)
        output['losses'].backward()
        for branch in (model.quadtree_sparse, model.quadtree_dense):
            gradients = [
                parameter.grad
                for parameter in branch.query_feature_interpolator.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_scratch_run_rejects_existing_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / 'best_checkpoint.pth').write_bytes(b'checkpoint')
            with self.assertRaises(FileExistsError):
                main.validate_training_output_dir(output_dir, checkpoint=None)
            main.validate_training_output_dir(output_dir, checkpoint={'model': {}})


if __name__ == '__main__':
    unittest.main()
