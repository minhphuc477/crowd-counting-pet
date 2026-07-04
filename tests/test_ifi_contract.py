import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

import main
from models import build_model
from models.pet import ImplicitFeatureInterpolator, SharedMultiScaleIFI


class _KeepFeature(nn.Module):
    def forward(self, value):
        return value[..., :1]


def _recipe_args(recipe):
    argv = [
        '--model_recipe', recipe,
        '--dataset_file', 'SHB',
        '--data_path', 'unused',
        '--output_dir', 'outputs/_ifi_test',
        '--device', 'cpu',
        '--no_pretrained_backbone',
        '--hidden_dim', '64',
        '--dim_feedforward', '128',
        '--nheads', '8',
    ]
    if recipe in main.EXPERIMENTAL_MODEL_RECIPES:
        argv.append('--allow_experimental_model_recipe')
    args = main.get_args_parser().parse_args(argv)
    args._explicit_args = main.get_explicit_arg_names(argv)
    main.apply_model_recipe(args)
    return main.sanitize_unstable_training_args(args)


def _branch_ifi_args():
    return _recipe_args('vgg_pet_branch_ifi')


class IFIContractTest(unittest.TestCase):
    def test_paper_apg_recipe_uses_independent_auxiliary_points(self):
        args = _recipe_args('vgg_apgcc_paper_ifi')
        self.assertEqual(args.apg_loss_coef, 0.0)
        self.assertEqual(args.ifi_loss_coef, 0.2)
        self.assertEqual(args.ifi_pos_k, 2)
        self.assertEqual(args.ifi_pos_radius, 2.0)
        self.assertEqual(args.ifi_neg_k, 2)
        self.assertEqual(args.ifi_neg_min_dist, 2.0)
        self.assertEqual(args.ifi_neg_radius, 8.0)
        self.assertTrue(args.ifi_random_sampling)
        self.assertEqual(args.ifi_end_epoch, -1)

        model, _ = build_model(args)
        gt = torch.tensor([[64.0, 64.0], [96.0, 96.0]])
        torch.manual_seed(4)
        positives, targets = model._build_ifi_positives(gt, 160, 160)
        self.assertEqual(positives.shape, (4, 2))
        self.assertEqual(targets.shape, (4, 2))
        self.assertTrue(((positives - targets).abs() <= 2.0).all())
        torch.testing.assert_close(targets[0], gt[0])
        torch.testing.assert_close(targets[1], gt[0])
        torch.testing.assert_close(targets[2], gt[1])
        torch.testing.assert_close(targets[3], gt[1])

        isolated_gt = torch.tensor([[80.0, 80.0]])
        negatives = model._build_ifi_negatives(isolated_gt, 160, 160)
        self.assertEqual(negatives.shape, (2, 2))
        displacement = (negatives - isolated_gt).abs()
        self.assertTrue((displacement >= 2.0).all())
        self.assertTrue((displacement <= 8.0).all())

    def test_image_centers_map_exactly_to_feature_centers(self):
        interpolator = ImplicitFeatureInterpolator(1, pos_dim=2, mlp_hidden_dim=1)
        interpolator.mlp = _KeepFeature()
        feature = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        points = torch.tensor([[4.0, 4.0], [12.0, 12.0]])

        sampled = interpolator.sample_points(feature, 0, points, 32, 32).squeeze(1)

        torch.testing.assert_close(sampled, torch.tensor([0.0, 5.0]), rtol=0.0, atol=0.0)

    def test_residual_multiscale_ifi_starts_as_native_pet_features(self):
        interpolator = SharedMultiScaleIFI(
            1,
            pos_dim=2,
            mlp_hidden_dim=1,
            residual_base=True,
            residual_init=0.0,
        )
        feature_maps = {
            '8x': torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
            '4x': torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8),
        }
        points = torch.tensor([[4.0, 4.0], [12.0, 12.0]])

        sampled = interpolator.sample_batch(feature_maps, points, (32, 32), primary='8x')

        torch.testing.assert_close(
            sampled.squeeze(0).squeeze(-1),
            torch.tensor([0.0, 5.0]),
            rtol=0.0,
            atol=0.0,
        )

    def test_residual_mode_preserves_shared_ifi_checkpoint_keys(self):
        legacy = SharedMultiScaleIFI(8, residual_base=False)
        residual = SharedMultiScaleIFI(8, residual_base=True)

        self.assertEqual(set(legacy.state_dict()), set(residual.state_dict()))

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

    def test_original_annotations_can_run_as_custom_decoder_queries(self):
        torch.manual_seed(0)
        args = _branch_ifi_args()
        model, _ = build_model(args)
        model.eval()
        image = torch.rand(3, 128, 128)
        points = torch.tensor([
            [16.0, 20.0],
            [63.0, 71.0],
            [110.0, 100.0],
        ])
        with torch.no_grad():
            output = model(
                [image],
                test=True,
                epoch=0,
                custom_query_points=points,
            )
        self.assertEqual(output['pred_logits'].shape, (1, 3, 2))
        self.assertEqual(output['pred_points'].shape, (1, 3, 2))
        self.assertEqual(output['pred_offsets'].shape, (1, 3, 2))
        self.assertEqual(output['custom_dense_mask'].shape, (1, 3))
        torch.testing.assert_close(
            output['points_queries'][0],
            points / 256.0,
        )
        self.assertTrue(torch.isfinite(output['pred_points']).all())

    def test_rmi_recipe_is_shared_residual_and_trains_both_paths(self):
        torch.manual_seed(0)
        args = _recipe_args('vgg_pet_rmi')
        self.assertEqual(args.epochs, 1500)
        self.assertEqual(args.apg_loss_coef, 0.0)
        self.assertEqual(args.count_head_loss_coef, 0.0)
        self.assertEqual(args.ifi_loss_coef, 0.2)
        self.assertEqual(args.query_ifi_sharing, 'shared')
        self.assertEqual(args.query_ifi_feature_source, 'fpn4x8x')
        self.assertTrue(args.query_ifi_residual)
        self.assertEqual(args.split_loss_variant, 'paper')

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
        output['losses'].backward()

        shared_ifi = model.shared_query_feature_interpolator
        self.assertIsNotNone(shared_ifi)
        self.assertIsNotNone(shared_ifi.fusion_scale.grad)
        self.assertTrue(torch.isfinite(shared_ifi.fusion_scale.grad).all())
        for module in (shared_ifi.interpolator, shared_ifi.fusion):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

        model.eval()
        with torch.no_grad():
            eval_output = model([image], test=True, epoch=0)
        self.assertTrue(torch.isfinite(eval_output['pred_logits']).all())
        self.assertTrue(torch.isfinite(eval_output['pred_points']).all())

    def test_scratch_run_rejects_existing_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / 'best_checkpoint.pth').write_bytes(b'checkpoint')
            with self.assertRaises(FileExistsError):
                main.validate_training_output_dir(output_dir, checkpoint=None)
            main.validate_training_output_dir(output_dir, checkpoint={'model': {}})


if __name__ == '__main__':
    unittest.main()
