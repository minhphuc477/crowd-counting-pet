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
    if 'counthead_stage2' in recipe:
        argv.extend(('--resume', 'synthetic_stage1.pth', '--resume_model_only'))
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
        self.assertEqual(args.ifi_point_loss_type, 'mse')
        self.assertTrue(args.ifi_balance_pos_neg)
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

        features = torch.randn(3, args.hidden_dim)
        labels = torch.ones(3, dtype=torch.long)
        dense_mask = torch.tensor([False, True, True])
        cls_losses = []
        point_losses = []
        model._append_ifi_losses(
            features,
            labels,
            cls_losses,
            point_losses,
            dense_mask=dense_mask,
            offset_targets=torch.zeros(3, 2),
        )
        self.assertEqual(sum(loss.numel() for loss in cls_losses), 3)
        self.assertEqual(sum(loss.numel() for loss in point_losses), 3)

    def test_paper_count_head_stage_keeps_stage_one_ifi_architecture(self):
        stage1 = _recipe_args('vgg_apgcc_paper_ifi')
        stage2 = _recipe_args('vgg_apgcc_paper_ifi_counthead_stage2')
        architecture_keys = (
            'query_feature_interpolation',
            'query_ifi_sharing',
            'query_ifi_feature_source',
            'query_ifi_residual',
            'ifi_interpolation',
            'ifi_feature_source',
            'ifi_pos_dim',
            'ifi_mlp_hidden_dim',
            'ifi_activation',
            'ifi_head_source',
        )
        for key in architecture_keys:
            self.assertEqual(getattr(stage1, key), getattr(stage2, key), key)
        self.assertEqual(stage2.ifi_loss_coef, 0.0)
        self.assertGreater(stage2.count_head_loss_coef, 0.0)

    def test_robust_apg_ifi_starts_from_pet_identity_and_keeps_stage_architecture(self):
        stage1 = _recipe_args('vgg_pet_apg_rifi')
        stage2 = _recipe_args('vgg_pet_apg_rifi_counthead_stage2')
        self.assertTrue(stage1.query_ifi_residual)
        self.assertEqual(stage1.query_ifi_residual_init, 0.0)
        self.assertEqual(stage1.ifi_loss_coef, 0.02)
        self.assertEqual(stage1.lr_scheduler, 'step')
        self.assertEqual(stage1.lr_drop, 700)
        self.assertEqual(stage1.lr_gamma, 0.1)
        self.assertTrue(stage1.ifi_balance_pos_neg)
        self.assertEqual(stage1.ifi_point_loss_type, 'mse')
        for key in (
            'query_feature_interpolation',
            'query_ifi_sharing',
            'query_ifi_feature_source',
            'query_ifi_residual',
            'query_ifi_residual_init',
            'ifi_interpolation',
            'ifi_feature_source',
            'ifi_pos_dim',
            'ifi_mlp_hidden_dim',
            'ifi_activation',
            'ifi_head_source',
        ):
            self.assertEqual(getattr(stage1, key), getattr(stage2, key), key)

        interpolator = SharedMultiScaleIFI(
            1,
            pos_dim=2,
            mlp_hidden_dim=1,
            residual_base=True,
            residual_init=stage1.query_ifi_residual_init,
        )
        feature_maps = {
            '8x': torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
            '4x': torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8),
        }
        points = torch.tensor([[4.0, 4.0], [12.0, 12.0]])
        sampled = interpolator.sample_batch(
            feature_maps,
            points,
            (32, 32),
            primary='8x',
        )
        torch.testing.assert_close(
            sampled.squeeze(0).squeeze(-1),
            torch.tensor([0.0, 5.0]),
            rtol=0.0,
            atol=0.0,
        )

    def test_scale_rifi_has_a_single_controlled_scale_aware_delta(self):
        rifi = _recipe_args('vgg_apglc_rifi')
        scale_rifi = _recipe_args('vgg_apglc_scale_rifi')

        architecture_keys = (
            'query_feature_interpolation',
            'query_ifi_sharing',
            'query_ifi_feature_source',
            'query_ifi_residual',
            'query_ifi_residual_init',
            'ifi_interpolation',
            'ifi_feature_source',
            'ifi_loss_coef',
            'ifi_head_source',
            'ifi_point_coef',
            'ifi_pos_k',
            'ifi_neg_k',
            'ifi_start_epoch',
            'ifi_end_epoch',
        )
        for key in architecture_keys:
            self.assertEqual(getattr(rifi, key), getattr(scale_rifi, key), key)

        self.assertEqual(rifi.set_cost_context, 0.0)
        self.assertEqual(rifi.set_cost_query, 0.0)
        self.assertEqual(rifi.scale_point_loss_coef, 0.0)
        self.assertGreater(scale_rifi.set_cost_context, 0.0)
        self.assertGreater(scale_rifi.set_cost_query, 0.0)
        self.assertGreater(scale_rifi.scale_point_loss_coef, 0.0)

    def test_scale_rifi_count_head_stage_preserves_stage_one_architecture(self):
        stage1 = _recipe_args('vgg_apglc_scale_rifi')
        stage2 = _recipe_args('vgg_apglc_scale_rifi_counthead_stage2')
        architecture_keys = (
            'query_feature_interpolation',
            'query_ifi_sharing',
            'query_ifi_feature_source',
            'query_ifi_residual',
            'query_ifi_residual_init',
            'ifi_interpolation',
            'ifi_feature_source',
            'ifi_pos_dim',
            'ifi_mlp_hidden_dim',
            'ifi_activation',
            'ifi_head_source',
            'set_cost_context',
            'set_cost_query',
            'matcher_context_k',
            'matcher_context_min_scale',
            'scale_point_sigma',
            'scale_point_sigma_min',
            'scale_point_sigma_max',
            'scale_point_fallback',
            'scale_point_fallback_k',
            'scale_point_fallback_factor',
        )
        for key in architecture_keys:
            self.assertEqual(getattr(stage1, key), getattr(stage2, key), key)
        self.assertEqual(stage2.ifi_loss_coef, 0.0)
        self.assertEqual(stage2.scale_point_loss_coef, 0.0)
        self.assertGreater(stage2.count_head_loss_coef, 0.0)
        self.assertEqual(stage2.eval_count_source, 'pet')
        self.assertEqual(stage2.eval_count_mode, 'threshold')

    def test_scale_rifi_count_head_stage_rejects_fresh_training(self):
        argv = [
            '--model_recipe', 'vgg_apglc_scale_rifi_counthead_stage2',
            '--allow_experimental_model_recipe',
            '--dataset_file', 'SHB',
            '--data_path', 'unused',
            '--device', 'cpu',
            '--no_pretrained_backbone',
        ]
        args = main.get_args_parser().parse_args(argv)
        args._explicit_args = main.get_explicit_arg_names(argv)
        main.apply_model_recipe(args)
        with self.assertRaisesRegex(ValueError, 'requires a trained vgg_apglc_scale_rifi checkpoint'):
            main.sanitize_unstable_training_args(args)

    def test_density_routed_ifi_separates_sparse_and_dense_guidance(self):
        torch.manual_seed(0)
        args = _recipe_args('vgg_apglc_density_routed_ifi')
        self.assertEqual(args.apg_sparse_coef, 1.0)
        self.assertEqual(args.apg_dense_coef, 0.0)
        self.assertEqual(args.query_ifi_branch_scope, 'dense')
        self.assertEqual(args.query_ifi_residual_init, 1e-3)
        self.assertEqual(args.ifi_branch_scope, 'dense')
        self.assertEqual(args.ifi_negative_policy, 'paper')
        self.assertEqual(args.ifi_end_epoch, 350)
        self.assertEqual(args.lr_drop, 700)
        self.assertEqual(args.count_head_loss_coef, 0.0)

        model, criterion = build_model(args)
        self.assertIsNotNone(model.shared_query_feature_interpolator)
        self.assertIsNone(model.quadtree_sparse.query_feature_interpolator)
        self.assertIsNone(model.quadtree_dense.query_feature_interpolator)

        sampled_primaries = []
        original_sample_batch = model.shared_query_feature_interpolator.sample_batch

        def record_sample_batch(feature_maps, points_abs, image_shape, primary):
            sampled_primaries.append(primary)
            return original_sample_batch(
                feature_maps,
                points_abs,
                image_shape,
                primary,
            )

        model.shared_query_feature_interpolator.sample_batch = record_sample_batch
        model.train()
        image = torch.rand(3, 128, 128)
        points = torch.tensor([
            [30.0, 30.0],
            [32.0, 32.0],
            [34.0, 34.0],
            [90.0, 80.0],
        ])
        target = {
            'points': points,
            'labels': torch.ones(points.shape[0], dtype=torch.long),
            'density': torch.tensor(4.0),
        }
        output = model(
            [image],
            train=True,
            criterion=criterion,
            targets=[target],
            epoch=0,
        )

        self.assertTrue(torch.isfinite(output['losses']))
        self.assertIn('loss_ifi', output['loss_dict'])
        self.assertIn('loss_apg_sp', output['loss_dict'])
        self.assertEqual(output['weight_dict']['loss_apg_ds'], 0.0)
        self.assertTrue(sampled_primaries)
        self.assertEqual(set(sampled_primaries), {'4x'})

        output['losses'].backward()
        shared_ifi = model.shared_query_feature_interpolator
        self.assertIsNotNone(shared_ifi.fusion_scale.grad)
        self.assertTrue(torch.isfinite(shared_ifi.fusion_scale.grad).all())
        ifi_gradients = [
            parameter.grad
            for parameter in shared_ifi.interpolator.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(ifi_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in ifi_gradients))

        late_output = model(
            [image],
            train=True,
            criterion=criterion,
            targets=[target],
            epoch=351,
        )
        self.assertEqual(late_output['weight_dict']['loss_ifi'], 0.0)
        self.assertEqual(late_output['weight_dict']['loss_apg_sp'], 0.0)
        self.assertEqual(late_output['weight_dict']['loss_apg_ds'], 0.0)

        model.eval()
        with torch.no_grad():
            eval_output = model([image])
        self.assertTrue(torch.isfinite(eval_output['pred_logits']).all())
        self.assertTrue(torch.isfinite(eval_output['pred_points']).all())

    def test_paper_negative_policy_keeps_exact_dense_samples(self):
        args = _recipe_args('vgg_apglc_density_routed_ifi')
        model, _ = build_model(args)
        close_gt = torch.tensor([
            [64.0, 64.0],
            [66.0, 66.0],
            [68.0, 68.0],
        ])
        torch.manual_seed(7)
        negatives = model._build_ifi_negatives(close_gt, 128, 128)
        self.assertEqual(
            negatives.shape,
            (close_gt.shape[0] * args.ifi_neg_k, 2),
        )

    def test_density_routed_ifi_rejects_batched_inference_explicitly(self):
        args = _recipe_args('vgg_apglc_density_routed_ifi')
        model, _ = build_model(args)
        model.eval()
        with self.assertRaisesRegex(ValueError, 'batch_size=1'):
            model([torch.rand(3, 96, 128), torch.rand(3, 80, 112)])

    def test_nwpu_density_routed_recipe_preserves_dense_data_contract(self):
        args = _recipe_args('vgg_apglc_density_routed_ifi_nwpu')
        self.assertEqual(args.query_ifi_branch_scope, 'dense')
        self.assertEqual(args.ifi_branch_scope, 'dense')
        self.assertEqual(args.apg_dense_coef, 0.0)
        self.assertEqual(args.apg_end_epoch, 350)
        self.assertEqual(args.ifi_end_epoch, 350)
        self.assertEqual(args.crop_attempts, 12)
        self.assertEqual(args.min_crop_points, 1)
        self.assertEqual(args.nwpu_dense_crop_prob, 0.5)
        self.assertEqual(args.nwpu_dense_crop_attempts, 32)
        self.assertEqual(args.eval_tile_size, 1536)
        self.assertEqual(args.eval_tile_trigger_count, 1500.0)

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
            checkpoint_path = output_dir / 'best_checkpoint.pth'
            checkpoint_path.write_bytes(b'checkpoint')
            with self.assertRaises(FileExistsError):
                main.validate_training_output_dir(output_dir, checkpoint=None)
            main.validate_training_output_dir(
                output_dir,
                checkpoint={'model': {}},
                resume_path=checkpoint_path,
            )

    def test_cross_run_resume_rejects_existing_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / 'target'
            source_dir = root / 'source'
            output_dir.mkdir()
            source_dir.mkdir()
            (output_dir / 'best_checkpoint.pth').write_bytes(b'target')
            source_checkpoint = source_dir / 'best_checkpoint.pth'
            source_checkpoint.write_bytes(b'source')
            with self.assertRaises(FileExistsError):
                main.validate_training_output_dir(
                    output_dir,
                    checkpoint={'model': {}},
                    resume_path=source_checkpoint,
                )


if __name__ == '__main__':
    unittest.main()
