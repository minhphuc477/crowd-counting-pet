import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import savemat

from datasets.JHU import build as build_jhu
from datasets.JHU import load_jhu_annotation
from datasets.NWPU import build as build_nwpu
from datasets.QNRF import build as build_qnrf
from datasets.SHA import fixed_partial_annotation_mask
from datasets import build_dataset
from main import resolve_validation_protocol
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

    def test_train_eval_resizes_points_and_box_sigma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / 'train' / 'images'
            gt_dir = root / 'train' / 'gt'
            image_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            Image.new('RGB', (200, 100)).save(image_dir / '0001.jpg')
            (gt_dir / '0001.txt').write_text(
                '100 50 20 10 0 0\n',
                encoding='utf-8',
            )
            args = argparse.Namespace(
                data_path=str(root),
                eval_max_size=100,
                patch_size=32,
                patch_size_choices='',
                crop_attempts=1,
                min_crop_points=0,
            )
            image, target = build_jhu('train_eval', args)[0]
            self.assertEqual(tuple(image.shape[-2:]), (50, 100))
            self.assertEqual(target['points'].tolist(), [[25.0, 50.0]])
            expected = torch.tensor(
                [[np.sqrt(200.0) / 4.0, np.sqrt(200.0) / 2.0]],
                dtype=torch.float32,
            )
            self.assertTrue(torch.allclose(target['sigma'], expected))


class HighResolutionDatasetContractTests(unittest.TestCase):
    @staticmethod
    def _args(root):
        return argparse.Namespace(
            data_path=str(root),
            eval_max_size=100,
            patch_size=32,
            patch_size_choices='',
            crop_attempts=1,
            min_crop_points=0,
            nwpu_sigma_mode='area',
        )

    def test_qnrf_train_eval_uses_evaluation_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / 'Train'
            split.mkdir()
            Image.new('RGB', (200, 100)).save(split / 'img_0001.jpg')
            savemat(
                split / 'img_0001_ann.mat',
                {'annPoints': np.asarray([[100.0, 50.0]], dtype=np.float32)},
            )
            image, target = build_qnrf('train_eval', self._args(root))[0]
            self.assertEqual(tuple(image.shape[-2:]), (50, 100))
            self.assertEqual(target['points'].tolist(), [[25.0, 50.0]])

    def test_nwpu_train_eval_resizes_points_and_sigma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / 'images'
            json_dir = root / 'jsons'
            image_dir.mkdir()
            json_dir.mkdir()
            Image.new('RGB', (200, 100)).save(image_dir / '0001.jpg')
            (json_dir / '0001.json').write_text(
                json.dumps({
                    'points': [[100.0, 50.0]],
                    'sigma': [[10.0, 20.0]],
                }),
                encoding='utf-8',
            )
            (root / 'train.txt').write_text('0001\n', encoding='utf-8')
            image, target = build_nwpu('train_eval', self._args(root))[0]
            self.assertEqual(tuple(image.shape[-2:]), (50, 100))
            self.assertEqual(target['points'].tolist(), [[25.0, 50.0]])
            self.assertEqual(target['sigma'].tolist(), [[5.0, 10.0]])

    def test_nwpu_missing_split_file_does_not_fall_back_to_all_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / 'images'
            json_dir = root / 'jsons'
            image_dir.mkdir()
            json_dir.mkdir()
            Image.new('RGB', (32, 32)).save(image_dir / '0001.jpg')
            (json_dir / '0001.json').write_text(
                json.dumps({'points': [[16.0, 16.0]]}),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(FileNotFoundError, 'split file'):
                build_nwpu('train_eval', self._args(root))

    def test_nwpu_val_does_not_fall_back_to_test_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / 'images'
            json_dir = root / 'jsons'
            image_dir.mkdir()
            json_dir.mkdir()
            Image.new('RGB', (32, 32)).save(image_dir / '0001.jpg')
            (json_dir / '0001.json').write_text(
                json.dumps({'points': [[16.0, 16.0]]}),
                encoding='utf-8',
            )
            (root / 'test.txt').write_text('0001\n', encoding='utf-8')
            args = self._args(root)
            args.nwpu_eval_split = 'val'
            with self.assertRaisesRegex(FileNotFoundError, 'val.txt'):
                build_nwpu('val', args)

    def test_nwpu_hidden_test_split_does_not_require_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / 'images'
            image_dir.mkdir()
            Image.new('RGB', (32, 32)).save(image_dir / '0001.jpg')
            (root / 'test.txt').write_text('0001\n', encoding='utf-8')
            args = self._args(root)
            args.nwpu_eval_split = 'test'

            image, target = build_nwpu('val', args)[0]

            self.assertEqual(tuple(image.shape[-2:]), (32, 32))
            self.assertEqual(target['points'].shape, (0, 2))
            self.assertFalse(bool(target['has_annotation']))
            self.assertEqual(target['image_id'], '0001')


class DatasetRegistryContractTests(unittest.TestCase):
    def test_official_validation_aliases_do_not_change_protocol(self):
        for dataset_file in (
            'NWPU', 'NWPU_Crowd', 'NWPU-Crowd',
            'JHU', 'JHU_Crowd', 'JHU-Crowd++',
        ):
            args = argparse.Namespace(
                dataset_file=dataset_file,
                validation_protocol='auto',
            )
            self.assertEqual(resolve_validation_protocol(args), 'benchmark_test')

    def test_ambiguous_ucf_alias_is_rejected(self):
        args = argparse.Namespace(dataset_file='UCF')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            build_dataset('train', args)

    def test_ucfcc50_auto_validation_uses_training_holdout(self):
        args = argparse.Namespace(
            dataset_file='UCFCC50',
            validation_protocol='auto',
        )
        self.assertEqual(resolve_validation_protocol(args), 'train_holdout')


if __name__ == '__main__':
    unittest.main()
