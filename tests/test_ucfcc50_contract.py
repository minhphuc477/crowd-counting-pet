import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from datasets import build_dataset
from datasets.UCFCC50 import build_folds


class UCFCC50ContractTest(unittest.TestCase):
    def _dataset_root(self, root):
        images = root / 'images'
        annotations = root / 'ground_truth'
        images.mkdir()
        annotations.mkdir()
        for index in range(1, 51):
            Image.new('RGB', (32, 24), color=(index, 0, 0)).save(
                images / f'{index}.jpg'
            )
            savemat(
                annotations / f'{index}_ann.mat',
                {'annPoints': np.asarray([[10.0, 8.0]], dtype=np.float32)},
            )
        return root

    @staticmethod
    def _args(data_path, fold=0, manifest=''):
        return argparse.Namespace(
            dataset_file='UCFCC50',
            data_path=str(data_path),
            patch_size=16,
            patch_size_choices='',
            crop_attempts=1,
            min_crop_points=0,
            eval_max_size=0,
            ucfcc50_fold=fold,
            ucfcc50_fold_seed=17,
            ucfcc50_fold_manifest=str(manifest),
        )

    def test_seeded_folds_are_complete_disjoint_and_reproducible(self):
        stems = [str(index) for index in range(1, 51)]
        first = build_folds(stems, seed=17)
        second = build_folds(stems, seed=17)
        self.assertEqual(first, second)
        self.assertTrue(all(len(fold) == 10 for fold in first))
        flattened = [stem for fold in first for stem in fold]
        self.assertEqual(len(flattened), 50)
        self.assertEqual(len(set(flattened)), 50)

    def test_loader_uses_40_train_and_10_held_out_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._dataset_root(Path(tmp))
            args = self._args(root, fold=2)
            train = build_dataset('train_eval', args)
            val = build_dataset('val', args)

            self.assertEqual(len(train), 40)
            self.assertEqual(len(val), 10)
            train_ids = {Path(path).stem for path in train.img_list}
            val_ids = {Path(path).stem for path in val.img_list}
            self.assertFalse(train_ids & val_ids)
            self.assertEqual(len(train_ids | val_ids), 50)

            _, target = val[0]
            self.assertEqual(target['points'].shape, (1, 2))
            self.assertEqual(target['points'][0].tolist(), [8.0, 10.0])

    def test_manifest_must_cover_every_image_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._dataset_root(Path(tmp))
            folds = build_folds([str(index) for index in range(1, 51)], seed=7)
            manifest = root / 'folds.json'
            manifest.write_text(
                json.dumps({'folds': folds}),
                encoding='utf-8',
            )
            dataset = build_dataset('val', self._args(root, manifest=manifest))
            self.assertEqual(tuple(folds[0]), dataset.test_stems)

            folds[1][0] = folds[0][0]
            manifest.write_text(
                json.dumps({'folds': folds}),
                encoding='utf-8',
            )
            with self.assertRaises(ValueError):
                build_dataset('val', self._args(root, manifest=manifest))


if __name__ == '__main__':
    unittest.main()
