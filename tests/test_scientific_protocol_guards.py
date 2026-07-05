import argparse
import tempfile
import unittest
from pathlib import Path

import torch

from engine import evaluate
from main import resolve_validation_protocol
from scripts.sweep_eval_thresholds import resolve_runtime_args


class EvaluationLeakageGuardTests(unittest.TestCase):
    def test_ground_truth_tiling_trigger_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ground-truth count"):
            evaluate(
                model=None,
                data_loader=None,
                device=torch.device("cpu"),
                eval_tile_min_gt=1,
            )

    def test_benchmark_test_checkpoint_selection_is_rejected(self):
        args = argparse.Namespace(
            dataset_file="SHA",
            validation_protocol="benchmark_test",
            allow_benchmark_test_selection=False,
        )
        with self.assertRaisesRegex(ValueError, "benchmark test split"):
            resolve_validation_protocol(args)

    def test_final_test_once_is_allowed_for_full_data_refit(self):
        args = argparse.Namespace(
            dataset_file="QNRF",
            validation_protocol="final_test_once",
            eval_before_train=False,
        )
        self.assertEqual(resolve_validation_protocol(args), "final_test_once")

    def test_final_test_once_rejects_official_validation_dataset(self):
        args = argparse.Namespace(
            dataset_file="NWPU",
            validation_protocol="final_test_once",
            eval_before_train=False,
        )
        with self.assertRaisesRegex(ValueError, "only defined"):
            resolve_validation_protocol(args)


class SweepProtocolTests(unittest.TestCase):
    @staticmethod
    def _args(checkpoint, **overrides):
        values = {
            "resume": str(checkpoint),
            "dataset_file": "",
            "data_path": "",
            "backbone": "",
            "ucfcc50_fold": None,
            "ucfcc50_fold_seed": None,
            "ucfcc50_fold_manifest": "",
            "train_holdout_fraction": None,
            "train_holdout_seed": None,
            "output_dir": "",
            "eval_image_set": "train_holdout",
            "allow_benchmark_test_sweep": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_sweep_inherits_exact_holdout_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pth"
            torch.save(
                {
                    "args": argparse.Namespace(
                        dataset_file="SHA",
                        data_path="data/SHA",
                        backbone="vgg16_bn",
                        train_holdout_fraction=0.2,
                        train_holdout_seed=137,
                    )
                },
                checkpoint,
            )
            args = resolve_runtime_args(self._args(checkpoint))
            self.assertEqual(args.train_holdout_fraction, 0.2)
            self.assertEqual(args.train_holdout_seed, 137)

    def test_sweep_rejects_benchmark_test_threshold_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pth"
            torch.save(
                {
                    "args": argparse.Namespace(
                        dataset_file="SHB",
                        data_path="data/SHB",
                        backbone="vgg16_bn",
                    )
                },
                checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "benchmark test split"):
                resolve_runtime_args(
                    self._args(checkpoint, eval_image_set="val")
                )


if __name__ == "__main__":
    unittest.main()
