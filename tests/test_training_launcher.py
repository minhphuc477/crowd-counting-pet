import sys
import unittest
from unittest import mock

from scripts.train_apglc_then_counthead import parse_args


class TrainingLauncherContractTest(unittest.TestCase):
    def parse(self, *arguments):
        argv = ['train_apglc_then_counthead.py', *arguments]
        with mock.patch.object(sys, 'argv', argv):
            return parse_args()

    def test_sha_default_is_apglc_then_visible_legacy_recovery(self):
        args = self.parse('--dataset_file', 'SHA')
        self.assertEqual(args.ifi_variant, 'none')
        self.assertEqual(args.stage1_recipe, 'vgg_apglc')
        self.assertEqual(
            args.stage2_recipe,
            'vgg_apglc_density_counthead_ft_legacy',
        )
        self.assertEqual(args.stage2_epochs, 80)
        self.assertNotIn('ifi', args.stage1_output.lower())

    def test_shb_default_is_apglc_without_ifi(self):
        args = self.parse('--dataset_file', 'SHB')
        self.assertEqual(args.ifi_variant, 'none')
        self.assertEqual(args.stage1_recipe, 'vgg_apglc')
        self.assertEqual(args.stage2_recipe, 'vgg_apglc_counthead_stage2_adapt')
        self.assertNotIn('ifi', args.stage1_output.lower())

    def test_ifi_requires_explicit_variant(self):
        args = self.parse('--dataset_file', 'SHA', '--ifi_variant', 'robust')
        self.assertEqual(args.stage1_recipe, 'vgg_pet_apg_rifi')
        self.assertIn('ifi', args.stage1_output.lower())


if __name__ == '__main__':
    unittest.main()
