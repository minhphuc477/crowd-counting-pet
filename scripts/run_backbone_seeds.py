"""Run training for a backbone across multiple random seeds and collect outputs.

Usage example:
  python scripts/run_backbone_seeds.py --backbone swinv2_base_window8_256 --seeds 42 7 13 99 --epochs 1500 --extra_args "--patch_size 256 --batch_size 2"

This script launches sequential runs of `main.py` with different `--seed` values and stores outputs under `outputs/<dataset>/<output_dir>_seed_<s>`.
"""
import argparse
import shlex
import subprocess
import os
from pathlib import Path


def run_seed(backbone, seed, base_output, extra_args, data_path=None, python_cmd="python"):
    out_dir = f"{base_output}_seed_{seed}"
    data_arg = f"--data_path {shlex.quote(data_path)}" if data_path else ""
    cmd = f"{python_cmd} main.py --backbone {shlex.quote(backbone)} --seed {seed} --output_dir {shlex.quote(out_dir)} {data_arg} {extra_args}"
    print('\nRunning:', cmd)
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--seeds', required=True, nargs='+', type=int)
    parser.add_argument('--dataset', default='SHA')
    parser.add_argument('--data_path', default=None, help='Path to dataset (e.g., ./data/ShanghaiTech/PartA)')
    parser.add_argument('--output_dir', default='', help='base output_dir suffix')
    parser.add_argument('--python_cmd', default='python')
    parser.add_argument('--extra_args', default='', help='extra CLI args to pass to main.py (quoted)')
    args = parser.parse_args()

    base_output = args.output_dir or args.backbone
    # ensure outputs folder exists
    Path('outputs').mkdir(exist_ok=True)

    for s in args.seeds:
        try:
            run_seed(args.backbone, s, base_output, args.extra_args, data_path=args.data_path, python_cmd=args.python_cmd)
        except subprocess.CalledProcessError as e:
            print(f"Run for seed {s} failed: {e}")
            # continue to next seed


if __name__ == '__main__':
    main()
