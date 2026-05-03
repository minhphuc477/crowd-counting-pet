#!/usr/bin/env python3
"""
Test evaluation script for a single checkpoint.
Runs eval.py for a specific checkpoint and shows detailed output.

Usage:
  python scripts/test_eval.py --checkpoint outputs/SHA/maxvit_rmlp_tiny_poly/best_checkpoint.pth --backbone maxvit_rmlp_tiny_poly --dataset_file SHA --data_path ./data/ShanghaiTech/part_A
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser('Test evaluation for single checkpoint')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--backbone', required=True, help='Backbone name')
    parser.add_argument('--dataset_file', default='SHA')
    parser.add_argument('--data_path', default='./data/ShanghaiTech/part_A')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    cmd = [
        sys.executable,
        'eval.py',
        '--backbone', args.backbone,
        '--dataset_file', args.dataset_file,
        '--data_path', args.data_path,
        '--resume', args.checkpoint,
        '--device', args.device,
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
