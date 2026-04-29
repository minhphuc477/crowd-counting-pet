"""
Auto-evaluate ensemble once all training seeds complete.

Usage:
  python scripts/monitor_and_evaluate.py --backbone maxvit_small_tf_224 --num_seeds 5

This script:
1. Waits for all N seed training runs to complete
2. Automatically runs ensemble evaluation
3. Reports final MAE
"""

import argparse
import time
import glob
from pathlib import Path
import subprocess
import sys


def wait_for_checkpoint(checkpoint_path, timeout_seconds=86400):
    """Wait for checkpoint file to exist (with timeout)."""
    start_time = time.time()
    while True:
        if Path(checkpoint_path).exists():
            # Give file a moment to finish writing
            time.sleep(2)
            return True
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"Timeout waiting for {checkpoint_path}")
            return False
        print(f"Waiting for {checkpoint_path}... ({elapsed:.0f}s)")
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--dataset', default='SHA')
    args = parser.parse_args()
    
    print(f"Monitoring ensemble training for {args.backbone} ({args.num_seeds} seeds)...")
    print()
    
    # This script assumes run_backbone_seeds.py is already running
    # We just wait for all checkpoints to appear, then evaluate
    
    checkpoints = []
    for i in range(args.num_seeds):
        seed_idx = [42, 7, 13, 99, 1234][i] if i < 5 else 1000 + i
        checkpoint_path = f"outputs/{args.dataset}/{args.backbone}_seed_{seed_idx}/best_checkpoint.pth"
        checkpoints.append(checkpoint_path)
        print(f"Waiting for seed {seed_idx}: {checkpoint_path}")
    
    print()
    print("This may take several hours. Waiting for all checkpoints...")
    print()
    
    for checkpoint_path in checkpoints:
        if not wait_for_checkpoint(checkpoint_path):
            print(f"Failed to find checkpoint: {checkpoint_path}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("All checkpoints ready! Running ensemble evaluation...")
    print("="*60 + "\n")
    
    # Run ensemble evaluation
    checkpoint_pattern = f"outputs/{args.dataset}/{args.backbone}_seed_*/best_checkpoint.pth"
    cmd = [
        "python", "scripts/ensemble_evaluate.py",
        "--backbone", args.backbone,
        "--checkpoints", checkpoint_pattern,
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
