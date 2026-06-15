import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import util.misc as utils
from datasets import build_dataset
from engine import evaluate
from main import get_args_parser as get_train_args_parser
from main import merge_checkpoint_args, seed_worker, set_reproducibility
from models import build_model


def get_args():
    parser = argparse.ArgumentParser("Debug PET checkpoint loading and counting")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--dataset_file", default="SHA")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--checkpoint_model_key", default="model", choices=("model", "model_ema", "model_raw"))
    parser.add_argument("--resume_allow_arch_change", action="store_true")
    parser.add_argument("--backbone", default="vgg16_bn")
    parser.add_argument("--timm_adapter", default="lite_fpn")
    parser.add_argument("--score_threshold", default=0.575, type=float)
    parser.add_argument("--split_threshold", default=0.47, type=float)
    parser.add_argument("--eval_nms_radius", default=4.0, type=float)
    parser.add_argument("--eval_branch_gate", default="pred", choices=("none", "query", "pred"))
    parser.add_argument("--eval_soft_split_gate", default="pred", choices=("none", "query", "pred"))
    parser.add_argument("--eval_count_mode", default="threshold", choices=("threshold", "count_head_topk"))
    parser.add_argument("--eval_count_head_min_score", default=0.5, type=float)
    parser.add_argument("--eval_score_calibration", default="none", choices=("none", "count_head_bias"))
    parser.add_argument("--eval_score_calibration_strength", default=1.0, type=float)
    parser.add_argument("--eval_score_calibration_min_bias", default=-8.0, type=float)
    parser.add_argument("--eval_score_calibration_max_bias", default=8.0, type=float)
    parser.add_argument("--eval_debug_counting", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def main():
    args = get_args()
    checkpoint_path = Path(args.resume)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cli_args = get_train_args_parser().parse_args([])
    for key, value in vars(args).items():
        setattr(cli_args, key, value)
    cli_args.resume_model_only = True
    cli_args._explicit_args = set(vars(args).keys())
    merged_args = merge_checkpoint_args(cli_args, checkpoint)
    merged_args.resume = str(checkpoint_path)
    merged_args.resume_model_only = True
    merged_args.resume_allow_arch_change = bool(args.resume_allow_arch_change)
    merged_args.device = args.device
    merged_args.num_workers = args.num_workers
    merged_args.dataset_file = args.dataset_file
    merged_args.data_path = args.data_path
    merged_args.score_threshold = args.score_threshold
    merged_args.split_threshold = args.split_threshold
    merged_args.eval_nms_radius = args.eval_nms_radius
    merged_args.eval_branch_gate = args.eval_branch_gate
    merged_args.eval_soft_split_gate = args.eval_soft_split_gate
    merged_args.eval_count_mode = args.eval_count_mode
    merged_args.eval_count_head_min_score = args.eval_count_head_min_score
    merged_args.eval_score_calibration = args.eval_score_calibration
    merged_args.eval_score_calibration_strength = args.eval_score_calibration_strength
    merged_args.eval_score_calibration_min_bias = args.eval_score_calibration_min_bias
    merged_args.eval_score_calibration_max_bias = args.eval_score_calibration_max_bias
    merged_args.eval_debug_counting = bool(args.eval_debug_counting)

    set_reproducibility(args.seed, deterministic=True)
    device = torch.device(args.device)

    model, _ = build_model(merged_args)
    state = checkpoint[args.checkpoint_model_key]
    incompatible = model.load_state_dict(state, strict=not args.resume_allow_arch_change)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    model.to(device)

    dataset_val = build_dataset(image_set="val", args=merged_args)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    data_loader_val = DataLoader(
        dataset_val,
        1,
        sampler=SequentialSampler(dataset_val),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    stats = evaluate(model, data_loader_val, device)
    record = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_model_key": args.checkpoint_model_key,
        "missing_keys": missing[:50],
        "unexpected_keys": unexpected[:50],
        "num_missing_keys": len(missing),
        "num_unexpected_keys": len(unexpected),
        "eval": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
        "args": {
            "backbone": getattr(merged_args, "backbone", None),
            "timm_adapter": getattr(merged_args, "timm_adapter", None),
            "score_threshold": float(getattr(merged_args, "score_threshold", -1)),
            "split_threshold": float(getattr(merged_args, "split_threshold", -1)),
            "eval_count_mode": getattr(merged_args, "eval_count_mode", None),
            "eval_score_calibration": getattr(merged_args, "eval_score_calibration", None),
        },
    }
    print(json.dumps(record, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
