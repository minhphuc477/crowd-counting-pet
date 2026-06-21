#!/usr/bin/env python3
"""Sweep PET inference thresholds for an existing checkpoint.

This is intentionally an evaluation-only tool. Including the current paper
thresholds in the grid gives a validation-set result that cannot be worse than
the baseline on the same checkpoint and split.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_THRESHOLDS = (0.35, 0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6, 0.65)
DEFAULT_SPLIT_THRESHOLDS = (0.5,)


def _unique_sorted(values: list[float]) -> list[float]:
    return sorted({round(float(v), 6) for v in values})


def _format_loc_metrics(record: dict, prefix: str = " ") -> str:
    if "loc_f1_large" not in record or "loc_f1_small" not in record:
        return ""
    return (
        f"{prefix}loc_sigma_l(F1/Prec/Rec)="
        f"{float(record['loc_f1_large']):.4f}/"
        f"{float(record.get('loc_prec_large', 0.0)):.4f}/"
        f"{float(record.get('loc_rec_large', 0.0)):.4f} "
        f"loc_sigma_s(F1/Prec/Rec)="
        f"{float(record['loc_f1_small']):.4f}/"
        f"{float(record.get('loc_prec_small', 0.0)):.4f}/"
        f"{float(record.get('loc_rec_small', 0.0)):.4f}"
    )


def _load_checkpoint_args(checkpoint_path: Path):
    try:
        import torch
    except Exception:
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args")
    if args is None:
        return None
    return args


def _checkpoint_arg(checkpoint_args, key: str):
    if checkpoint_args is None:
        return None
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(key)
    return getattr(checkpoint_args, key, None)


def resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint_path = Path(args.resume)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint_args = _load_checkpoint_args(checkpoint_path)
    if not args.dataset_file:
        args.dataset_file = _checkpoint_arg(checkpoint_args, "dataset_file") or "SHA"
    if not args.data_path:
        args.data_path = _checkpoint_arg(checkpoint_args, "data_path") or ""
    if not args.backbone:
        args.backbone = _checkpoint_arg(checkpoint_args, "backbone") or "vgg16_bn"
    if not args.output_dir:
        args.output_dir = str(checkpoint_path.resolve().parent / "threshold_sweep")
    return args


def run_eval(
    args: argparse.Namespace,
    score_threshold: float,
    split_threshold: float,
    query_prune_threshold: float,
    eval_nms_radius: float,
    eval_branch_gate: str,
    eval_soft_split_gate: str,
    eval_count_mode: str,
    eval_count_source: str,
    eval_count_head_min_score: float,
    eval_score_calibration: str,
    eval_score_calibration_strength: float,
    eval_score_calibration_start_epoch: int,
    eval_score_calibration_min_bias: float,
    eval_score_calibration_max_bias: float,
    eval_score_calibration_count_blend: float,
    eval_score_calibration_count_ratio_min: float,
    eval_score_calibration_count_ratio_max: float,
    eval_foreground_gate: str | None,
    eval_foreground_gate_mode: str | None,
    eval_foreground_gate_strength: float | None,
    run_dir: Path,
) -> dict:
    tag = (
        f"score_{score_threshold:.6g}_split_{split_threshold:.6g}_prune_{query_prune_threshold:.6g}_"
        f"nms_{eval_nms_radius:.6g}_gate_{eval_branch_gate}_soft_{eval_soft_split_gate}_"
        f"count_{eval_count_mode}_source_{eval_count_source}_min_{eval_count_head_min_score:.6g}_cal_{eval_score_calibration}"
    ).replace(".", "p")
    if eval_foreground_gate is not None:
        tag += f"_fg_{eval_foreground_gate}"
    if eval_foreground_gate_mode is not None:
        tag += f"_fgm_{eval_foreground_gate_mode}"
    if eval_foreground_gate_strength is not None:
        tag += f"_fgs_{eval_foreground_gate_strength:.6g}".replace(".", "p")
    results_file = run_dir / f"{tag}.json"
    log_file = run_dir / f"{tag}.log"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "eval.py"),
        "--backbone",
        args.backbone,
        "--dataset_file",
        args.dataset_file,
        "--resume",
        args.resume,
        "--device",
        args.device,
        "--num_workers",
        str(args.num_workers),
        "--results_file",
        str(results_file),
        "--override_score_threshold",
        str(score_threshold),
        "--override_split_threshold",
        str(split_threshold),
        "--override_query_prune_threshold",
        str(query_prune_threshold),
        "--eval_nms_radius",
        str(eval_nms_radius),
        "--eval_branch_gate",
        eval_branch_gate,
        "--eval_soft_split_gate",
        eval_soft_split_gate,
        "--eval_count_mode",
        eval_count_mode,
    ]
    if eval_count_source != "checkpoint":
        cmd.extend(["--eval_count_source", eval_count_source])
    cmd.extend([
        "--eval_count_head_min_score",
        str(eval_count_head_min_score),
        "--eval_score_calibration",
        eval_score_calibration,
        "--eval_score_calibration_strength",
        str(eval_score_calibration_strength),
        "--eval_score_calibration_start_epoch",
        str(eval_score_calibration_start_epoch),
        "--eval_score_calibration_min_bias",
        str(eval_score_calibration_min_bias),
        "--eval_score_calibration_max_bias",
        str(eval_score_calibration_max_bias),
        "--eval_score_calibration_count_blend",
        str(eval_score_calibration_count_blend),
        "--eval_score_calibration_count_ratio_min",
        str(eval_score_calibration_count_ratio_min),
        "--eval_score_calibration_count_ratio_max",
        str(eval_score_calibration_count_ratio_max),
        "--eval_protocol",
        args.eval_protocol,
    ])
    if args.data_path:
        cmd.extend(["--data_path", args.data_path])
    if args.tta_flip:
        cmd.append("--tta_flip")
    if args.tta_scales:
        cmd.extend(["--tta_scales", args.tta_scales])
    if args.no_localization_metrics:
        cmd.append("--no_localization_metrics")
    cmd.extend([
        "--localization_large_threshold",
        str(args.localization_large_threshold),
        "--localization_small_threshold",
        str(args.localization_small_threshold),
    ])
    if eval_foreground_gate is not None:
        cmd.extend(["--eval_foreground_gate", eval_foreground_gate])
    if eval_foreground_gate_mode is not None:
        cmd.extend(["--eval_foreground_gate_mode", eval_foreground_gate_mode])
    if eval_foreground_gate_strength is not None:
        cmd.extend(["--eval_foreground_gate_strength", str(eval_foreground_gate_strength)])

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout if args.timeout > 0 else None,
    )
    log_file.write_text(
        "command: " + " ".join(cmd) + "\n\n" + proc.stdout,
        encoding="utf-8",
    )

    record = {
        "score_threshold": float(score_threshold),
        "split_threshold": float(split_threshold),
        "query_prune_threshold": float(query_prune_threshold),
        "eval_nms_radius": float(eval_nms_radius),
        "eval_branch_gate": eval_branch_gate,
        "eval_soft_split_gate": eval_soft_split_gate,
        "eval_count_mode": eval_count_mode,
        "eval_count_source": eval_count_source,
        "eval_count_head_min_score": float(eval_count_head_min_score),
        "eval_score_calibration": eval_score_calibration,
        "eval_score_calibration_strength": float(eval_score_calibration_strength),
        "eval_score_calibration_start_epoch": int(eval_score_calibration_start_epoch),
        "eval_score_calibration_min_bias": float(eval_score_calibration_min_bias),
        "eval_score_calibration_max_bias": float(eval_score_calibration_max_bias),
        "eval_score_calibration_count_blend": float(eval_score_calibration_count_blend),
        "eval_score_calibration_count_ratio_min": float(eval_score_calibration_count_ratio_min),
        "eval_score_calibration_count_ratio_max": float(eval_score_calibration_count_ratio_max),
        "eval_foreground_gate": eval_foreground_gate if eval_foreground_gate is not None else "checkpoint",
        "eval_foreground_gate_mode": (
            eval_foreground_gate_mode if eval_foreground_gate_mode is not None else "checkpoint"
        ),
        "eval_foreground_gate_strength": (
            float(eval_foreground_gate_strength) if eval_foreground_gate_strength is not None else "checkpoint"
        ),
        "eval_protocol": args.eval_protocol,
        "tta_flip": bool(args.tta_flip),
        "tta_scales": args.tta_scales,
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.time() - started),
        "results_file": str(results_file),
        "log_file": str(log_file),
    }
    if proc.returncode == 0 and results_file.is_file():
        metrics = json.loads(results_file.read_text(encoding="utf-8"))
        record.update(metrics)
        record["ok"] = True
    else:
        record["ok"] = False
        record["error"] = f"eval.py failed with return code {proc.returncode}"
    return record


def write_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep_results.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "ok",
        "score_threshold",
        "split_threshold",
        "query_prune_threshold",
        "eval_nms_radius",
        "eval_branch_gate",
        "eval_soft_split_gate",
        "eval_count_mode",
        "eval_count_source",
        "eval_count_head_min_score",
        "eval_score_calibration",
        "eval_score_calibration_strength",
        "eval_score_calibration_start_epoch",
        "eval_score_calibration_min_bias",
        "eval_score_calibration_max_bias",
        "eval_score_calibration_count_blend",
        "eval_score_calibration_count_ratio_min",
        "eval_score_calibration_count_ratio_max",
        "eval_foreground_gate",
        "eval_foreground_gate_mode",
        "eval_foreground_gate_strength",
        "eval_protocol",
        "tta_flip",
        "tta_scales",
        "eval_mae",
        "eval_mse",
        "pred_cnt",
        "gt_cnt",
        "localization_metrics",
        "localization_large_threshold",
        "localization_small_threshold",
        "loc_threshold_large",
        "loc_f1_large",
        "loc_prec_large",
        "loc_rec_large",
        "loc_tp_large",
        "loc_fp_large",
        "loc_fn_large",
        "loc_threshold_sigma_l",
        "loc_f1_sigma_l",
        "loc_prec_sigma_l",
        "loc_rec_sigma_l",
        "loc_tp_sigma_l",
        "loc_fp_sigma_l",
        "loc_fn_sigma_l",
        "loc_threshold_small",
        "loc_f1_small",
        "loc_prec_small",
        "loc_rec_small",
        "loc_tp_small",
        "loc_fp_small",
        "loc_fn_small",
        "loc_threshold_sigma_s",
        "loc_f1_sigma_s",
        "loc_prec_sigma_s",
        "loc_rec_sigma_s",
        "loc_tp_sigma_s",
        "loc_fp_sigma_s",
        "loc_fn_sigma_s",
        "epoch",
        "returncode",
        "elapsed_sec",
        "results_file",
        "log_file",
    ]
    with (output_dir / "sweep_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    ok_records = [record for record in records if record.get("ok") and "eval_mae" in record]
    if ok_records:
        best = min(ok_records, key=lambda item: item["eval_mae"])
        (output_dir / "best_thresholds.json").write_text(
            json.dumps(best, indent=2) + "\n",
            encoding="utf-8",
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Sweep PET eval thresholds")
    parser.add_argument("--resume", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--dataset_file", default="", help="Dataset name; default reads checkpoint args")
    parser.add_argument("--data_path", default="", help="Dataset root; default reads checkpoint args")
    parser.add_argument("--backbone", default="", help="Backbone; default reads checkpoint args")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--timeout", default=1800, type=int, help="Per-eval timeout in seconds; 0 disables")
    parser.add_argument("--output_dir", default="", help="Where to save sweep logs/results")
    parser.add_argument("--tta_flip", action="store_true", help="average original and horizontal-flip counts")
    parser.add_argument("--tta_scales", default="1.0", help="comma-separated eval scales passed to eval.py")
    parser.add_argument("--no_localization_metrics", action="store_true", help="disable localization metrics in eval.py")
    parser.add_argument("--localization_large_threshold", default=8.0, type=float)
    parser.add_argument("--localization_small_threshold", default=4.0, type=float)
    parser.add_argument(
        "--eval_nms_radii",
        nargs="+",
        type=float,
        default=[4.0],
        help="eval-only point NMS radii in pixels; 0 disables",
    )
    parser.add_argument(
        "--eval_branch_gates",
        nargs="+",
        choices=("none", "query", "pred"),
        default=["pred"],
        help="eval-only sparse/dense split-ownership gates",
    )
    parser.add_argument(
        "--eval_soft_split_gates",
        nargs="+",
        choices=("none", "query", "pred"),
        default=["pred"],
        help="eval-only soft split responsibility gates multiplied into scores",
    )
    parser.add_argument(
        "--eval_count_modes",
        nargs="+",
        choices=("threshold", "count_head_topk"),
        default=["threshold"],
        help="count selection mode passed to eval.py",
    )
    parser.add_argument(
        "--eval_count_sources",
        nargs="+",
        choices=("checkpoint", "pet", "zip"),
        default=["checkpoint"],
        help="count source passed to eval.py; checkpoint preserves the checkpoint args",
    )
    parser.add_argument(
        "--eval_count_head_min_scores",
        nargs="+",
        type=float,
        default=[0.5],
        help="candidate score floors for count_head_topk",
    )
    parser.add_argument(
        "--eval_score_calibrations",
        nargs="+",
        choices=("none", "count_head_bias"),
        default=["none"],
        help="eval-only score calibration modes passed to eval.py",
    )
    parser.add_argument("--eval_score_calibration_strength", default=1.0, type=float)
    parser.add_argument("--eval_score_calibration_start_epoch", default=0, type=int)
    parser.add_argument("--eval_score_calibration_min_bias", default=-8.0, type=float)
    parser.add_argument("--eval_score_calibration_max_bias", default=8.0, type=float)
    parser.add_argument("--eval_score_calibration_count_blend", default=1.0, type=float)
    parser.add_argument("--eval_score_calibration_count_ratio_min", default=0.0, type=float)
    parser.add_argument("--eval_score_calibration_count_ratio_max", default=1e6, type=float)
    parser.add_argument(
        "--eval_foreground_gates",
        nargs="+",
        choices=("none", "query", "pred"),
        default=None,
        help="optional foreground-gate overrides; omitted preserves checkpoint args",
    )
    parser.add_argument(
        "--eval_foreground_gate_modes",
        nargs="+",
        choices=("suppress", "logit_add"),
        default=None,
        help="optional foreground-gate mode overrides; omitted preserves checkpoint args",
    )
    parser.add_argument(
        "--eval_foreground_gate_strengths",
        nargs="+",
        type=float,
        default=None,
        help="optional foreground-gate strength overrides; omitted preserves checkpoint args",
    )
    parser.add_argument(
        "--eval_protocol",
        default="pet",
        choices=("pet", "crowd_no_overlap"),
        help="evaluation protocol passed to eval.py",
    )
    parser.add_argument(
        "--score_thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_SCORE_THRESHOLDS),
    )
    parser.add_argument(
        "--split_thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_SPLIT_THRESHOLDS),
    )
    parser.add_argument(
        "--query_prune_thresholds",
        nargs="+",
        type=float,
        default=[0.5],
        help="PET decoder-window pruning thresholds; original PET uses 0.5",
    )
    return parser.parse_args()


def main() -> int:
    args = resolve_runtime_args(get_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = _unique_sorted(args.score_thresholds)
    splits = _unique_sorted(args.split_thresholds)
    query_prunes = _unique_sorted(args.query_prune_thresholds)
    score_prune_pairs = [
        (score_threshold, query_prune_threshold)
        for query_prune_threshold in query_prunes
        for score_threshold in scores
    ]
    radii = _unique_sorted(args.eval_nms_radii)
    gates = list(dict.fromkeys(args.eval_branch_gates))
    soft_gates = list(dict.fromkeys(args.eval_soft_split_gates))
    count_modes = list(dict.fromkeys(args.eval_count_modes))
    count_sources = list(dict.fromkeys(args.eval_count_sources))
    count_min_scores = _unique_sorted(args.eval_count_head_min_scores)
    score_calibrations = list(dict.fromkeys(args.eval_score_calibrations))
    foreground_gates = (
        [None]
        if args.eval_foreground_gates is None
        else list(dict.fromkeys(args.eval_foreground_gates))
    )
    foreground_modes = (
        [None]
        if args.eval_foreground_gate_modes is None
        else list(dict.fromkeys(args.eval_foreground_gate_modes))
    )
    foreground_strengths = (
        [None]
        if args.eval_foreground_gate_strengths is None
        else _unique_sorted(args.eval_foreground_gate_strengths)
    )
    records = []
    total = (
        len(score_prune_pairs) * len(splits) * len(radii) * len(gates) * len(soft_gates)
        * len(count_modes) * len(count_sources) * len(count_min_scores) * len(score_calibrations)
        * len(foreground_gates) * len(foreground_modes) * len(foreground_strengths)
    )
    index = 0
    for split_threshold in splits:
        for eval_branch_gate in gates:
            for eval_soft_split_gate in soft_gates:
                for eval_count_mode in count_modes:
                    for eval_count_source in count_sources:
                        for eval_count_head_min_score in count_min_scores:
                            for eval_score_calibration in score_calibrations:
                                for eval_foreground_gate in foreground_gates:
                                    for eval_foreground_gate_mode in foreground_modes:
                                        for eval_foreground_gate_strength in foreground_strengths:
                                            for eval_nms_radius in radii:
                                                for score_threshold, query_prune_threshold in score_prune_pairs:
                                                    index += 1
                                                    fg_gate_text = (
                                                        eval_foreground_gate
                                                        if eval_foreground_gate is not None
                                                        else "checkpoint"
                                                    )
                                                    fg_strength_text = (
                                                        eval_foreground_gate_strength
                                                        if eval_foreground_gate_strength is not None
                                                        else "checkpoint"
                                                    )
                                                    fg_mode_text = (
                                                        eval_foreground_gate_mode
                                                        if eval_foreground_gate_mode is not None
                                                        else "checkpoint"
                                                    )
                                                    print(
                                                        f"[{index}/{total}] score_threshold={score_threshold} "
                                                        f"split_threshold={split_threshold} "
                                                        f"query_prune_threshold={query_prune_threshold} "
                                                        f"eval_nms_radius={eval_nms_radius} "
                                                        f"eval_branch_gate={eval_branch_gate} "
                                                        f"eval_soft_split_gate={eval_soft_split_gate} "
                                                        f"eval_count_mode={eval_count_mode} "
                                                        f"eval_count_source={eval_count_source} "
                                                        f"eval_count_head_min_score={eval_count_head_min_score} "
                                                        f"eval_score_calibration={eval_score_calibration} "
                                                        f"eval_score_calibration_count_blend={args.eval_score_calibration_count_blend} "
                                                        f"eval_score_calibration_count_ratio="
                                                        f"{args.eval_score_calibration_count_ratio_min}:"
                                                        f"{args.eval_score_calibration_count_ratio_max} "
                                                        f"eval_foreground_gate={fg_gate_text} "
                                                        f"eval_foreground_gate_mode={fg_mode_text} "
                                                        f"eval_foreground_gate_strength={fg_strength_text}"
                                                    )
                                                    record = run_eval(
                                                        args,
                                                        score_threshold,
                                                        split_threshold,
                                                        query_prune_threshold,
                                                        eval_nms_radius,
                                                        eval_branch_gate,
                                                        eval_soft_split_gate,
                                                        eval_count_mode,
                                                        eval_count_source,
                                                        eval_count_head_min_score,
                                                        eval_score_calibration,
                                                        args.eval_score_calibration_strength,
                                                        args.eval_score_calibration_start_epoch,
                                                        args.eval_score_calibration_min_bias,
                                                        args.eval_score_calibration_max_bias,
                                                        args.eval_score_calibration_count_blend,
                                                        args.eval_score_calibration_count_ratio_min,
                                                        args.eval_score_calibration_count_ratio_max,
                                                        eval_foreground_gate,
                                                        eval_foreground_gate_mode,
                                                        eval_foreground_gate_strength,
                                                        output_dir,
                                                    )
                                                    records.append(record)
                                                    if record.get("ok"):
                                                        pred_cnt = record.get("pred_cnt")
                                                        gt_cnt = record.get("gt_cnt")
                                                        if pred_cnt is not None and gt_cnt is not None:
                                                            loc_text = _format_loc_metrics(record)
                                                            print(
                                                                f"  mae={record['eval_mae']:.4f} "
                                                                f"mse={record['eval_mse']:.4f} "
                                                                f"pred={float(pred_cnt):.4f} "
                                                                f"gt={float(gt_cnt):.4f}"
                                                                f"{loc_text}"
                                                            )
                                                        else:
                                                            print(f"  mae={record['eval_mae']:.4f} mse={record['eval_mse']:.4f}")
                                                    else:
                                                        print(f"  failed; see {record['log_file']}")
                                                    write_outputs(records, output_dir)

    ok_records = [record for record in records if record.get("ok") and "eval_mae" in record]
    if not ok_records:
        print(f"No successful evaluations. Logs are in {output_dir}")
        return 1

    best = min(ok_records, key=lambda item: item["eval_mae"])
    print(
        "Best: "
        f"mae={best['eval_mae']:.4f} mse={best['eval_mse']:.4f} "
        f"score_threshold={best['score_threshold']} "
        f"split_threshold={best['split_threshold']} "
        f"query_prune_threshold={best.get('query_prune_threshold', 0.5)} "
        f"eval_nms_radius={best['eval_nms_radius']} "
        f"eval_branch_gate={best['eval_branch_gate']} "
        f"eval_soft_split_gate={best['eval_soft_split_gate']} "
        f"eval_count_source={best.get('eval_count_source', 'checkpoint')} "
        f"eval_score_calibration={best.get('eval_score_calibration', 'none')} "
        f"{_format_loc_metrics(best, prefix='')}"
    )
    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
