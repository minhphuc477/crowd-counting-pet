#!/usr/bin/env python3
"""Audit PET checkpoints and evaluation JSON for report-invalid settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


BENCHMARK_ONLY = {
    "SHA", "SHB", "QNRF",
    "UCFCC50", "UCF_CC_50", "UCF-CC-50",
}


def _value(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _finding(severity, path, message):
    return {
        "severity": severity,
        "path": str(path),
        "message": message,
    }


def audit_record(record, path):
    findings = []
    dataset = _value(record, "dataset_file", "")
    protocol = _value(record, "validation_protocol", "")
    eval_image_set = _value(record, "eval_image_set", "")

    if int(_value(record, "eval_tile_min_gt", 0) or 0) > 0:
        findings.append(_finding(
            "ERROR",
            path,
            "ground-truth count controlled the tiled inference path",
        ))
    if protocol == "benchmark_test" and dataset in BENCHMARK_ONLY:
        findings.append(_finding(
            "ERROR",
            path,
            "checkpoint selection used the benchmark test split",
        ))
    if (
        path.name in {"sweep_results.json", "best_thresholds.json"}
        and eval_image_set == "val"
        and dataset in BENCHMARK_ONLY
    ):
        findings.append(_finding(
            "ERROR",
            path,
            "threshold selection used the benchmark test split",
        ))
    if bool(_value(record, "train_count_head_only", False)):
        if not bool(_value(record, "freeze_bn", False)):
            findings.append(_finding(
                "ERROR",
                path,
                "count-head-only training left BatchNorm buffers mutable",
            ))
        if _value(record, "eval_count_mode", "threshold") == "threshold":
            findings.append(_finding(
                "WARNING",
                path,
                "threshold inference does not use the scalar count head",
            ))
    if bool(_value(record, "no_pretrained_backbone", False)):
        findings.append(_finding(
            "WARNING",
            path,
            "backbone was randomly initialized",
        ))
    return findings


def audit_path(path):
    if path.suffix.lower() == ".pth":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        args = checkpoint.get("args", {})
        findings = audit_record(args, path)
        metrics = checkpoint.get("checkpoint_eval_metrics")
        if isinstance(metrics, dict):
            merged = dict(metrics)
            merged.setdefault("dataset_file", _value(args, "dataset_file", ""))
            merged.setdefault(
                "validation_protocol",
                _value(args, "validation_protocol", ""),
            )
            findings.extend(audit_record(merged, path))
        return findings

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        findings = []
        for record in payload:
            if isinstance(record, dict):
                findings.extend(audit_record(record, path))
        return findings
    if isinstance(payload, dict):
        return audit_record(payload, path)
    return [_finding("WARNING", path, "unsupported JSON payload")]


def collect_paths(inputs):
    paths = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.pth")))
            paths.extend(sorted(path.rglob("*.json")))
        elif path.is_file() and path.suffix.lower() in {".pth", ".json"}:
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    return list(dict.fromkeys(paths))


def main():
    parser = argparse.ArgumentParser(
        description="Audit actual checkpoints/results for evaluation leakage.",
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    paths = collect_paths(args.paths)
    findings = []
    failures = []
    for path in paths:
        try:
            findings.extend(audit_path(path))
        except Exception as exc:
            failures.append(_finding("WARNING", path, f"audit failed: {exc}"))
    findings.extend(failures)

    report = {
        "files_checked": len(paths),
        "errors": sum(item["severity"] == "ERROR" for item in findings),
        "warnings": sum(item["severity"] == "WARNING" for item in findings),
        "findings": findings,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(2 if report["errors"] else 0)


if __name__ == "__main__":
    main()
