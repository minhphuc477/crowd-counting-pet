#!/usr/bin/env python3
"""Inspect QNRF image metadata and annotation coordinate extents."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.QNRF import find_annotation_path, load_raw_points_xy  # noqa: E402
from scripts.preprocess_qnrf_aligned import (  # noqa: E402
    normalize_orientation,
    orient_points,
)


def find_split(root: Path, name: str) -> Path:
    for candidate in (root / name, root / name.lower(), root / name.upper()):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"missing {name} directory under {root}")


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    return str(value)


def inspect_image(root: Path, split: str, image_name: str):
    split_dir = find_split(root, split)
    image_path = split_dir / image_name
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    annotation_path = find_annotation_path(str(image_path))
    if annotation_path is None:
        raise FileNotFoundError(f"missing annotation for {image_path}")

    with Image.open(image_path) as image:
        width, height = image.size
        exif = image.getexif()
        raw_orientation = exif.get(274, 1)
        orientation = normalize_orientation(raw_orientation)
        metadata = {
            ExifTags.TAGS.get(key, str(key)): json_value(value)
            for key, value in exif.items()
            if key in (256, 257, 274, 40962, 40963)
        }

    points = load_raw_points_xy(annotation_path).astype(np.float64)
    oriented = orient_points(points, orientation, width, height).astype(np.float64)
    if orientation in (5, 6, 7, 8):
        oriented_width, oriented_height = height, width
    else:
        oriented_width, oriented_height = width, height

    outside = (
        (oriented[:, 0] < 0)
        | (oriented[:, 0] >= oriented_width)
        | (oriented[:, 1] < 0)
        | (oriented[:, 1] >= oriented_height)
    )
    quantiles = {}
    for axis, values in (("x", oriented[:, 0]), ("y", oriented[:, 1])):
        quantiles[axis] = {
            "min": float(values.min()),
            "p01": float(np.quantile(values, 0.01)),
            "p50": float(np.quantile(values, 0.50)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
        }

    positive_x = oriented[:, 0][oriented[:, 0] >= 0]
    positive_y = oriented[:, 1][oriented[:, 1] >= 0]
    robust_canvas_width = float(np.quantile(positive_x, 0.999)) if positive_x.size else None
    robust_canvas_height = float(np.quantile(positive_y, 0.999)) if positive_y.size else None
    return {
        "image": image_name,
        "image_size": [width, height],
        "oriented_size": [oriented_width, oriented_height],
        "raw_exif_orientation": json_value(raw_orientation),
        "normalized_exif_orientation": orientation,
        "dimension_metadata": metadata,
        "points": int(points.shape[0]),
        "outside_points": int(outside.sum()),
        "outside_fraction": float(outside.mean()),
        "coordinate_quantiles": quantiles,
        "robust_coordinate_to_image_ratio": {
            "x": robust_canvas_width / oriented_width if robust_canvas_width else None,
            "y": robust_canvas_height / oriented_height if robust_canvas_height else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--split", default="Train", choices=("Train", "Test"))
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.data_path.expanduser().resolve()
    report = {
        "root": str(root),
        "split": args.split,
        "samples": [
            inspect_image(root, args.split, image_name)
            for image_name in args.images
        ],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
