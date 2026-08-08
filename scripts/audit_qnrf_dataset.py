#!/usr/bin/env python3
"""Read every QNRF image/annotation pair and validate the training contract."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.QNRF import find_annotation_path, load_raw_points_xy  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
EXPECTED_IMAGES = {"Train": 1201, "Test": 334}


def find_split(root: Path, name: str) -> Path:
    for candidate in (root / name, root / name.lower(), root / name.upper()):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"missing QNRF {name} directory under {root}")


def inspect_split(
    root: Path,
    split_name: str,
    expected_max_side: int,
    max_outside_fraction: float,
) -> dict:
    split_dir = find_split(root, split_name)
    images = sorted(
        path for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) != EXPECTED_IMAGES[split_name]:
        raise ValueError(
            f"{split_name}: expected {EXPECTED_IMAGES[split_name]} images, found {len(images)}"
        )

    missing = []
    total_points = 0
    outside_xy = 0
    outside_if_swapped = 0
    outside_images = 0
    outside_beyond_one_pixel = 0
    maximum_outside_distance = 0.0
    outside_examples = []
    nonfinite = 0
    max_side = 0
    min_side = math.inf
    oversized = []
    zero_count = []

    for image_path in images:
        annotation_path = find_annotation_path(str(image_path))
        if annotation_path is None:
            missing.append(image_path.name)
            continue

        with Image.open(image_path) as image:
            width, height = image.size
        max_side = max(max_side, width, height)
        min_side = min(min_side, width, height)
        if expected_max_side > 0 and max(width, height) > expected_max_side:
            oversized.append(
                {"image": image_path.name, "size": [width, height]}
            )

        points = np.asarray(load_raw_points_xy(annotation_path), dtype=np.float64)
        total_points += int(points.shape[0])
        if points.shape[0] == 0:
            zero_count.append(image_path.name)
            continue

        finite = np.isfinite(points).all(axis=1)
        nonfinite += int((~finite).sum())
        points = points[finite]
        x, y = points[:, 0], points[:, 1]
        outside = (x < 0) | (x >= width) | (y < 0) | (y >= height)
        outside_count = int(outside.sum())
        outside_xy += outside_count
        if outside_count:
            outside_images += 1
            # Distance from the nearest valid zero-based pixel coordinate.
            distance = np.maximum.reduce(
                (
                    np.maximum(-x, 0.0),
                    np.maximum(x - (width - 1), 0.0),
                    np.maximum(-y, 0.0),
                    np.maximum(y - (height - 1), 0.0),
                )
            )
            outside_distance = distance[outside]
            outside_beyond_one_pixel += int((outside_distance > 1.0).sum())
            maximum_outside_distance = max(
                maximum_outside_distance, float(outside_distance.max())
            )
            if len(outside_examples) < 10:
                for point, point_distance in zip(
                    points[outside][: 10 - len(outside_examples)],
                    outside_distance[: 10 - len(outside_examples)],
                ):
                    outside_examples.append(
                        {
                            "image": image_path.name,
                            "size": [width, height],
                            "point_xy": [float(point[0]), float(point[1])],
                            "outside_distance": float(point_distance),
                        }
                    )
        outside_if_swapped += int(
            ((x < 0) | (x >= height) | (y < 0) | (y >= width)).sum()
        )

    if missing:
        raise FileNotFoundError(
            f"{split_name}: {len(missing)} annotations are missing; first={missing[0]}"
        )
    if nonfinite:
        raise ValueError(f"{split_name}: found {nonfinite} non-finite point coordinates")
    outside_fraction = outside_xy / max(total_points - nonfinite, 1)
    if outside_xy and outside_if_swapped < outside_xy:
        raise ValueError(
            f"{split_name}: interpreting annPoints as y/x produces fewer out-of-bounds "
            f"coordinates than x/y ({outside_if_swapped} versus {outside_xy}); inspect "
            "the preprocessing and annotation orientation"
        )
    if outside_fraction > max_outside_fraction:
        raise ValueError(
            f"{split_name}: {outside_xy}/{total_points} annPoints coordinates "
            f"({outside_fraction:.4%}) are outside image bounds, exceeding "
            f"--max_outside_fraction={max_outside_fraction:.4%}; "
            f"outside_if_swapped={outside_if_swapped}, "
            f"maximum_outside_distance={maximum_outside_distance:.3f}"
        )
    if oversized:
        first = oversized[0]
        raise ValueError(
            f"{split_name}: {len(oversized)} images exceed --expected_max_side="
            f"{expected_max_side}; first={first['image']} size={first['size']}"
        )

    return {
        "directory": str(split_dir.resolve()),
        "images": len(images),
        "points": total_points,
        "zero_count_images": zero_count,
        "minimum_image_side": int(min_side),
        "maximum_image_side": int(max_side),
        "outside_xy": outside_xy,
        "outside_xy_fraction": outside_fraction,
        "outside_images": outside_images,
        "outside_beyond_one_pixel": outside_beyond_one_pixel,
        "maximum_outside_distance": maximum_outside_distance,
        "outside_examples": outside_examples,
        "outside_if_swapped": outside_if_swapped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-fast audit of the complete UCF-QNRF Train/Test data contract."
    )
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument(
        "--expected_max_side",
        default=0,
        type=int,
        help="Require every image long side to be at most this value; 0 disables the check.",
    )
    parser.add_argument(
        "--max_outside_fraction",
        default=0.01,
        type=float,
        help=(
            "Maximum fraction of finite annotations allowed outside decoded image bounds. "
            "QNRF/PET preserves these rows for benchmark count parity; default: 0.01."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.max_outside_fraction <= 1.0:
        raise ValueError("--max_outside_fraction must be between 0 and 1")
    root = args.data_path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"QNRF root does not exist: {root}")

    report = {
        "dataset": "UCF-QNRF",
        "root": str(root),
        "expected_max_side": args.expected_max_side,
        "max_outside_fraction": args.max_outside_fraction,
        "splits": {
            split: inspect_split(
                root,
                split,
                args.expected_max_side,
                args.max_outside_fraction,
            )
            for split in ("Train", "Test")
        },
    }
    report["total_images"] = sum(row["images"] for row in report["splits"].values())
    report["total_points"] = sum(row["points"] for row in report["splits"].values())

    rendered = json.dumps(report, indent=2)
    print(rendered)
    for split, row in report["splits"].items():
        if row["outside_xy"]:
            print(
                f"WARNING: {split} preserves {row['outside_xy']} annotations "
                f"({row['outside_xy_fraction']:.4%}) outside decoded image bounds; "
                "see outside_examples in the report.",
                file=sys.stderr,
            )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print("QNRF data audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
