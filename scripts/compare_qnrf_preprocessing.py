#!/usr/bin/env python3
"""Compare a resized QNRF tree against its original source tree."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.QNRF import find_annotation_path, load_raw_points_xy  # noqa: E402
from scripts.preprocess_qnrf_aligned import orient_points  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def find_split(root: Path, name: str) -> Path:
    for candidate in (root / name, root / name.lower(), root / name.upper()):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"missing {name} directory under {root}")


def point_difference(source_points, processed_points, factor):
    if source_points.shape != processed_points.shape:
        return None
    if source_points.size == 0:
        return 0.0
    expected = source_points.astype(np.float64) / factor
    return float(np.abs(expected - processed_points.astype(np.float64)).max())


def inspect_split(source_root: Path, processed_root: Path, split: str, max_side: int):
    source_dir = find_split(source_root, split)
    processed_dir = find_split(processed_root, split)
    processed_images = sorted(
        path for path in processed_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    rows = {
        "images": len(processed_images),
        "size_matches": 0,
        "swapped_size_matches": 0,
        "other_size_mismatches": 0,
        "annotation_scale_matches": 0,
        "annotation_exif_matches": 0,
        "annotation_scale_mismatches": 0,
        "count_mismatches": 0,
        "suspected_orientation_mismatches": 0,
        "examples": [],
    }

    for processed_image in processed_images:
        source_image = source_dir / processed_image.name
        if not source_image.is_file():
            raise FileNotFoundError(f"missing source image: {source_image}")
        source_gt = find_annotation_path(str(source_image))
        processed_gt = find_annotation_path(str(processed_image))
        if source_gt is None or processed_gt is None:
            raise FileNotFoundError(f"missing annotation for {processed_image.name}")

        with Image.open(source_image) as image:
            source_size = image.size
            exif_orientation = int(image.getexif().get(274, 1))
        with Image.open(processed_image) as image:
            processed_size = image.size

        source_width, source_height = source_size
        factor = max(source_width / max_side, source_height / max_side, 1.0)
        expected_size = (
            max(1, int(source_width / factor)),
            max(1, int(source_height / factor)),
        )
        swapped_size = (expected_size[1], expected_size[0])
        size_matches = processed_size == expected_size
        swapped_matches = (
            expected_size[0] != expected_size[1] and processed_size == swapped_size
        )
        rows["size_matches"] += int(size_matches)
        rows["swapped_size_matches"] += int(swapped_matches)
        rows["other_size_mismatches"] += int(not size_matches and not swapped_matches)

        source_points = load_raw_points_xy(source_gt)
        processed_points = load_raw_points_xy(processed_gt)
        if source_points.shape[0] != processed_points.shape[0]:
            rows["count_mismatches"] += 1
            max_difference = None
            annotation_matches = False
        else:
            max_difference = point_difference(source_points, processed_points, factor)
            annotation_matches = max_difference is not None and max_difference <= 1e-3
        oriented_source_points = orient_points(
            source_points, exif_orientation, source_size[0], source_size[1]
        )
        exif_max_difference = point_difference(
            oriented_source_points, processed_points, factor
        )
        annotation_exif_matches = (
            exif_max_difference is not None and exif_max_difference <= 1e-3
        )
        rows["annotation_scale_matches"] += int(annotation_matches)
        rows["annotation_exif_matches"] += int(annotation_exif_matches)
        rows["annotation_scale_mismatches"] += int(
            not annotation_matches and not annotation_exif_matches
        )

        suspected = swapped_matches and annotation_matches
        rows["suspected_orientation_mismatches"] += int(suspected)
        if (suspected or not size_matches or not annotation_matches) and len(rows["examples"]) < 25:
            rows["examples"].append(
                {
                    "image": processed_image.name,
                    "source_size": list(source_size),
                    "processed_size": list(processed_size),
                    "expected_size": list(expected_size),
                    "source_exif_orientation": exif_orientation,
                    "annotation_max_abs_difference": max_difference,
                    "annotation_exif_max_abs_difference": exif_max_difference,
                    "annotation_exif_matches": annotation_exif_matches,
                    "suspected_orientation_mismatch": suspected,
                }
            )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare resized QNRF images/annotations with the original dataset."
    )
    parser.add_argument("--source_data_path", required=True, type=Path)
    parser.add_argument("--processed_data_path", required=True, type=Path)
    parser.add_argument("--max_side", default=1536, type=int)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.max_side <= 0:
        raise ValueError("--max_side must be positive")

    report = {
        "source": str(args.source_data_path.expanduser().resolve()),
        "processed": str(args.processed_data_path.expanduser().resolve()),
        "max_side": args.max_side,
        "splits": {
            split: inspect_split(
                args.source_data_path.expanduser().resolve(),
                args.processed_data_path.expanduser().resolve(),
                split,
                args.max_side,
            )
            for split in ("Train", "Test")
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    suspected = sum(
        split["suspected_orientation_mismatches"]
        for split in report["splits"].values()
    )
    if suspected:
        raise SystemExit(
            f"Detected {suspected} image/annotation pairs with swapped processed dimensions "
            "but unrotated scaled annotations. Rebuild the processed dataset before training."
        )
    print("QNRF preprocessing comparison passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
