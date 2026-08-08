#!/usr/bin/env python3
"""Build a QNRF long-side dataset with image/point EXIF alignment."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
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
    raise FileNotFoundError(f"missing {name} directory under {root}")


def normalize_orientation(value) -> int:
    if value is None or value == "" or value == b"":
        return 1
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="strict")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 1
    try:
        orientation = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid EXIF orientation: {value!r}") from exc
    # A small number of QNRF JPEGs use the non-standard value 0 to mean
    # unspecified orientation. Decoders treat it as no transform.
    if orientation == 0:
        return 1
    if orientation not in range(1, 9):
        raise ValueError(f"unsupported EXIF orientation: {value!r}")
    return orientation


def orient_image(image: Image.Image, orientation) -> Image.Image:
    orientation = normalize_orientation(orientation)
    transpose = getattr(Image, "Transpose", Image)
    transforms = {
        1: None,
        2: transpose.FLIP_LEFT_RIGHT,
        3: transpose.ROTATE_180,
        4: transpose.FLIP_TOP_BOTTOM,
        5: transpose.TRANSPOSE,
        6: transpose.ROTATE_270,
        7: transpose.TRANSVERSE,
        8: transpose.ROTATE_90,
    }
    operation = transforms[orientation]
    return image.copy() if operation is None else image.transpose(operation)


def orient_points(points: np.ndarray, orientation: int, width: int, height: int):
    orientation = normalize_orientation(orientation)
    if points.size == 0 or orientation == 1:
        return points.copy()
    x = points[:, 0]
    y = points[:, 1]
    transforms = {
        2: (width - 1 - x, y),
        3: (width - 1 - x, height - 1 - y),
        4: (x, height - 1 - y),
        5: (y, x),
        6: (height - 1 - y, x),
        7: (height - 1 - y, width - 1 - x),
        8: (y, width - 1 - x),
    }
    new_x, new_y = transforms[orientation]
    return np.stack((new_x, new_y), axis=1).astype(np.float32, copy=False)


def save_mat_with_points(source_path: str, output_path: Path, points: np.ndarray):
    annotation = {
        key: value
        for key, value in sio.loadmat(source_path).items()
        if not key.startswith("__")
    }
    annotation["annPoints"] = points.astype(np.float32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    sio.savemat(temporary, annotation, appendmat=False)
    os.replace(temporary, output_path)


def preflight_source(source_root: Path):
    summary = {}
    for split in ("Train", "Test"):
        source_dir = find_split(source_root, split)
        images = sorted(
            path for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(images) != EXPECTED_IMAGES[split]:
            raise ValueError(
                f"{split}: expected {EXPECTED_IMAGES[split]} images, found {len(images)}"
            )
        orientations = {}
        for image_path in images:
            annotation_path = find_annotation_path(str(image_path))
            if annotation_path is None:
                raise FileNotFoundError(f"missing annotation for {image_path}")
            with Image.open(image_path) as image:
                raw_orientation = image.getexif().get(274, 1)
                orientation = normalize_orientation(raw_orientation)
                image.verify()
            key = str(orientation)
            orientations[key] = orientations.get(key, 0) + 1
        summary[split] = {
            "images": len(images),
            "normalized_exif_orientations": orientations,
        }
    return summary


def process_split(source_root: Path, output_root: Path, split: str, max_side: int):
    source_dir = find_split(source_root, split)
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) != EXPECTED_IMAGES[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED_IMAGES[split]} images, found {len(images)}"
        )

    exif_oriented = 0
    total_points = 0
    outside_points = 0
    samples = []
    for index, image_path in enumerate(images, start=1):
        annotation_path = find_annotation_path(str(image_path))
        if annotation_path is None:
            raise FileNotFoundError(f"missing annotation for {image_path}")
        points = load_raw_points_xy(annotation_path)
        total_points += int(points.shape[0])

        with Image.open(image_path) as source_image:
            source_width, source_height = source_image.size
            orientation = normalize_orientation(source_image.getexif().get(274, 1))
            points = orient_points(
                points, orientation, source_width, source_height
            )
            image = orient_image(source_image, orientation).convert("RGB")
        exif_oriented += int(orientation != 1)

        width, height = image.size
        factor = max(width / max_side, height / max_side, 1.0)
        new_width = max(1, int(width / factor))
        new_height = max(1, int(height / factor))
        if factor > 1.0:
            image = image.resize((new_width, new_height), Image.BILINEAR)
            points = points / factor

        outside = (
            (points[:, 0] < 0)
            | (points[:, 0] >= new_width)
            | (points[:, 1] < 0)
            | (points[:, 1] >= new_height)
        )
        outside_points += int(outside.sum())

        output_image = output_dir / image_path.name
        output_annotation = output_dir / Path(annotation_path).name
        image.save(output_image, quality=100)
        save_mat_with_points(annotation_path, output_annotation, points)

        if orientation != 1:
            samples.append(
                {
                    "image": image_path.name,
                    "source_size": [source_width, source_height],
                    "orientation": orientation,
                    "output_size": [new_width, new_height],
                    "points": int(points.shape[0]),
                }
            )
        if index % 100 == 0 or index == len(images):
            print(f"{split}: {index}/{len(images)}", flush=True)

    return {
        "images": len(images),
        "points": total_points,
        "exif_oriented_images": exif_oriented,
        "outside_points_after_alignment": outside_points,
        "exif_samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess QNRF while applying identical EXIF geometry to images and points."
    )
    parser.add_argument("--source_data_path", required=True, type=Path)
    parser.add_argument("--output_data_path", required=True, type=Path)
    parser.add_argument("--max_side", default=1536, type=int)
    args = parser.parse_args()
    source_root = args.source_data_path.expanduser().resolve()
    output_root = args.output_data_path.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if output_root.exists():
        raise FileExistsError(
            f"output path already exists: {output_root}; use a new path"
        )
    if args.max_side <= 0:
        raise ValueError("--max_side must be positive")

    preflight = preflight_source(source_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.building-",
            dir=str(output_root.parent),
        )
    )
    try:
        report = {
            "dataset": "UCF-QNRF",
            "method": "exif_aligned_long_side_resize",
            "source": str(source_root),
            "output": str(output_root),
            "max_side": args.max_side,
            "preflight": preflight,
            "splits": {
                split: process_split(
                    source_root, staging_root, split, args.max_side
                )
                for split in ("Train", "Test")
            },
        }
        manifest_path = staging_root / "preprocess_manifest.json"
        manifest_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging_root, output_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    final_manifest_path = output_root / "preprocess_manifest.json"
    print(json.dumps(report, indent=2))
    print(f"Wrote {final_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
