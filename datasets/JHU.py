import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as standard_transforms
from torch.utils.data import Dataset

from .QNRF import resize_long_side
from .SHA import (
    IMAGE_EXTENSIONS,
    _parse_patch_size_choices,
    random_crop_with_retries,
    safe_random_scale,
)
from .image_io import load_rgb_image


class JHUCrowd(Dataset):
    """JHU-Crowd++ point loader for the official train/val/test layout."""

    def __init__(
        self,
        data_root,
        split='train',
        transform=None,
        train=False,
        flip=False,
        patch_size=256,
        crop_attempts=1,
        min_crop_points=0,
        eval_max_size=2048,
    ):
        self.root_path = Path(data_root)
        self.split = split
        self.split_dir = self.root_path / split
        self.image_dir = self.split_dir / 'images'
        self.gt_dir = self.split_dir / 'gt'
        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f'Could not find JHU-Crowd++ image directory: {self.image_dir}'
            )
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(
                f'Could not find JHU-Crowd++ annotation directory: {self.gt_dir}'
            )

        self.samples = []
        missing = []
        for image_path in sorted(self.image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            annotation_path = self.gt_dir / f'{image_path.stem}.txt'
            if not annotation_path.is_file():
                missing.append(annotation_path)
            self.samples.append((image_path, annotation_path))
        if not self.samples:
            raise FileNotFoundError(
                f'No JHU-Crowd++ images found in {self.image_dir}'
            )
        if missing:
            raise FileNotFoundError(
                f'Missing {len(missing)} JHU-Crowd++ annotation file(s) '
                f'for split={split}. First missing file: {missing[0]}'
            )

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = patch_size
        self.patch_size_choices = _parse_patch_size_choices(patch_size)
        self.crop_attempts = max(1, int(crop_attempts))
        self.min_crop_points = max(0, int(min_crop_points))
        self.eval_max_size = int(eval_max_size)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def compute_density(points):
        points_tensor = torch.from_numpy(points.copy())
        if points_tensor.shape[0] > 1:
            distances = torch.cdist(points_tensor, points_tensor, p=2)
            return distances.sort(dim=1)[0][:, 1].mean().reshape(-1)
        return torch.tensor(999.0).reshape(-1)

    def __getitem__(self, index):
        image_path, annotation_path = self.samples[index]
        image = load_rgb_image(str(image_path))
        points, sigma = load_jhu_annotation(
            annotation_path,
            image_size=(image.height, image.width),
        )

        # Official PET preprocesses both JHU train and evaluation splits to
        # the same long-side limit before sampling training crops.
        if self.eval_max_size > 0:
            old_width, old_height = image.size
            image, points = resize_long_side(
                image,
                points,
                self.eval_max_size,
            )
            scale_x = image.width / float(old_width)
            scale_y = image.height / float(old_height)
            sigma = sigma * np.sqrt(scale_x * scale_y)

        if self.transform is not None:
            image = self.transform(image)
        image = torch.as_tensor(image, dtype=torch.float32)

        if self.train:
            patch_size = random.choice(self.patch_size_choices)
            records = np.concatenate([points, sigma], axis=1)
            image, records = safe_random_scale(
                image,
                records,
                patch_size,
            )
            image, records = random_crop_with_retries(
                image,
                records,
                patch_size=patch_size,
                attempts=self.crop_attempts,
                min_points=self.min_crop_points,
            )
            points = records[:, :2]
            sigma = records[:, 2:]

        if self.train and self.flip and random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            if points.shape[0] > 0:
                points[:, 1] = (image.shape[2] - 1) - points[:, 1]

        target = {
            'points': torch.as_tensor(points, dtype=torch.float32),
            'labels': torch.ones(points.shape[0], dtype=torch.long),
            'sigma': torch.as_tensor(sigma, dtype=torch.float32),
        }
        if self.train:
            target['density'] = self.compute_density(points)
        else:
            target['image_path'] = str(image_path)
            target['image_id'] = image_path.stem
            target['sigma_source'] = 'jhu_approximate_head_box'
        return image, target


def load_jhu_annotation(path, image_size=None):
    """Read x, y, width, height, occlusion, blur JHU annotations."""
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(
            encoding='utf-8',
            errors='ignore',
        ).splitlines(),
        start=1,
    ):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError(
                f'Invalid JHU annotation line {line_number} in {path}: '
                f'expected 6 fields, got {len(fields)}'
            )
        try:
            rows.append([float(value) for value in fields])
        except ValueError as exc:
            raise ValueError(
                f'Invalid numeric value on JHU annotation line '
                f'{line_number} in {path}'
            ) from exc

    if not rows:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty.copy()

    annotations = np.asarray(rows, dtype=np.float32)
    points = annotations[:, :2][:, ::-1].copy()
    width = annotations[:, 2].clip(min=1.0)
    height = annotations[:, 3].clip(min=1.0)
    box_scale = np.sqrt(width * height)
    sigma = np.stack(
        [
            np.maximum(0.5 * box_scale, 1.0),
            np.maximum(box_scale, 1.0),
        ],
        axis=1,
    ).astype(np.float32)

    if image_size is not None:
        image_height, image_width = image_size
        keep = (
            (points[:, 0] >= 0)
            & (points[:, 0] < image_height)
            & (points[:, 1] >= 0)
            & (points[:, 1] < image_width)
        )
        points = points[keep]
        sigma = sigma[keep]
    return points.astype(np.float32, copy=False), sigma


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    eval_max_size = int(getattr(args, 'eval_max_size', -1))
    if eval_max_size < 0:
        eval_max_size = 2048
    if image_set == 'train':
        return JHUCrowd(
            args.data_path,
            split='train',
            train=True,
            transform=transform,
            flip=True,
            patch_size=getattr(args, 'patch_size_choices', '') or args.patch_size,
            crop_attempts=getattr(args, 'crop_attempts', 1),
            min_crop_points=getattr(args, 'min_crop_points', 0),
            eval_max_size=eval_max_size,
        )
    if image_set == 'train_eval':
        return JHUCrowd(
            args.data_path,
            split='train',
            train=False,
            transform=transform,
            eval_max_size=eval_max_size,
        )
    if image_set == 'val':
        return JHUCrowd(
            args.data_path,
            split=getattr(args, 'jhu_eval_split', 'val'),
            train=False,
            transform=transform,
            eval_max_size=eval_max_size,
        )
    raise ValueError(f'Unsupported image_set for JHU-Crowd++: {image_set}')
