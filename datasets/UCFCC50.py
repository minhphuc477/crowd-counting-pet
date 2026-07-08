import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as standard_transforms
from torch.utils.data import Dataset

from .QNRF import resize_long_side
from .SHA import (
    IMAGE_EXTENSIONS,
    _parse_patch_size_choices,
    load_points,
    random_crop_with_retries,
    safe_random_scale,
)
from .image_io import load_rgb_image


def _natural_key(path):
    parts = re.split(r'(\d+)', Path(path).stem)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def find_images_dir(data_root):
    root = Path(data_root)
    candidates = (root / 'images', root / 'Images', root)
    for candidate in candidates:
        if candidate.is_dir() and any(
            path.suffix.lower() in IMAGE_EXTENSIONS
            for path in candidate.iterdir()
            if path.is_file()
        ):
            return candidate
    return candidates[0]


def find_annotation_path(image_path, data_root):
    image_path = Path(image_path)
    root = Path(data_root)
    stem = image_path.stem
    names = (
        f'{stem}_ann.mat',
        f'GT_{stem}.mat',
        f'{stem}.mat',
    )
    directories = (
        image_path.parent,
        root,
        root / 'ground_truth',
        root / 'ground-truth',
        root / 'annotations',
        root / 'gt',
    )
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _validate_folds(folds, stems):
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError('UCF-CC-50 fold manifest must contain exactly five folds')
    normalized = []
    for fold_index, fold in enumerate(folds):
        if not isinstance(fold, list):
            raise ValueError(f'UCF-CC-50 fold {fold_index} must be a list')
        normalized.append([Path(str(stem)).stem for stem in fold])

    flattened = [stem for fold in normalized for stem in fold]
    if len(flattened) != len(set(flattened)):
        raise ValueError('UCF-CC-50 fold manifest contains duplicate image IDs')
    expected = set(stems)
    actual = set(flattened)
    if actual != expected:
        missing = sorted(expected - actual, key=_natural_key)
        extra = sorted(actual - expected, key=_natural_key)
        raise ValueError(
            'UCF-CC-50 fold manifest does not match the dataset: '
            f'missing={missing[:5]} extra={extra[:5]}'
        )
    return normalized


def build_folds(stems, seed=42, manifest_path=''):
    stems = sorted((Path(stem).stem for stem in stems), key=_natural_key)
    if len(stems) != 50:
        raise ValueError(
            f'UCF-CC-50 must contain exactly 50 images, found {len(stems)}'
        )

    if manifest_path:
        manifest_path = Path(manifest_path)
        with manifest_path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        folds = payload.get('folds') if isinstance(payload, dict) else payload
        return _validate_folds(folds, stems)

    rng = np.random.RandomState(int(seed))
    shuffled = np.asarray(stems, dtype=object)[rng.permutation(len(stems))]
    folds = [
        [str(stem) for stem in fold.tolist()]
        for fold in np.array_split(shuffled, 5)
    ]
    return _validate_folds(folds, stems)


class UCFCC50(Dataset):
    def __init__(
        self,
        data_root,
        transform=None,
        train=False,
        source_split='train',
        fold=0,
        fold_seed=42,
        fold_manifest='',
        flip=False,
        patch_size=256,
        crop_attempts=1,
        min_crop_points=0,
        eval_max_size=2048,
        no_random_scale=False,
    ):
        if source_split not in ('train', 'test'):
            raise ValueError("source_split must be 'train' or 'test'")
        if not 0 <= int(fold) < 5:
            raise ValueError('ucfcc50_fold must be in [0, 4]')

        self.root_path = str(data_root)
        self.images_dir = find_images_dir(data_root)
        image_paths = sorted(
            (
                path for path in self.images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=_natural_key,
        )
        if len(image_paths) != 50:
            raise FileNotFoundError(
                f'Expected 50 UCF-CC-50 images in {self.images_dir}, '
                f'found {len(image_paths)}'
            )

        stems = [path.stem for path in image_paths]
        self.folds = build_folds(stems, seed=fold_seed, manifest_path=fold_manifest)
        self.fold = int(fold)
        self.fold_seed = int(fold_seed)
        self.test_stems = tuple(self.folds[self.fold])
        test_ids = set(self.test_stems)
        if source_split == 'test':
            selected = [path for path in image_paths if path.stem in test_ids]
        else:
            selected = [path for path in image_paths if path.stem not in test_ids]

        self.gt_list = {}
        missing = []
        for image_path in selected:
            gt_path = find_annotation_path(image_path, data_root)
            if gt_path is None:
                missing.append(str(image_path))
            else:
                self.gt_list[str(image_path)] = str(gt_path)
        if missing:
            raise FileNotFoundError(
                f'Missing {len(missing)} UCF-CC-50 annotation file(s). '
                f'First image without annotation: {missing[0]}'
            )

        self.img_list = list(self.gt_list)
        self.nSamples = len(self.img_list)
        expected = 40 if source_split == 'train' else 10
        if self.nSamples != expected:
            raise RuntimeError(
                f'UCF-CC-50 fold {self.fold} {source_split} split must have '
                f'{expected} samples, found {self.nSamples}'
            )

        self.transform = transform
        self.train = bool(train)
        self.flip = bool(flip)
        self.patch_size_choices = _parse_patch_size_choices(patch_size)
        self.crop_attempts = max(1, int(crop_attempts))
        self.min_crop_points = max(0, int(min_crop_points))
        self.eval_max_size = int(eval_max_size)
        self.no_random_scale = bool(no_random_scale)

    def __len__(self):
        return self.nSamples

    @staticmethod
    def compute_density(points):
        points_tensor = torch.from_numpy(points.copy())
        if points_tensor.shape[0] > 1:
            distances = torch.cdist(points_tensor, points_tensor, p=2)
            return distances.sort(dim=1)[0][:, 1].mean().reshape(-1)
        return torch.tensor(999.0).reshape(-1)

    def __getitem__(self, index):
        image_path = self.img_list[index]
        gt_path = self.gt_list[image_path]
        image = load_rgb_image(image_path)
        width, height = image.size
        points = load_points(gt_path, image_size=(height, width)).astype(float)
        if points.shape[0] == 0:
            raise ValueError(f'Annotation file has zero valid points: {gt_path}')

        if self.eval_max_size > 0:
            image, points = resize_long_side(
                image,
                points,
                self.eval_max_size,
            )
        if self.transform is not None:
            image = self.transform(image)
        image = torch.as_tensor(image, dtype=torch.float32)

        if self.train:
            patch_size = random.choice(self.patch_size_choices)
            if not self.no_random_scale:
                image, points = safe_random_scale(image, points, patch_size)
            image, points = random_crop_with_retries(
                image,
                points,
                patch_size=patch_size,
                attempts=self.crop_attempts,
                min_points=self.min_crop_points,
            )
            if self.flip and random.random() > 0.5:
                image = torch.flip(image, dims=[2])
                points[:, 1] = (image.shape[2] - 1) - points[:, 1]

        target = {
            'points': torch.as_tensor(points, dtype=torch.float32),
            'labels': torch.ones(points.shape[0], dtype=torch.long),
            'ucfcc50_fold': torch.tensor(self.fold, dtype=torch.long),
        }
        if self.train:
            target['density'] = self.compute_density(points)
        else:
            target['image_path'] = image_path
        return image, target


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
    common = {
        'data_root': args.data_path,
        'transform': transform,
        'fold': getattr(args, 'ucfcc50_fold', 0),
        'fold_seed': getattr(args, 'ucfcc50_fold_seed', 42),
        'fold_manifest': getattr(args, 'ucfcc50_fold_manifest', ''),
        'eval_max_size': eval_max_size,
        'no_random_scale': getattr(args, 'no_random_scale', False),
    }
    if image_set == 'train':
        return UCFCC50(
            **common,
            train=True,
            source_split='train',
            flip=True,
            patch_size=getattr(args, 'patch_size_choices', '') or args.patch_size,
            crop_attempts=getattr(args, 'crop_attempts', 1),
            min_crop_points=getattr(args, 'min_crop_points', 0),
        )
    if image_set == 'train_eval':
        return UCFCC50(**common, train=False, source_split='train')
    if image_set == 'val':
        return UCFCC50(**common, train=False, source_split='test')
    raise ValueError(f'Unsupported image_set for UCF-CC-50: {image_set}')
