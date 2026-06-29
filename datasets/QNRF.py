import os
import random

import numpy as np
import scipy.io as io
import torch
import torchvision.transforms as standard_transforms
from PIL import Image
from torch.utils.data import Dataset

from .image_io import load_rgb_image
from .SHA import IMAGE_EXTENSIONS, random_crop_with_retries, safe_random_scale
from .SHA import _parse_patch_size_choices


class QNRF(Dataset):
    def __init__(
        self,
        data_root,
        transform=None,
        train=False,
        flip=False,
        patch_size=256,
        crop_attempts=1,
        min_crop_points=0,
        eval_max_size=1536,
    ):
        self.root_path = data_root
        self.split_name = 'Train' if train else 'Test'
        self.split_dir = find_split_dir(data_root, self.split_name)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f'Could not find QNRF split directory: {self.split_dir}')

        img_names = [
            img_name for img_name in os.listdir(self.split_dir)
            if img_name.lower().endswith(IMAGE_EXTENSIONS)
        ]

        self.gt_list = {}
        missing_gt = []
        for img_name in img_names:
            img_path = os.path.join(self.split_dir, img_name)
            gt_path = find_annotation_path(img_path)
            if gt_path is None:
                gt_path = os.path.join(self.split_dir, f'{os.path.splitext(img_name)[0]}_ann.mat')
                missing_gt.append(gt_path)
            self.gt_list[img_path] = gt_path

        self.img_list = sorted(self.gt_list.keys())
        self.nSamples = len(self.img_list)
        if self.nSamples == 0:
            raise FileNotFoundError(
                f'No QNRF images found in {self.split_dir}. '
                f'Expected .jpg/.jpeg/.png/.bmp files.'
            )
        if missing_gt:
            raise FileNotFoundError(
                f'Missing {len(missing_gt)} QNRF annotation file(s) for {self.split_name}. '
                f'First missing file: {missing_gt[0]}'
            )

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = patch_size
        self.patch_size_choices = _parse_patch_size_choices(patch_size)
        self.crop_attempts = max(1, int(crop_attempts))
        self.min_crop_points = max(0, int(min_crop_points))
        self.eval_max_size = int(eval_max_size) if eval_max_size is not None else 1536

    def compute_density(self, points):
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:, 1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]
        img, points = load_data((img_path, gt_path))
        points = points.astype(float)

        if not self.train and self.eval_max_size > 0:
            img, points = resize_long_side(img, points, self.eval_max_size)

        if self.transform is not None:
            img = self.transform(img)
        img = torch.as_tensor(img, dtype=torch.float32)
        patch_size = random.choice(self.patch_size_choices) if self.train else int(self.patch_size)

        if self.train:
            img, points = safe_random_scale(img, points, patch_size)

            img, points = random_crop_with_retries(
                img,
                points,
                patch_size=patch_size,
                attempts=self.crop_attempts,
                min_points=self.min_crop_points,
            )

        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            if points.shape[0] > 0:
                points[:, 1] = (img.shape[2] - 1) - points[:, 1]

        target = {
            'points': torch.as_tensor(points, dtype=torch.float32),
            'labels': torch.ones([points.shape[0]]).long(),
        }

        if self.train:
            target['density'] = self.compute_density(points)
        else:
            target['image_path'] = img_path

        return img, target


def find_split_dir(data_root, split_name):
    candidates = [
        os.path.join(data_root, split_name),
        os.path.join(data_root, split_name.lower()),
        os.path.join(data_root, split_name.upper()),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def find_annotation_path(img_path):
    img_dir = os.path.dirname(img_path)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    candidates = [
        os.path.join(img_dir, f'{stem}_ann.mat'),
        os.path.join(img_dir, f'{stem}.mat'),
        os.path.join(img_dir, f'GT_{stem}.mat'),
        os.path.join(img_dir, 'ground_truth', f'{stem}_ann.mat'),
        os.path.join(img_dir, 'ground_truth', f'GT_{stem}.mat'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def load_data(img_gt_path):
    img_path, gt_path = img_gt_path
    try:
        img = load_rgb_image(img_path)
    except OSError as exc:
        raise FileNotFoundError(f'Could not read image: {img_path}')
    width, height = img.size
    points = load_points(gt_path, image_size=(height, width))
    if points.shape[0] == 0:
        raise ValueError(f'Annotation file has zero valid points: {gt_path}')
    return img, points


def resize_long_side(img, points, max_size):
    width, height = img.size
    factor = max(width / float(max_size), height / float(max_size), 1.0)
    if factor <= 1.0:
        return img, points
    new_width = max(1, int(width / factor))
    new_height = max(1, int(height / factor))
    img = img.resize((new_width, new_height), Image.BILINEAR)
    if points.shape[0] > 0:
        points = points / factor
    return img, points


def _normalize_points_array(value):
    points = np.asarray(value)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    candidates = (points, np.squeeze(points))
    for candidate in candidates:
        if candidate.ndim != 2:
            continue
        if candidate.shape[1] == 2:
            return candidate
        if candidate.shape[0] == 2:
            return candidate.T
    return None


def _find_points_array(value):
    if isinstance(value, np.ndarray):
        points = _normalize_points_array(value)
        if points is not None and np.issubdtype(points.dtype, np.number):
            return points
        if value.dtype.names is not None:
            for field_name in value.dtype.names:
                found = _find_points_array(value[field_name])
                if found is not None:
                    return found
        if value.dtype == object or value.dtype.names is not None:
            for item in value.flat:
                found = _find_points_array(item)
                if found is not None:
                    return found
    elif isinstance(value, np.void):
        for field_name in value.dtype.names or ():
            found = _find_points_array(value[field_name])
            if found is not None:
                return found
    elif hasattr(value, '_fieldnames'):
        for field_name in value._fieldnames:
            found = _find_points_array(getattr(value, field_name))
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_points_array(item)
            if found is not None:
                return found
    return None


def load_points(gt_path, image_size=None):
    if not gt_path or not os.path.exists(gt_path):
        return np.empty((0, 2), dtype=np.float32)

    try:
        mat = io.loadmat(gt_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f'Could not read annotation file {gt_path}: {exc}') from exc

    points = None
    if 'annPoints' in mat:
        points = _normalize_points_array(mat['annPoints'])
    if points is None:
        for key, value in mat.items():
            if key.startswith('__'):
                continue
            points = _find_points_array(value)
            if points is not None:
                break
    if points is None:
        raise ValueError(f'Could not find Nx2 point array in annotation file {gt_path}')

    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    # UCF-QNRF stores points as (x, y); PET targets use (y, x).
    points = points.reshape(-1, 2)[:, ::-1].copy()

    if image_size is not None and points.shape[0] > 0:
        height, width = image_size
        keep = (
            (points[:, 0] >= 0)
            & (points[:, 0] < height)
            & (points[:, 1] >= 0)
            & (points[:, 1] < width)
        )
        points = points[keep]

    return points.astype(np.float32, copy=False)


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    data_root = args.data_path
    eval_max_size = int(getattr(args, 'eval_max_size', -1))
    if eval_max_size < 0:
        eval_max_size = 1536
    if image_set == 'train':
        return QNRF(
            data_root,
            train=True,
            transform=transform,
            flip=True,
            patch_size=getattr(args, 'patch_size_choices', '') or args.patch_size,
            crop_attempts=getattr(args, 'crop_attempts', 1),
            min_crop_points=getattr(args, 'min_crop_points', 0),
        )
    if image_set == 'val':
        return QNRF(
            data_root,
            train=False,
            transform=transform,
            eval_max_size=eval_max_size,
        )
    raise ValueError(f'Unsupported image_set for QNRF: {image_set}')
