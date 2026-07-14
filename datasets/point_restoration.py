import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .image_io import load_rgb_image
from .QNRF import find_annotation_path, find_split_dir, load_raw_points_xy
from .SHA import IMAGE_EXTENSIONS


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def qnrf_train_samples(data_root):
    split_dir = find_split_dir(data_root, 'Train')
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f'Could not find QNRF training directory: {split_dir}')
    samples = []
    for name in sorted(os.listdir(split_dir)):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        image_path = os.path.join(split_dir, name)
        annotation_path = find_annotation_path(image_path)
        if annotation_path is None:
            raise FileNotFoundError(f'Could not find QNRF annotation for {image_path}')
        samples.append((image_path, annotation_path))
    if not samples:
        raise FileNotFoundError(f'No QNRF training images found in {split_dir}')
    return samples


def build_holdout_indices(num_samples, holdout_fraction, seed):
    fraction = float(holdout_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError('holdout_fraction must be in [0, 1)')
    if fraction == 0.0:
        return list(range(num_samples)), []
    if num_samples < 2:
        raise ValueError('a holdout split requires at least two samples')
    num_val = max(1, min(num_samples - 1, int(round(num_samples * fraction))))
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(num_samples, generator=generator).tolist()
    return sorted(permutation[num_val:]), sorted(permutation[:num_val])


def restoration_radii(points_xy, alpha=0.4):
    """Implement SAE Eq. (2) without constructing an O(N^2) matrix."""
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    count = points.shape[0]
    if count == 0:
        return np.empty((0,), dtype=np.float32)
    if not 0.0 < float(alpha) <= 0.5:
        raise ValueError('SAE alpha must be in (0, 0.5] to avoid overlapping shift regions')
    if count == 1:
        return np.full((1,), float(alpha) * 16.0, dtype=np.float32)

    neighbor_count = min(4, count)
    _, neighbor_indices = cKDTree(points).query(points, k=neighbor_count)
    if neighbor_count == 1:
        neighbor_indices = neighbor_indices[:, None]
    nearest_indices = neighbor_indices[:, 1]
    nearest = np.linalg.norm(points - points[nearest_indices], axis=1)
    if neighbor_count > 2:
        nearby_nearest = nearest[neighbor_indices[:, 1:]].mean(axis=1)
    else:
        nearby_nearest = nearest
    radius = float(alpha) * np.minimum(nearest, nearby_nearest)
    return np.maximum(radius, 1e-3).astype(np.float32, copy=False)


def _resize_for_crop(image, points_xy, radii, crop_size, scale_range):
    width, height = image.size
    scale = random.uniform(float(scale_range[0]), float(scale_range[1]))
    minimum_scale = (int(crop_size) + 8) / float(min(width, height))
    scale = max(scale, minimum_scale)
    new_width = max(int(crop_size), int(round(width * scale)))
    new_height = max(int(crop_size), int(round(height * scale)))
    image = image.resize((new_width, new_height), Image.BILINEAR)
    realized_x = new_width / float(width)
    realized_y = new_height / float(height)
    points = points_xy.copy()
    points[:, 0] *= realized_x
    points[:, 1] *= realized_y
    # Radius is isotropic; the geometric-mean realized scale avoids choosing
    # one rounded image axis as authoritative.
    radii = radii * float(np.sqrt(realized_x * realized_y))
    return image, points, radii


def _nonempty_random_crop(image, points_xy, radii, crop_size, attempts):
    width, height = image.size
    best = None
    for _ in range(max(1, int(attempts))):
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        keep = (
            (points_xy[:, 0] >= left)
            & (points_xy[:, 0] < left + crop_size)
            & (points_xy[:, 1] >= top)
            & (points_xy[:, 1] < top + crop_size)
        )
        candidate = (int(keep.sum()), left, top, keep)
        if best is None or candidate[0] > best[0]:
            best = candidate
        if candidate[0] > 0:
            break

    if best is None or best[0] == 0:
        point = points_xy[random.randrange(points_xy.shape[0])]
        left_min = max(0, int(np.ceil(point[0] - crop_size + 1)))
        left_max = min(int(np.floor(point[0])), width - crop_size)
        top_min = max(0, int(np.ceil(point[1] - crop_size + 1)))
        top_max = min(int(np.floor(point[1])), height - crop_size)
        left = random.randint(left_min, max(left_min, left_max))
        top = random.randint(top_min, max(top_min, top_max))
        keep = (
            (points_xy[:, 0] >= left)
            & (points_xy[:, 0] < left + crop_size)
            & (points_xy[:, 1] >= top)
            & (points_xy[:, 1] < top + crop_size)
        )
    else:
        _, left, top, keep = best

    image = image.crop((left, top, left + crop_size, top + crop_size))
    points = points_xy[keep].copy()
    points[:, 0] -= left
    points[:, 1] -= top
    return image, points, radii[keep].copy()


class QNRFShiftRestorationDataset(Dataset):
    """Generate SAE shifted-point supervision from QNRF training labels."""

    def __init__(
        self,
        data_root,
        indices=None,
        crop_size=512,
        alpha=0.4,
        scale_range=(0.7, 1.3),
        crop_attempts=8,
        flip=True,
    ):
        all_samples = qnrf_train_samples(data_root)
        if indices is None:
            indices = range(len(all_samples))
        self.samples = [all_samples[int(index)] for index in indices]
        self.crop_size = int(crop_size)
        if self.crop_size < 32:
            raise ValueError('crop_size must be at least 32')
        self.alpha = float(alpha)
        self.scale_range = tuple(float(value) for value in scale_range)
        if len(self.scale_range) != 2 or self.scale_range[0] <= 0 or self.scale_range[1] < self.scale_range[0]:
            raise ValueError('scale_range must contain two positive ascending values')
        self.crop_attempts = max(1, int(crop_attempts))
        self.flip = bool(flip)

        self.points_and_radii = []
        for _, annotation_path in self.samples:
            points = load_raw_points_xy(annotation_path)
            if points.shape[0] == 0:
                raise ValueError(f'QNRF restoration sample has no points: {annotation_path}')
            self.points_and_radii.append((points, restoration_radii(points, self.alpha)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, _ = self.samples[index]
        image = load_rgb_image(image_path)
        points, radii = self.points_and_radii[index]
        points = points.copy()
        radii = radii.copy()
        width, height = image.size
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)

        image, points, radii = _resize_for_crop(
            image,
            points,
            radii,
            self.crop_size,
            self.scale_range,
        )
        image, points, radii = _nonempty_random_crop(
            image,
            points,
            radii,
            self.crop_size,
            self.crop_attempts,
        )

        if self.flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            points[:, 0] = (self.crop_size - 1) - points[:, 0]

        angle = np.random.uniform(0.0, 2.0 * np.pi, size=points.shape[0])
        magnitude = np.random.uniform(0.0, 1.0, size=points.shape[0]) * radii
        shift_xy = np.stack((magnitude * np.cos(angle), magnitude * np.sin(angle)), axis=1)
        shifted = points + shift_xy
        shifted[:, 0] = np.clip(shifted[:, 0], 0, self.crop_size - 1)
        shifted[:, 1] = np.clip(shifted[:, 1], 0, self.crop_size - 1)
        inverse_shift = points - shifted

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        return {
            'image': image_tensor,
            'shifted_points_yx': torch.from_numpy(shifted[:, ::-1].copy()).float(),
            'inverse_shift_yx': torch.from_numpy(inverse_shift[:, ::-1].copy()).float(),
            'image_path': image_path,
        }


def restoration_collate(batch):
    return torch.stack([sample['image'] for sample in batch]), batch
