import os
import hashlib
import math
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as io
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
from .image_io import load_rgb_image

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
GT_DIR_NAMES = ('ground-truth', 'ground_truth', 'groundtruth')


class SHA(Dataset):
    def __init__(
        self,
        data_root,
        transform=None,
        train=False,
        source_split=None,
        flip=False,
        patch_size=256,
        crop_attempts=1,
        min_crop_points=0,
        no_random_scale=False,
        partial_annotation_ratio=1.0,
        partial_annotation_seed=0,
        partial_annotation_height_ratio=0.5,
        annotation_override_dir='',
    ):
        self.root_path = data_root

        if source_split is None:
            source_split = 'train' if train else 'test'
        if source_split not in ('train', 'test'):
            raise ValueError("source_split must be 'train' or 'test'")
        prefix = "train_data" if source_split == 'train' else "test_data"
        self.prefix = prefix
        split_dir = os.path.join(data_root, prefix)
        img_dir = os.path.join(split_dir, "images")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f'Could not find ShanghaiTech image directory: {img_dir}')
        gt_dir = find_ground_truth_dir(split_dir)
        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(f'Could not find ShanghaiTech annotation directory: {gt_dir}')
        img_names = [
            img_name for img_name in os.listdir(img_dir)
            if img_name.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if not img_names:
            raise FileNotFoundError(f'No image files found in {img_dir}')

        # get image and ground-truth list
        self.gt_list = {}
        missing_gt = []
        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)
            stem = os.path.splitext(img_name)[0]
            gt_path = os.path.join(gt_dir, f"GT_{stem}.mat")
            if annotation_override_dir and source_split == 'train':
                gt_path = os.path.join(
                    annotation_override_dir,
                    f"GT_{stem}.mat",
                )
            if not os.path.exists(gt_path):
                missing_gt.append(gt_path)
            self.gt_list[img_path] = gt_path
        self.img_list = sorted(list(self.gt_list.keys()))
        self.nSamples = len(self.img_list)
        if missing_gt:
            raise FileNotFoundError(
                f'Missing {len(missing_gt)} ShanghaiTech annotation file(s) for {prefix}. '
                f'First missing file: {missing_gt[0]}'
            )

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = patch_size
        self.patch_size_choices = _parse_patch_size_choices(patch_size)
        self.crop_attempts = max(1, int(crop_attempts))
        self.min_crop_points = max(0, int(min_crop_points))
        self.no_random_scale = bool(no_random_scale)
        self.partial_annotation_ratio = float(partial_annotation_ratio)
        if not 0.0 < self.partial_annotation_ratio <= 1.0:
            raise ValueError('partial_annotation_ratio must be in (0, 1]')
        self.partial_annotation_seed = int(partial_annotation_seed)
        self.partial_annotation_height_ratio = float(
            partial_annotation_height_ratio
        )
        if not 0.0 < self.partial_annotation_height_ratio <= 1.0:
            raise ValueError(
                'partial_annotation_height_ratio must be in (0, 1]'
            )
    
    def compute_density(self, points):
        """
        Compute crowd density:
            - defined as the average nearest distance between ground-truth points
        """
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:,1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        # load image and gt points
        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]
        img, points = load_data((img_path, gt_path), self.train)
        points = points.astype(float)

        # image transform
        if self.transform is not None:
            img = self.transform(img)
        img = torch.as_tensor(img, dtype=torch.float32)
        supervision_mask = None
        if self.train and self.partial_annotation_ratio < 1.0:
            supervision_mask, bounds = fixed_partial_annotation_mask(
                img.shape[-2:],
                img_path,
                ratio=self.partial_annotation_ratio,
                seed=self.partial_annotation_seed,
                height_ratio=self.partial_annotation_height_ratio,
            )
            top, left, bottom, right = bounds
            keep = (
                (points[:, 0] >= top)
                & (points[:, 0] < bottom)
                & (points[:, 1] >= left)
                & (points[:, 1] < right)
            )
            points = points[keep]
        patch_size = random.choice(self.patch_size_choices) if self.train else int(self.patch_size)

        # random scale
        if self.train and not self.no_random_scale:
            if supervision_mask is None:
                img, points = safe_random_scale(img, points, patch_size)
            else:
                img, points, supervision_mask = safe_random_scale(
                    img,
                    points,
                    patch_size,
                    spatial_mask=supervision_mask,
                )

        # crop/resize patch
        if self.train:
            if supervision_mask is None:
                img, points = random_crop_with_retries(
                    img,
                    points,
                    patch_size=patch_size,
                    attempts=self.crop_attempts,
                    min_points=self.min_crop_points,
                )
            else:
                img, points, supervision_mask = random_crop_with_retries(
                    img,
                    points,
                    patch_size=patch_size,
                    attempts=self.crop_attempts,
                    min_points=self.min_crop_points,
                    spatial_mask=supervision_mask,
                )
        # NOTE: validation/test images are kept at full resolution.
        # PET is fully convolutional and handles arbitrary image sizes at inference.
        # Resizing test images to a fixed patch_size was destroying crowd density
        # distributions and caused a ~145 MAE plateau.

        # random flip
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            if supervision_mask is not None:
                supervision_mask = torch.flip(
                    supervision_mask,
                    dims=[1],
                )
            if points.shape[0] > 0:
                points[:, 1] = (img.shape[2] - 1) - points[:, 1]

        # target
        target = {}
        target['points'] = torch.as_tensor(points, dtype=torch.float32)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target['density'] = density
            if supervision_mask is not None:
                target['supervision_mask'] = supervision_mask.to(
                    dtype=torch.bool
                )

        if not self.train:
            target['image_path'] = img_path

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    try:
        img = load_rgb_image(img_path)
    except OSError as exc:
        raise FileNotFoundError(f'Could not read image: {img_path}') from exc
    width, height = img.size
    points = load_points(gt_path, image_size=(height, width))
    if points.shape[0] == 0:
        raise ValueError(f'Annotation file has zero valid points: {gt_path}')
    return img, points


def find_ground_truth_dir(split_dir):
    for gt_name in GT_DIR_NAMES:
        gt_dir = os.path.join(split_dir, gt_name)
        if os.path.isdir(gt_dir):
            return gt_dir
    return os.path.join(split_dir, GT_DIR_NAMES[0])


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


def _original_shanghai_points(image_info):
    try:
        return _normalize_points_array(image_info[0][0][0][0][0])
    except (IndexError, TypeError, ValueError):
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
    if not os.path.exists(gt_path):
        return np.empty((0, 2), dtype=np.float32)

    try:
        mat = io.loadmat(gt_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f'Could not read annotation file {gt_path}: {exc}') from exc

    points = None
    if 'image_info' in mat:
        points = _original_shanghai_points(mat['image_info'])
        if points is None:
            points = _find_points_array(mat['image_info'])
    if points is None and 'annPoints' in mat:
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
    points = points.reshape(-1, 2)
    points = points[:, ::-1].copy()

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


def random_crop(img, points, patch_size=256, spatial_mask=None):
    patch_h = patch_size
    patch_w = patch_size
    
    # random crop
    if spatial_mask is not None and spatial_mask.any():
        annotated = torch.nonzero(spatial_mask, as_tuple=False)
        anchor = annotated[random.randrange(annotated.shape[0])]
        max_start_h = max(img.size(1) - patch_h, 0)
        max_start_w = max(img.size(2) - patch_w, 0)
        start_h = min(
            max(int(anchor[0]) - random.randint(0, patch_h - 1), 0),
            max_start_h,
        )
        start_w = min(
            max(int(anchor[1]) - random.randint(0, patch_w - 1), 0),
            max_start_w,
        )
    else:
        start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
        start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (points[:, 0] >= start_h) & (points[:, 0] < end_h) & (points[:, 1] >= start_w) & (points[:, 1] < end_w)

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_mask = None
    if spatial_mask is not None:
        result_mask = spatial_mask[start_h:end_h, start_w:end_w]
    result_points = points[idx].copy()
    if result_points.shape[0] > 0:
        result_points[:, 0] -= start_h
        result_points[:, 1] -= start_w
    
    # resize to patchsize
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h/imgH, patch_w/imgW
    result_img = F.interpolate(
        result_img.unsqueeze(0),
        (patch_h, patch_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)
    if result_mask is not None:
        result_mask = F.interpolate(
            result_mask[None, None].float(),
            (patch_h, patch_w),
            mode='nearest',
        )[0, 0] >= 0.5
    if result_points.shape[0] > 0:
        result_points[:, 0] *= fH
        result_points[:, 1] *= fW
        # Optional per-point scale metadata follows y/x in NWPU training.
        if result_points.shape[1] > 2:
            result_points[:, 2:] *= math.sqrt(fH * fW)
    if result_mask is not None:
        return result_img, result_points, result_mask
    return result_img, result_points


def safe_random_scale(
    img,
    points,
    patch_size=256,
    scale_range=(0.8, 1.2),
    spatial_mask=None,
):
    """Apply scale jitter only when the scaled image can still supply a crop."""
    scale = random.uniform(*scale_range)
    min_size = min(img.shape[1:])
    if scale * min_size >= patch_size:
        img = F.interpolate(
            img.unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        if spatial_mask is not None:
            spatial_mask = F.interpolate(
                spatial_mask[None, None].float(),
                size=img.shape[-2:],
                mode='nearest',
            )[0, 0] >= 0.5
        if points.shape[0] > 0:
            points = points * scale
    if spatial_mask is not None:
        return img, points, spatial_mask
    return img, points


def random_crop_with_retries(
    img,
    points,
    patch_size=256,
    attempts=8,
    min_points=1,
    spatial_mask=None,
):
    if points.shape[0] == 0 or min_points <= 0:
        return random_crop(
            img,
            points,
            patch_size,
            spatial_mask=spatial_mask,
        )

    best_img, best_points, best_mask = None, None, None
    best_count = -1
    for _ in range(max(1, attempts)):
        crop_result = random_crop(
            img,
            points,
            patch_size,
            spatial_mask=spatial_mask,
        )
        if spatial_mask is None:
            crop_img, crop_points = crop_result
            crop_mask = None
        else:
            crop_img, crop_points, crop_mask = crop_result
        crop_count = crop_points.shape[0]
        if crop_count >= min_points:
            return crop_result
        if crop_count > best_count:
            best_img, best_points, best_mask = (
                crop_img,
                crop_points,
                crop_mask,
            )
            best_count = crop_count

    if spatial_mask is not None:
        return best_img, best_points, best_mask
    return best_img, best_points


def fixed_partial_annotation_mask(
    image_shape,
    image_key,
    ratio,
    seed=0,
    height_ratio=0.5,
):
    """Create the fixed rectangular annotation region used by PAL/PET."""
    height, width = (int(image_shape[0]), int(image_shape[1]))
    ratio = float(ratio)
    region_h_ratio = min(1.0, max(ratio, float(height_ratio)))
    region_w_ratio = min(1.0, ratio / region_h_ratio)
    region_h = min(height, max(1, int(round(height * region_h_ratio))))
    region_w = min(width, max(1, int(round(width * region_w_ratio))))

    digest = hashlib.sha256(
        f'{seed}:{stable_image_key(image_key)}'.encode('utf-8')
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], byteorder='big'))
    top = rng.randint(0, height - region_h)
    left = rng.randint(0, width - region_w)
    bottom = top + region_h
    right = left + region_w
    mask = torch.zeros((height, width), dtype=torch.bool)
    mask[top:bottom, left:right] = True
    return mask, (top, left, bottom, right)


def stable_image_key(path):
    return os.path.normcase(os.path.basename(os.path.normpath(str(path))))


def _parse_patch_size_choices(patch_size):
    if isinstance(patch_size, str):
        raw_values = [value.strip() for value in patch_size.replace(';', ',').split(',')]
        choices = [int(value) for value in raw_values if value]
    elif isinstance(patch_size, (list, tuple)):
        choices = [int(value) for value in patch_size]
    else:
        choices = [int(patch_size)]
    choices = sorted({value for value in choices if value > 0})
    if not choices:
        raise ValueError('patch_size must contain at least one positive integer')
    return choices


# center_crop removed: validation images are evaluated at full resolution.


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    
    data_root = args.data_path
    if image_set == 'train':
        train_set = SHA(
            data_root,
            train=True,
            source_split='train',
            transform=transform,
            flip=True,
            patch_size=getattr(args, 'patch_size_choices', '') or args.patch_size,
            crop_attempts=getattr(args, 'crop_attempts', 1),
            min_crop_points=getattr(args, 'min_crop_points', 0),
            no_random_scale=getattr(args, 'no_random_scale', False),
            partial_annotation_ratio=getattr(
                args,
                'partial_annotation_ratio',
                1.0,
            ),
            partial_annotation_seed=getattr(
                args,
                'partial_annotation_seed',
                0,
            ),
            partial_annotation_height_ratio=getattr(
                args,
                'partial_annotation_height_ratio',
                0.5,
            ),
            annotation_override_dir=getattr(
                args,
                'annotation_override_dir',
                '',
            ),
        )
        return train_set
    elif image_set == 'train_eval':
        return SHA(
            data_root,
            train=False,
            source_split='train',
            transform=transform,
        )
    elif image_set == 'val':
        # Evaluate at full resolution — PET is fully convolutional
        val_set = SHA(
            data_root,
            train=False,
            source_split='test',
            transform=transform,
        )
        return val_set
