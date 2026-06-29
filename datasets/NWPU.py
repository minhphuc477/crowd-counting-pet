import json
import os
import random
from pathlib import Path

import numpy as np
import scipy.io as io
import torch
import torchvision.transforms as standard_transforms
from PIL import Image
from torch.utils.data import Dataset

from .image_io import load_rgb_image
from .SHA import IMAGE_EXTENSIONS, random_crop, random_crop_with_retries, safe_random_scale
from .SHA import _parse_patch_size_choices


POINT_KEYS = (
    'points', 'point', 'annPoints', 'annpoints', 'locations', 'loc', 'pts',
    'human_points', 'head_points',
)
SIGMA_KEYS = ('sigma', 'sigmas', 'threshold', 'thresholds', 'sigma_s_l')
BOX_KEYS = ('boxes', 'box', 'bbox', 'bboxes', 'annBoxes', 'annboxes')


class NWPU(Dataset):
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
        eval_max_size=1536,
        sigma_mode='area',
        dense_crop_prob=0.0,
        dense_crop_attempts=16,
    ):
        self.root_path = str(data_root)
        self.data_root = Path(data_root)
        self.split = split
        self.train = train
        self.transform = transform
        self.flip = flip
        self.patch_size = patch_size
        self.patch_size_choices = _parse_patch_size_choices(patch_size)
        self.crop_attempts = max(1, int(crop_attempts))
        self.min_crop_points = max(0, int(min_crop_points))
        self.eval_max_size = int(eval_max_size) if eval_max_size is not None else 1536
        self.sigma_mode = sigma_mode
        self.dense_crop_prob = max(0.0, min(1.0, float(dense_crop_prob)))
        self.dense_crop_attempts = max(1, int(dense_crop_attempts))
        self.official_sigma = load_official_localization_sigma(
            self.data_root / f'{split}_gt_loc.txt'
        )

        self.images_dir = find_dir(self.data_root, ('images', 'Images'))
        self.jsons_dir = find_optional_dir(self.data_root, ('jsons', 'json', 'Jsons'))
        self.mats_dir = find_optional_dir(self.data_root, ('mats', 'mat', 'Mats'))
        if self.images_dir is None:
            raise FileNotFoundError(f'Could not find NWPU images directory under {self.data_root}')

        self.ids = read_split_ids(self.data_root, split)
        if not self.ids:
            self.ids = [
                path.stem for path in sorted(self.images_dir.iterdir())
                if path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        if not self.ids:
            raise FileNotFoundError(f'No NWPU image ids found for split={split} in {self.data_root}')

        image_index = build_image_index(self.data_root)
        json_index = build_annotation_index(self.data_root, '.json')
        mat_index = build_annotation_index(self.data_root, '.mat')

        self.samples = []
        missing = []
        for image_id in self.ids:
            image_path = find_image_path(self.images_dir, image_id)
            if image_path is None:
                image_path = image_index.get(Path(str(image_id)).stem)
            if image_path is None:
                missing.append(str(self.images_dir / f'{image_id}.jpg'))
                continue
            json_path = find_annotation_file(self.jsons_dir, image_id, '.json')
            mat_path = find_annotation_file(self.mats_dir, image_id, '.mat')
            if json_path is None:
                json_path = json_index.get(Path(str(image_id)).stem)
            if mat_path is None:
                mat_path = mat_index.get(Path(str(image_id)).stem)
            if json_path is None and mat_path is None:
                missing.append(str(self.data_root / f'{image_id}.json/.mat'))
                continue
            self.samples.append((image_id, image_path, json_path, mat_path))

        if missing:
            raise FileNotFoundError(
                f'Missing {len(missing)} NWPU file(s) for split={split}. '
                f'First missing file: {missing[0]}'
            )
        if not self.samples:
            raise FileNotFoundError(f'No usable NWPU samples found for split={split} in {self.data_root}')

        self.img_list = [str(sample[1]) for sample in self.samples]
        self.nSamples = len(self.samples)
        self.sample_counts = None

    def __len__(self):
        return self.nSamples

    def get_sample_counts(self):
        if self.sample_counts is None:
            counts = []
            for _image_id, _img_path, json_path, mat_path in self.samples:
                ann = load_annotation(json_path=json_path, mat_path=mat_path, sigma_mode=self.sigma_mode)
                counts.append(int(ann['points'].shape[0]))
            self.sample_counts = counts
        return self.sample_counts

    def compute_density(self, points):
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            return dist.sort(dim=1)[0][:, 1].mean().reshape(-1)
        return torch.tensor(999.0).reshape(-1)

    def __getitem__(self, index):
        image_id, img_path, json_path, mat_path = self.samples[index]
        try:
            img = load_rgb_image(img_path)
        except OSError as exc:
            raise FileNotFoundError(f'Could not read image: {img_path}') from exc

        width, height = img.size
        ann = load_annotation(
            json_path=json_path,
            mat_path=mat_path,
            image_size=(height, width),
            sigma_mode=self.sigma_mode,
        )
        official_sigma = self.official_sigma.get(Path(str(image_id)).stem)
        if official_sigma is not None and official_sigma.shape[0] == ann['points'].shape[0]:
            ann['sigma'] = official_sigma.copy()
            ann['sigma_source'] = 'official_localization_file'
        points = ann['points'].astype(np.float32, copy=True)
        sigma = ann.get('sigma')

        if points.shape[0] == 0 and self.train:
            # NWPU has negative samples; PET training targets can represent them.
            points = np.empty((0, 2), dtype=np.float32)

        if not self.train and self.eval_max_size > 0:
            img, points, sigma = resize_long_side_with_sigma(img, points, sigma, self.eval_max_size)

        if self.transform is not None:
            img = self.transform(img)
        img = torch.as_tensor(img, dtype=torch.float32)
        patch_size = random.choice(self.patch_size_choices) if self.train else int(self.patch_size)

        if self.train:
            sigma_valid = sigma is not None and len(sigma) == points.shape[0]
            if sigma_valid:
                point_records = np.concatenate(
                    [points, np.asarray(sigma, dtype=np.float32)],
                    axis=1,
                )
            else:
                point_records = points
            img, point_records = safe_random_scale(img, point_records, patch_size)
            if point_records.shape[0] > 0 and random.random() < self.dense_crop_prob:
                img, point_records = max_count_random_crop(
                    img,
                    point_records,
                    patch_size=patch_size,
                    attempts=self.dense_crop_attempts,
                )
            else:
                img, point_records = random_crop_with_retries(
                    img,
                    point_records,
                    patch_size=patch_size,
                    attempts=self.crop_attempts,
                    min_points=self.min_crop_points,
                )
            points = point_records[:, :2]
            sigma = point_records[:, 2:] if sigma_valid else None

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
            if sigma is not None and len(sigma) == points.shape[0]:
                target['sigma'] = torch.as_tensor(sigma, dtype=torch.float32)
        else:
            target['image_path'] = str(img_path)
            target['image_id'] = image_id
            if sigma is not None and len(sigma) == points.shape[0]:
                target['sigma'] = torch.as_tensor(sigma, dtype=torch.float32)
                target['sigma_source'] = ann.get('sigma_source', 'unknown')

        return img, target


def find_dir(root, names):
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    return None


def find_optional_dir(root, names):
    found = find_dir(root, names)
    return found if found is not None else root


def read_split_ids(root, split):
    candidates = [
        root / f'{split}.txt',
        root / f'{split.lower()}.txt',
        root / f'{split.upper()}.txt',
    ]
    if split == 'val':
        candidates.extend([root / 'validation.txt', root / 'test.txt'])
    for path in candidates:
        if not path.exists():
            continue
        ids = []
        for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
            token = line.strip().split()
            if not token:
                continue
            ids.append(Path(token[0]).stem)
        return ids
    return []


def load_official_localization_sigma(path):
    """Read NWPU's ``*_gt_loc.txt`` per-head small/large radii."""
    path = Path(path)
    if not path.is_file():
        return {}
    sigma_by_id = {}
    for line_number, line in enumerate(
        path.read_text(encoding='utf-8', errors='ignore').splitlines(),
        start=1,
    ):
        fields = line.strip().split()
        if not fields:
            continue
        try:
            image_id = Path(fields[0]).stem
            count = int(fields[1])
            values = np.asarray([float(value) for value in fields[2:]], dtype=np.float32)
        except (IndexError, ValueError) as exc:
            raise ValueError(f'Invalid NWPU localization line {line_number} in {path}') from exc
        expected = count * 5
        if values.size != expected:
            raise ValueError(
                f'Invalid NWPU localization line {line_number} in {path}: '
                f'expected {expected} values after count, found {values.size}'
            )
        sigma_by_id[image_id] = (
            values.reshape(count, 5)[:, 2:4].copy()
            if count > 0
            else np.empty((0, 2), dtype=np.float32)
        )
    return sigma_by_id


def find_image_path(images_dir, image_id):
    stem = Path(str(image_id)).stem
    for suffix in IMAGE_EXTENSIONS:
        candidate = images_dir / f'{stem}{suffix}'
        if candidate.exists():
            return candidate
    for candidate in images_dir.glob(f'{stem}.*'):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def build_image_index(root):
    root = Path(root)
    index = {}
    if not root.exists():
        return index
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem, path)
    return index


def find_annotation_file(base_dir, image_id, suffix):
    if base_dir is None:
        return None
    stem = Path(str(image_id)).stem
    for candidate in (base_dir / f'{stem}{suffix}', base_dir / f'{int(stem):04d}{suffix}' if stem.isdigit() else None):
        if candidate is not None and candidate.exists():
            return candidate
    return None


def build_annotation_index(root, suffix):
    root = Path(root)
    index = {}
    if not root.exists():
        return index
    for path in root.rglob(f'*{suffix}'):
        if not path.is_file():
            continue
        stem = path.stem
        if stem.startswith('GT_'):
            stem = stem[3:]
        index.setdefault(stem, path)
    return index


def load_annotation(json_path=None, mat_path=None, image_size=None, sigma_mode='area'):
    data = None
    source_path = None
    if json_path is not None and Path(json_path).exists():
        with open(json_path, 'r', encoding='utf-8', errors='ignore') as handle:
            data = json.load(handle)
        source_path = json_path
    elif mat_path is not None and Path(mat_path).exists():
        data = io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        source_path = mat_path
    else:
        return {'points': np.empty((0, 2), dtype=np.float32)}

    points_xy = _find_named_array(data, POINT_KEYS, min_cols=2)
    boxes = _find_named_array(data, BOX_KEYS, min_cols=4)
    sigma = _find_named_array(data, SIGMA_KEYS, min_cols=1)

    if points_xy is None:
        records = _collect_instance_records(data)
        if records:
            points_xy = np.asarray([record[0] for record in records], dtype=np.float32)
            boxes_from_records = [record[1] for record in records if record[1] is not None]
            if boxes is None and len(boxes_from_records) == len(records):
                boxes = np.asarray(boxes_from_records, dtype=np.float32)

    points_xy = _normalize_points_array(points_xy)
    if points_xy is None:
        raise ValueError(f'Could not find NWPU point annotations in {source_path}')
    points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    points = points_xy[:, ::-1].copy()

    sigma_norm = _normalize_sigma_array(sigma, points.shape[0])
    sigma_source = 'annotation'
    if sigma_norm is None and boxes is not None:
        sigma_norm = _sigma_from_boxes(boxes, points.shape[0], mode=sigma_mode)
        sigma_source = f'box_derived:{sigma_mode}'

    if image_size is not None and points.shape[0] > 0:
        height, width = image_size
        keep = (
            (points[:, 0] >= 0)
            & (points[:, 0] < height)
            & (points[:, 1] >= 0)
            & (points[:, 1] < width)
        )
        points = points[keep]
        if sigma_norm is not None and len(sigma_norm) == len(keep):
            sigma_norm = sigma_norm[keep]

    result = {'points': points.astype(np.float32, copy=False)}
    if sigma_norm is not None and sigma_norm.shape[0] == points.shape[0]:
        result['sigma'] = sigma_norm.astype(np.float32, copy=False)
        result['sigma_source'] = sigma_source
    return result


def resize_long_side_with_sigma(img, points, sigma, max_size):
    width, height = img.size
    factor = max(width / float(max_size), height / float(max_size), 1.0)
    if factor <= 1.0:
        return img, points, sigma
    new_width = max(1, int(width / factor))
    new_height = max(1, int(height / factor))
    img = img.resize((new_width, new_height), Image.BILINEAR)
    if points.shape[0] > 0:
        points = points / factor
    if sigma is not None:
        sigma = sigma / factor
    return img, points, sigma


def max_count_random_crop(img, points, patch_size=256, attempts=16):
    """Return the densest crop among random candidates for high-density training."""
    best_img, best_points = None, None
    best_count = -1
    for _ in range(max(1, int(attempts))):
        crop_img, crop_points = random_crop(img, points, patch_size)
        crop_count = int(crop_points.shape[0])
        if crop_count > best_count:
            best_img, best_points = crop_img, crop_points
            best_count = crop_count
    return best_img, best_points


def _as_numeric_array(value):
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.dtype == object or arr.dtype.names is not None:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return arr


def _normalize_points_array(value):
    if value is None:
        return None
    arr = _as_numeric_array(value)
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if arr.ndim == 1 and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2:
        return None
    if arr.shape[1] >= 2:
        return arr[:, :2]
    if arr.shape[0] == 2:
        return arr.T
    return None


def _normalize_boxes_array(value):
    if value is None:
        return None
    arr = _as_numeric_array(value)
    if arr is None:
        return None
    arr = np.squeeze(arr)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if arr.ndim == 1 and arr.size % 4 == 0:
        arr = arr.reshape(-1, 4)
    if arr.ndim != 2:
        return None
    if arr.shape[1] >= 4:
        return arr[:, :4]
    if arr.shape[0] == 4:
        return arr.T
    return None


def _normalize_sigma_array(value, count):
    if value is None:
        return None
    arr = _as_numeric_array(value)
    if arr is None:
        return None
    arr = np.squeeze(arr).astype(np.float32)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        arr = np.full((count, 2), float(arr), dtype=np.float32)
    elif arr.ndim == 1:
        if arr.shape[0] == count:
            arr = np.stack([arr, arr], axis=1)
        elif arr.shape[0] == 2 and count == 1:
            arr = arr.reshape(1, 2)
        else:
            return None
    elif arr.ndim == 2:
        if arr.shape[0] == count and arr.shape[1] >= 2:
            arr = arr[:, :2]
        elif arr.shape[1] == count and arr.shape[0] >= 2:
            arr = arr[:2, :].T
        elif arr.shape[0] == count and arr.shape[1] == 1:
            arr = np.repeat(arr, 2, axis=1)
        else:
            return None
    else:
        return None
    return np.maximum(arr, 1.0).astype(np.float32, copy=False)


def _find_named_array(value, keys, min_cols=1):
    if isinstance(value, dict):
        for key in keys:
            for candidate_key, candidate_value in value.items():
                if str(candidate_key).lower() == key.lower():
                    normalized = _normalize_by_cols(candidate_value, min_cols)
                    if normalized is not None:
                        return normalized
        for candidate_value in value.values():
            found = _find_named_array(candidate_value, keys, min_cols=min_cols)
            if found is not None:
                return found
    elif isinstance(value, np.ndarray):
        if value.dtype.names is not None:
            for name in value.dtype.names:
                found = _find_named_array(value[name], keys, min_cols=min_cols)
                if found is not None:
                    return found
        if value.dtype == object:
            for item in value.flat:
                found = _find_named_array(item, keys, min_cols=min_cols)
                if found is not None:
                    return found
    elif hasattr(value, '_fieldnames'):
        for name in value._fieldnames:
            if name.lower() in [key.lower() for key in keys]:
                normalized = _normalize_by_cols(getattr(value, name), min_cols)
                if normalized is not None:
                    return normalized
        for name in value._fieldnames:
            found = _find_named_array(getattr(value, name), keys, min_cols=min_cols)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_named_array(item, keys, min_cols=min_cols)
            if found is not None:
                return found
    return None


def _normalize_by_cols(value, min_cols):
    if min_cols >= 4:
        return _normalize_boxes_array(value)
    if min_cols == 2:
        return _normalize_points_array(value)
    arr = _as_numeric_array(value)
    if arr is None:
        return None
    return np.asarray(arr)


def _collect_instance_records(value):
    records = []
    if isinstance(value, dict):
        point = _extract_point_from_record(value)
        if point is not None:
            records.append((point, _extract_box_from_record(value)))
        for item in value.values():
            records.extend(_collect_instance_records(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            records.extend(_collect_instance_records(item))
    return records


def _extract_point_from_record(record):
    for key in ('point', 'points', 'location', 'loc', 'head'):
        if key in record:
            point = _normalize_points_array(record[key])
            if point is not None and point.shape[0] >= 1:
                return point[0]
    x = record.get('x', record.get('X'))
    y = record.get('y', record.get('Y'))
    if x is not None and y is not None:
        return np.asarray([x, y], dtype=np.float32)
    return None


def _extract_box_from_record(record):
    for key in BOX_KEYS:
        if key in record:
            box = _normalize_boxes_array(record[key])
            if box is not None and box.shape[0] >= 1:
                return box[0]
    return None


def _sigma_from_boxes(boxes, count, mode='area'):
    boxes = _normalize_boxes_array(boxes)
    if boxes is None or boxes.shape[0] != count:
        return None
    if mode not in ('area', 'diag', 'min_diag', 'official'):
        raise ValueError(f'Unsupported NWPU sigma mode: {mode}')
    boxes = boxes.astype(np.float32)
    x1, y1, a, b = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    wh_xyxy = np.stack([np.maximum(a - x1, 1.0), np.maximum(b - y1, 1.0)], axis=1)
    wh_xywh = np.stack([np.maximum(a, 1.0), np.maximum(b, 1.0)], axis=1)
    use_xyxy = (a > x1) & (b > y1)
    wh = np.where(use_xyxy[:, None], wh_xyxy, wh_xywh)
    width = np.maximum(wh[:, 0], 1.0)
    height = np.maximum(wh[:, 1], 1.0)
    if mode == 'official':
        # NWPU's localization protocol uses integer radii derived from half
        # the annotated head-box size and half its diagonal.
        width = np.ceil(width)
        height = np.ceil(height)
        sigma_s = np.maximum(np.ceil(0.5 * np.minimum(width, height)), 1.0)
        sigma_l = np.maximum(
            np.ceil(0.5 * np.sqrt(width * width + height * height)),
            1.0,
        )
    elif mode == 'area':
        # Historical repo behavior: a lenient large threshold based on box area.
        # This is the default because official NWPU eval consumes prepared sigma
        # files; when those are absent, this fallback preserves prior results.
        scale = np.sqrt(np.maximum(width * height, 1.0))
        sigma_s = np.maximum(0.5 * scale, 1.0)
        sigma_l = np.maximum(scale, 1.0)
    elif mode == 'diag':
        diag = np.sqrt(width * width + height * height)
        sigma_s = np.maximum(0.5 * diag, 1.0)
        sigma_l = np.maximum(diag, 1.0)
    else:
        # Strict diagnostic mode. Useful for ablations, but not the default.
        sigma_s = np.maximum(0.5 * np.minimum(width, height), 1.0)
        sigma_l = np.maximum(0.5 * np.sqrt(width * width + height * height), 1.0)
    return np.stack([sigma_s, sigma_l], axis=1).astype(np.float32)


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
        return NWPU(
            args.data_path,
            split='train',
            train=True,
            transform=transform,
            flip=True,
            patch_size=getattr(args, 'patch_size_choices', '') or args.patch_size,
            crop_attempts=getattr(args, 'crop_attempts', 1),
            min_crop_points=getattr(args, 'min_crop_points', 0),
            eval_max_size=eval_max_size,
            sigma_mode=getattr(args, 'nwpu_sigma_mode', 'area'),
            dense_crop_prob=getattr(args, 'nwpu_dense_crop_prob', 0.0),
            dense_crop_attempts=getattr(args, 'nwpu_dense_crop_attempts', 16),
        )
    if image_set == 'train_eval':
        return NWPU(
            args.data_path,
            split='train',
            train=False,
            transform=transform,
            eval_max_size=0,
            sigma_mode=getattr(args, 'nwpu_sigma_mode', 'area'),
        )
    if image_set == 'val':
        split = getattr(args, 'nwpu_eval_split', 'val') or 'val'
        return NWPU(
            args.data_path,
            split=split,
            train=False,
            transform=transform,
            eval_max_size=eval_max_size,
            sigma_mode=getattr(args, 'nwpu_sigma_mode', 'area'),
        )
    raise ValueError(f'Unsupported image_set for NWPU: {image_set}')
