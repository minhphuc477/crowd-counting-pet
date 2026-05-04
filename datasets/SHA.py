import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import scipy.io as io
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
import warnings
warnings.filterwarnings('ignore')

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


class SHA(Dataset):
    def __init__(
        self,
        data_root,
        transform=None,
        train=False,
        flip=False,
        patch_size=256,
        crop_attempts=8,
        min_crop_points=1,
    ):
        self.root_path = data_root
        
        prefix = "train_data" if train else "test_data"
        self.prefix = prefix
        img_dir = os.path.join(data_root, prefix, "images")
        gt_dir = os.path.join(data_root, prefix, "ground-truth")
        img_names = [
            img_name for img_name in os.listdir(img_dir)
            if img_name.lower().endswith(IMAGE_EXTENSIONS)
        ]

        # get image and ground-truth list
        self.gt_list = {}
        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)
            stem = os.path.splitext(img_name)[0]
            gt_path = os.path.join(gt_dir, f"GT_{stem}.mat")
            self.gt_list[img_path] = gt_path
        self.img_list = sorted(list(self.gt_list.keys()))
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = patch_size
        self.crop_attempts = max(1, int(crop_attempts))
        self.min_crop_points = max(0, int(min_crop_points))
    
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

        # random scale
        if self.train:
            scale_range = [0.8, 1.2]           
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            
            # interpolation
            if scale * min_size > self.patch_size:  
                img = F.interpolate(
                    img.unsqueeze(0),
                    scale_factor=scale,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
                if points.shape[0] > 0:
                    points *= scale

        # random crop patch
        if self.train:
            img, points = random_crop_with_retries(
                img,
                points,
                patch_size=self.patch_size,
                attempts=self.crop_attempts,
                min_points=self.min_crop_points,
            )

        # random flip
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            if points.shape[0] > 0:
                points[:, 1] = (img.shape[2] - 1) - points[:, 1]

        # target
        target = {}
        target['points'] = torch.as_tensor(points, dtype=torch.float32)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target['density'] = density

        if not self.train:
            target['image_path'] = img_path

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {img_path}')
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = load_points(gt_path)
    return img, points


def load_points(gt_path):
    if not os.path.exists(gt_path):
        return np.empty((0, 2), dtype=np.float32)

    try:
        points = io.loadmat(gt_path)['image_info'][0][0][0][0][0]
    except (KeyError, IndexError, TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    points = points.reshape(-1, 2)
    return points[:, ::-1].copy()


def random_crop(img, points, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size
    
    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (points[:, 0] >= start_h) & (points[:, 0] < end_h) & (points[:, 1] >= start_w) & (points[:, 1] < end_w)

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
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
    if result_points.shape[0] > 0:
        result_points[:, 0] *= fH
        result_points[:, 1] *= fW
    return result_img, result_points


def random_crop_with_retries(img, points, patch_size=256, attempts=8, min_points=1):
    if points.shape[0] == 0 or min_points <= 0:
        return random_crop(img, points, patch_size)

    best_img, best_points = None, None
    best_count = -1
    for _ in range(max(1, attempts)):
        crop_img, crop_points = random_crop(img, points, patch_size)
        crop_count = crop_points.shape[0]
        if crop_count >= min_points:
            return crop_img, crop_points
        if crop_count > best_count:
            best_img, best_points = crop_img, crop_points
            best_count = crop_count

    return best_img, best_points


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
            transform=transform,
            flip=True,
            patch_size=args.patch_size,
            crop_attempts=getattr(args, 'crop_attempts', 8),
            min_crop_points=getattr(args, 'min_crop_points', 1),
        )
        return train_set
    elif image_set == 'val':
        val_set = SHA(data_root, train=False, transform=transform, patch_size=args.patch_size)
        return val_set
